from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from active_wasserstein.measures import EmpiricalMeasure
from active_wasserstein.utils import TimeGrid

Array = np.ndarray


def smooth_transition(t: Array, a: float, b: float) -> Array:
    """Clamp((t - a) / (b - a), 0, 1) with broadcasting support."""
    t_arr = np.asarray(t, dtype=float)
    if b <= a:
        raise ValueError("transition requires b > a")
    return np.clip((t_arr - a) / (b - a), 0.0, 1.0)


def orthonormal_matrix(
    obs_dim: int, latent_dim: int, rng: np.random.Generator
) -> Array:
    """Return a (obs_dim, latent_dim) matrix with orthonormal columns."""
    if obs_dim < latent_dim:
        raise ValueError("obs_dim must be >= latent_dim")
    mat = rng.normal(size=(obs_dim, latent_dim))
    q, _ = np.linalg.qr(mat)
    return q[:, :latent_dim]


@dataclass
class SequentialBranchingTrajectory:
    """Sequential branching SDE with a high-dimensional observation model."""

    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.005

    kappa: float = 10.0
    sigma_diff: float = 0.2
    init_sigma: float = 0.05

    obs_dim: int = 10
    latent_dim: int = 2
    obs_noise: float = 0.05

    branch_scale: float = 2.0
    event1: tuple[float, float] = (0.3, 0.4)
    event2: tuple[float, float] = (0.7, 0.8)
    split_branch_sign: int = -1
    time_tolerance: float = 1e-6

    q_seed: Optional[int] = None
    q_matrix: Optional[Array] = None
    _persistent_cache: dict[tuple[int, bool], dict[str, Array]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError("t_end must exceed t_start")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.sigma_diff < 0:
            raise ValueError("sigma_diff must be nonnegative")
        if self.init_sigma <= 0:
            raise ValueError("init_sigma must be positive")
        if self.obs_noise < 0:
            raise ValueError("obs_noise must be nonnegative")
        if self.latent_dim != 2:
            raise ValueError("latent_dim must be 2 for this trajectory")
        if self.obs_dim < self.latent_dim:
            raise ValueError("obs_dim must be >= latent_dim")
        if self.branch_scale <= 0:
            raise ValueError("branch_scale must be positive")
        if self.event1[1] <= self.event1[0]:
            raise ValueError("event1 must satisfy (start, end) with end > start")
        if self.event2[1] <= self.event2[0]:
            raise ValueError("event2 must satisfy (start, end) with end > start")
        if self.split_branch_sign not in (-1, 1):
            raise ValueError("split_branch_sign must be -1 or 1")

        self._validate_time_grid()
        self._set_observation_matrix()

    def _validate_time_grid(self) -> None:
        duration = self.t_end - self.t_start
        steps = duration / self.dt
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("dt must evenly divide (t_end - t_start)")

    def _set_observation_matrix(self) -> None:
        if self.q_matrix is None:
            rng = np.random.default_rng(self.q_seed)
            self.q_matrix = orthonormal_matrix(self.obs_dim, self.latent_dim, rng)
        else:
            q = np.asarray(self.q_matrix, dtype=float)
            if q.shape != (self.obs_dim, self.latent_dim):
                raise ValueError("q_matrix must have shape (obs_dim, latent_dim)")
            eye = np.eye(self.latent_dim)
            if not np.allclose(q.T @ q, eye, atol=1e-6):
                raise ValueError("q_matrix columns must be orthonormal")
            self.q_matrix = q

    def time_grid(self, num: int | None = None) -> TimeGrid:
        if num is None:
            num = self.num_timepoints()
        return TimeGrid(self.t_start, self.t_end, int(num))

    def num_timepoints(self) -> int:
        duration = self.t_end - self.t_start
        num_steps = int(round(duration / self.dt))
        return num_steps + 1

    def observation_times(self) -> Array:
        num_steps = self.num_timepoints() - 1
        return self.t_start + self.dt * np.arange(num_steps + 1, dtype=float)

    def target_mean(self, t: float, y_fate: Array, x_fate: Array) -> Array:
        s1 = smooth_transition(t, self.event1[0], self.event1[1])
        s2 = smooth_transition(t, self.event2[0], self.event2[1])
        mu_y = y_fate * self.branch_scale * s1
        gate = (y_fate == self.split_branch_sign).astype(float)
        mu_x = gate * x_fate * self.branch_scale * s2
        return np.stack([mu_x, mu_y], axis=-1)

    def observe(
        self, latent: Array, rng: np.random.Generator, noisy: bool = True
    ) -> Array:
        obs = latent @ self.q_matrix.T
        if noisy and self.obs_noise > 0:
            obs = obs + rng.normal(
                scale=self.obs_noise, size=(latent.shape[0], self.obs_dim)
            )
        return obs

    def _euler_maruyama_step(
        self,
        latent: Array,
        mu: Array,
        dt: float,
        rng: np.random.Generator,
    ) -> Array:
        noise = rng.normal(scale=np.sqrt(dt), size=latent.shape)
        drift = -self.kappa * (latent - mu)
        return latent + drift * dt + self.sigma_diff * noise

    def _simulate_observations(
        self,
        times: Array,
        n: int,
        rng: np.random.Generator,
        noisy: bool,
    ) -> Array:
        times_arr = np.asarray(times, dtype=float)
        if times_arr.ndim != 1:
            raise ValueError("times must be a 1D array")
        if np.any(times_arr < self.t_start) or np.any(times_arr > self.t_end):
            raise ValueError("times must lie within [t_start, t_end]")
        if np.any(np.diff(times_arr) < 0):
            raise ValueError("times must be sorted ascending")

        y_fate = rng.choice([-1, 1], size=n)
        x_fate = rng.choice([-1, 1], size=n)
        latent = rng.normal(scale=self.init_sigma, size=(n, self.latent_dim))

        obs = np.empty((len(times_arr), n, self.obs_dim))
        obs[0] = self.observe(latent, rng, noisy=noisy)

        current_time = float(times_arr[0])
        for idx in range(1, len(times_arr)):
            target_time = float(times_arr[idx])
            while current_time + self.dt < target_time - 1e-12:
                mu = self.target_mean(current_time, y_fate, x_fate)
                latent = self._euler_maruyama_step(latent, mu, self.dt, rng)
                current_time += self.dt
            remainder = target_time - current_time
            if remainder > 1e-12:
                mu = self.target_mean(current_time, y_fate, x_fate)
                latent = self._euler_maruyama_step(latent, mu, remainder, rng)
                current_time = target_time
            obs[idx] = self.observe(latent, rng, noisy=noisy)
        return obs

    def _build_persistent_cache(
        self,
        times: Array,
        n: int,
        seed: int,
        noisy: bool,
    ) -> None:
        times_arr = np.asarray(times, dtype=float)
        rng = np.random.default_rng(int(seed))
        obs = self._simulate_observations(times_arr, n=n, rng=rng, noisy=noisy)
        self._persistent_cache[(n, noisy)] = {
            "times": times_arr,
            "obs": obs,
            "seed": int(seed),
        }

    def prepare_persistent_cache(
        self,
        times: Array,
        n: int,
        rng: Optional[np.random.Generator] = None,
        noisy: bool = True,
        seed: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """Precompute persistent snapshots for a set of times."""
        if n <= 0:
            raise ValueError("n must be positive")
        times_arr = np.unique(np.asarray(times, dtype=float))
        if times_arr.size == 0:
            raise ValueError("times must be non-empty")
        key = (int(n), bool(noisy))
        cache = self._persistent_cache.get(key)
        if cache is not None and not overwrite:
            base_seed = int(cache["seed"])
            union_times = np.unique(np.concatenate([cache["times"], times_arr]))
            if union_times.size == cache["times"].size:
                return
            self._build_persistent_cache(
                union_times, n=int(n), seed=base_seed, noisy=bool(noisy)
            )
            return
        if seed is None:
            rng = rng or np.random.default_rng()
            seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._build_persistent_cache(
            times_arr, n=int(n), seed=int(seed), noisy=bool(noisy)
        )

    @staticmethod
    def metric_speed(observed: Array, dt: float) -> Array:
        if observed.ndim != 3:
            raise ValueError("observed must have shape (T, N, D)")
        if dt <= 0:
            raise ValueError("dt must be positive")
        diffs = observed[1:] - observed[:-1]
        step_norms = np.linalg.norm(diffs, axis=-1)
        speeds = step_norms.mean(axis=1) / dt
        return np.concatenate([[0.0], speeds])

    def simulate(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        return_latent: bool = True,
        return_fates: bool = True,
    ) -> dict[str, Array | dict[str, Array]]:
        """Simulate the full trajectory on the default dt grid."""
        if n <= 0:
            raise ValueError("n must be positive")
        rng = rng or np.random.default_rng()
        times = self.observation_times()
        num_times = len(times)

        y_fate = rng.choice([-1, 1], size=n)
        x_fate = rng.choice([-1, 1], size=n)

        latent = rng.normal(scale=self.init_sigma, size=(n, self.latent_dim))
        latent_out = (
            np.empty((num_times, n, self.latent_dim)) if return_latent else None
        )
        observed = np.empty((num_times, n, self.obs_dim))
        speeds = np.empty(num_times, dtype=float)

        if return_latent:
            latent_out[0] = latent
        clean_prev = latent @ self.q_matrix.T
        observed[0] = clean_prev + rng.normal(
            scale=self.obs_noise, size=(n, self.obs_dim)
        )
        speeds[0] = 0.0

        for idx in range(1, num_times):
            t = float(times[idx - 1])
            mu = self.target_mean(t, y_fate, x_fate)
            latent = self._euler_maruyama_step(latent, mu, self.dt, rng)
            if return_latent:
                latent_out[idx] = latent
            clean = latent @ self.q_matrix.T
            speeds[idx] = np.linalg.norm(clean - clean_prev, axis=-1).mean() / self.dt
            clean_prev = clean
            observed[idx] = clean + rng.normal(
                scale=self.obs_noise, size=(n, self.obs_dim)
            )

        result: dict[str, Array | dict[str, Array]] = {
            "times": times,
            "observed": observed,
            "metric_speed": speeds,
        }
        if return_latent and latent_out is not None:
            result["latent"] = latent_out
        if return_fates:
            result["fates"] = {"y_fate": y_fate, "x_fate": x_fate}
        return result

    def sample(
        self, t: float, n: int, rng: Optional[np.random.Generator] = None
    ) -> EmpiricalMeasure:
        if n <= 0:
            raise ValueError("n must be positive")
        if t < self.t_start or t > self.t_end:
            raise ValueError("t must lie within [t_start, t_end]")
        rng = rng or np.random.default_rng()

        y_fate = rng.choice([-1, 1], size=n)
        x_fate = rng.choice([-1, 1], size=n)
        latent = rng.normal(scale=self.init_sigma, size=(n, self.latent_dim))

        current_time = float(self.t_start)
        while current_time + self.dt <= t + 1e-12:
            mu = self.target_mean(current_time, y_fate, x_fate)
            latent = self._euler_maruyama_step(latent, mu, self.dt, rng)
            current_time += self.dt

        remainder = float(t) - current_time
        if remainder > 1e-12:
            mu = self.target_mean(current_time, y_fate, x_fate)
            latent = self._euler_maruyama_step(latent, mu, remainder, rng)

        observed = self.observe(latent, rng, noisy=True)
        return EmpiricalMeasure(support=observed)

    def sample_clean(
        self, t: float, n: int, rng: Optional[np.random.Generator] = None
    ) -> EmpiricalMeasure:
        """Sample noiseless observations (QZ) at time t."""
        if n <= 0:
            raise ValueError("n must be positive")
        if t < self.t_start or t > self.t_end:
            raise ValueError("t must lie within [t_start, t_end]")
        rng = rng or np.random.default_rng()

        y_fate = rng.choice([-1, 1], size=n)
        x_fate = rng.choice([-1, 1], size=n)
        latent = rng.normal(scale=self.init_sigma, size=(n, self.latent_dim))

        current_time = float(self.t_start)
        while current_time + self.dt <= t + 1e-12:
            mu = self.target_mean(current_time, y_fate, x_fate)
            latent = self._euler_maruyama_step(latent, mu, self.dt, rng)
            current_time += self.dt

        remainder = float(t) - current_time
        if remainder > 1e-12:
            mu = self.target_mean(current_time, y_fate, x_fate)
            latent = self._euler_maruyama_step(latent, mu, remainder, rng)

        observed = self.observe(latent, rng, noisy=False)
        return EmpiricalMeasure(support=observed)

    def sample_persistent(
        self,
        t: float,
        n: int,
        rng: Optional[np.random.Generator] = None,
        noisy: bool = True,
    ) -> EmpiricalMeasure:
        """Sample from a persistent trajectory."""
        if n <= 0:
            raise ValueError("n must be positive")
        if t < self.t_start or t > self.t_end:
            raise ValueError("t must lie within [t_start, t_end]")
        key = (int(n), bool(noisy))
        cache = self._persistent_cache.get(key)
        if cache is None:
            self.prepare_persistent_cache(
                np.array([t], dtype=float), n=n, rng=rng, noisy=noisy
            )
            cache = self._persistent_cache[key]
        times = cache["times"]
        idx = np.where(np.isclose(times, float(t), atol=self.time_tolerance, rtol=0.0))[
            0
        ]
        if idx.size == 0:
            union_times = np.unique(np.concatenate([times, np.array([t], dtype=float)]))
            self._build_persistent_cache(
                union_times, n=int(n), seed=int(cache["seed"]), noisy=bool(noisy)
            )
            cache = self._persistent_cache[key]
            idx = np.where(
                np.isclose(cache["times"], float(t), atol=self.time_tolerance, rtol=0.0)
            )[0]
        if idx.size == 0:
            raise ValueError("time t not found in persistent cache")
        return EmpiricalMeasure(support=cache["obs"][int(idx[0])])

    def sample_persistent_clean(
        self,
        t: float,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> EmpiricalMeasure:
        """Persistent snapshots without observation noise."""
        return self.sample_persistent(t, n, rng=rng, noisy=False)

    def velocity_proxy(
        self,
        times: Array,
        n: int,
        rng: Optional[np.random.Generator] = None,
        noisy: bool = False,
    ) -> Array:
        """Compute per-interval speed proxy using persistent particles."""
        if n <= 0:
            raise ValueError("n must be positive")
        rng = rng or np.random.default_rng()
        obs = self._simulate_observations(times, n=n, rng=rng, noisy=noisy)
        diffs = obs[1:] - obs[:-1]
        step_norms = np.linalg.norm(diffs, axis=-1)
        speeds = step_norms.mean(axis=1) / np.diff(np.asarray(times, dtype=float))
        return speeds


@dataclass
class OscillatorySequentialBranching(SequentialBranchingTrajectory):
    """Sequential branching with oscillatory nuisance motion near events."""

    nuisance_freq: float = 30.0
    nuisance_amp: float = 0.5

    event1: tuple[float, float] = (0.30, 0.35)
    event2: tuple[float, float] = (0.70, 0.75)
    kappa: float = 2.0

    def target_mean(self, t: float, y_fate: Array, x_fate: Array) -> Array:
        s1 = smooth_transition(t, self.event1[0], self.event1[1])
        s2 = smooth_transition(t, self.event2[0], self.event2[1])

        mu_y = y_fate * self.branch_scale * s1
        gate = (y_fate == self.split_branch_sign).astype(float)
        mu_x = gate * x_fate * self.branch_scale * s2

        env1 = 20.0 * s1 * (1.0 - s1)
        env2 = 20.0 * s2 * (1.0 - s2)
        wobble = np.sin(self.nuisance_freq * t)

        nuisance_x = self.nuisance_amp * env1 * wobble
        nuisance_y = self.nuisance_amp * env2 * wobble

        return np.stack([mu_x + nuisance_x, mu_y + nuisance_y], axis=-1)


if __name__ == "__main__":
    traj = SequentialBranchingTrajectory()
    sim = traj.simulate(n=512, rng=np.random.default_rng(0))
    speeds = sim["metric_speed"]
    print(
        "speed peaks at indices:",
        int(np.argmax(speeds[: len(speeds) // 2])),
        int(np.argmax(speeds[len(speeds) // 2 :]) + len(speeds) // 2),
    )
