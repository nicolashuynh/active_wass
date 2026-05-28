"""Trajectory wrapper for preprocessed CPS monthly snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from active_wasserstein.measures import EmpiricalMeasure, WeightedEmpiricalMeasure


@dataclass
class CpsMonthlyTrajectory:
    """Expose pre-resampled CPS monthly snapshots as a trajectory.

    Expected NPZ fields:
    - train_times: (T,) float
    - eval_times: (T,) float
    - train_arrays: object array length T, each (N_train, d)
    - eval_arrays: object array length T, each (N_eval, d)
    Optional:
    - month_labels: object array length T
    """

    path: str
    time_tolerance: float = 1.0e-10

    t_start: float = field(init=False)
    t_end: float = field(init=False)
    candidate_times: np.ndarray = field(init=False, repr=False)
    eval_times: np.ndarray = field(init=False, repr=False)
    _train_data: dict[float, np.ndarray] = field(init=False, repr=False, default_factory=dict)
    _eval_data: dict[float, np.ndarray] = field(init=False, repr=False, default_factory=dict)
    _train_weights: dict[float, np.ndarray] | None = field(
        init=False, repr=False, default=None
    )
    _eval_weights: dict[float, np.ndarray] | None = field(
        init=False, repr=False, default=None
    )
    _train_times: np.ndarray = field(init=False, repr=False)
    _eval_times: np.ndarray = field(init=False, repr=False)
    _month_labels: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        npz_path = Path(self.path).expanduser().resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"CPS trajectory NPZ not found at {npz_path}")

        with np.load(npz_path, allow_pickle=True) as payload:
            train_times = np.asarray(payload["train_times"], dtype=float).reshape(-1)
            eval_times = np.asarray(payload["eval_times"], dtype=float).reshape(-1)
            train_arrays = list(np.asarray(payload["train_arrays"], dtype=object))
            eval_arrays = list(np.asarray(payload["eval_arrays"], dtype=object))
            train_weights_raw = (
                list(np.asarray(payload["train_weights"], dtype=object))
                if "train_weights" in payload
                else None
            )
            eval_weights_raw = (
                list(np.asarray(payload["eval_weights"], dtype=object))
                if "eval_weights" in payload
                else None
            )
            month_labels = (
                [str(v) for v in np.asarray(payload["month_labels"], dtype=object).reshape(-1).tolist()]
                if "month_labels" in payload
                else []
            )

        if train_times.size == 0 or eval_times.size == 0:
            raise ValueError("train_times/eval_times must be non-empty")
        if len(train_arrays) != train_times.size:
            raise ValueError("train_arrays length must match train_times")
        if len(eval_arrays) != eval_times.size:
            raise ValueError("eval_arrays length must match eval_times")
        if (train_weights_raw is None) != (eval_weights_raw is None):
            raise ValueError(
                "train_weights and eval_weights must either both be present or both be absent"
            )
        if train_weights_raw is not None and len(train_weights_raw) != train_times.size:
            raise ValueError("train_weights length must match train_times")
        if eval_weights_raw is not None and len(eval_weights_raw) != eval_times.size:
            raise ValueError("eval_weights length must match eval_times")

        train_data = {
            float(t): np.asarray(arr, dtype=float)
            for t, arr in zip(train_times.tolist(), train_arrays, strict=True)
        }
        eval_data = {
            float(t): np.asarray(arr, dtype=float)
            for t, arr in zip(eval_times.tolist(), eval_arrays, strict=True)
        }
        for t, arr in train_data.items():
            if arr.ndim != 2 or arr.shape[0] == 0:
                raise ValueError(f"train snapshot at t={t} must be 2D and non-empty")
        for t, arr in eval_data.items():
            if arr.ndim != 2 or arr.shape[0] == 0:
                raise ValueError(f"eval snapshot at t={t} must be 2D and non-empty")

        train_weights = None
        eval_weights = None
        if train_weights_raw is not None and eval_weights_raw is not None:
            train_weights = {}
            for t, support, w in zip(
                train_times.tolist(), train_arrays, train_weights_raw, strict=True
            ):
                support_arr = np.asarray(support, dtype=float)
                w_arr = np.asarray(w, dtype=float).reshape(-1)
                if w_arr.shape[0] != support_arr.shape[0]:
                    raise ValueError(
                        f"train_weights at t={float(t)} must match support length"
                    )
                if np.any(w_arr < 0):
                    raise ValueError("train_weights must be nonnegative")
                total = float(np.sum(w_arr))
                if total <= 0.0:
                    raise ValueError("train_weights must sum to a positive value")
                train_weights[float(t)] = w_arr / total
            eval_weights = {}
            for t, support, w in zip(
                eval_times.tolist(), eval_arrays, eval_weights_raw, strict=True
            ):
                support_arr = np.asarray(support, dtype=float)
                w_arr = np.asarray(w, dtype=float).reshape(-1)
                if w_arr.shape[0] != support_arr.shape[0]:
                    raise ValueError(
                        f"eval_weights at t={float(t)} must match support length"
                    )
                if np.any(w_arr < 0):
                    raise ValueError("eval_weights must be nonnegative")
                total = float(np.sum(w_arr))
                if total <= 0.0:
                    raise ValueError("eval_weights must sum to a positive value")
                eval_weights[float(t)] = w_arr / total

        self._train_data = train_data
        self._eval_data = eval_data
        self._train_weights = train_weights
        self._eval_weights = eval_weights
        self._train_times = np.sort(train_times)
        self._eval_times = np.sort(eval_times)
        self.candidate_times = self._train_times
        self.eval_times = self._eval_times
        self.t_start = float(self._train_times[0])
        self.t_end = float(self._train_times[-1])
        self._month_labels = month_labels

    def _resolve_time(self, t: float, available_times: np.ndarray) -> float:
        diffs = np.abs(available_times - float(t))
        idx = int(np.argmin(diffs))
        if float(diffs[idx]) > float(self.time_tolerance):
            raise ValueError(
                f"Requested time {float(t):.10f} not found "
                f"(closest={float(available_times[idx]):.10f})"
            )
        return float(available_times[idx])

    def _sample_from_map(
        self,
        time_map: dict[float, np.ndarray],
        weight_map: dict[float, np.ndarray] | None,
        available_times: np.ndarray,
        t: float,
        n: int | None,
        rng: np.random.Generator | None,
    ) -> EmpiricalMeasure | WeightedEmpiricalMeasure:
        if n is not None and int(n) <= 0:
            raise ValueError("n must be positive when provided")
        resolved_t = self._resolve_time(t, available_times)
        support = time_map[resolved_t]
        weights = None if weight_map is None else weight_map[resolved_t]
        if support.shape[0] == 0:
            raise ValueError("Empty snapshot support")
        if n is None:
            if weights is not None:
                return WeightedEmpiricalMeasure(support=support, weights=weights)
            return EmpiricalMeasure(support=support)
        rng = rng or np.random.default_rng()
        if weights is None:
            idx = rng.choice(support.shape[0], size=int(n), replace=True)
        else:
            idx = rng.choice(support.shape[0], size=int(n), replace=True, p=weights)
        return EmpiricalMeasure(support=support[idx])

    def sample(
        self, t: float, n: int | None, rng: np.random.Generator | None = None
    ) -> EmpiricalMeasure | WeightedEmpiricalMeasure:
        return self.sample_train(t=t, n=n, rng=rng)

    def sample_train(
        self,
        t: float,
        n: int | None,
        rng: np.random.Generator | None = None,
    ) -> EmpiricalMeasure | WeightedEmpiricalMeasure:
        return self._sample_from_map(
            self._train_data,
            self._train_weights,
            self._train_times,
            t,
            n,
            rng,
        )

    def sample_eval(
        self,
        t: float,
        n: int | None,
        rng: np.random.Generator | None = None,
    ) -> EmpiricalMeasure | WeightedEmpiricalMeasure:
        return self._sample_from_map(
            self._eval_data,
            self._eval_weights,
            self._eval_times,
            t,
            n,
            rng,
        )

    def metadata(self) -> dict:
        return {
            "path": str(self.path),
            "time_tolerance": float(self.time_tolerance),
            "candidate_times": self._train_times.tolist(),
            "eval_times": self._eval_times.tolist(),
            "month_labels": list(self._month_labels),
            "weighted_snapshots": bool(self._train_weights is not None),
            "train_counts": {
                str(k): int(v.shape[0]) for k, v in self._train_data.items()
            },
            "eval_counts": {
                str(k): int(v.shape[0]) for k, v in self._eval_data.items()
            },
        }
