"""Trajectory wrapper for the Schiebinger reprogramming dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from active_wasserstein.measures import EmpiricalMeasure

logger = logging.getLogger(__name__)


@dataclass
class SchiebingerReprogrammingTrajectory:
    """Expose the Schiebinger reprogramming dataset as a sampling trajectory."""

    path: str | None = None
    subset_to_serum: bool = True
    batch_key: str | None = None
    time_key: str | None = None
    train_batch: str | int | None = None
    eval_batch: str | int | None = None
    train_batch_index: int = 0
    eval_batch_index: int = 1
    split_within_batch: bool = False
    eval_time_fraction: float = 0.5
    embedding_key: str = "X_pca"
    n_pcs: int = 20
    pca_random_state: int = 0
    whiten_pca: bool = False
    allow_download: bool = False
    use_cellrank_loader: bool = True
    min_cells_per_time: int = 1
    time_tolerance: float = 1.0e-6

    t_start: float = field(init=False)
    t_end: float = field(init=False)
    candidate_times: np.ndarray = field(init=False, repr=False)
    eval_times: np.ndarray = field(init=False, repr=False)
    _train_data: dict[float, np.ndarray] = field(
        init=False, repr=False, default_factory=dict
    )
    _eval_data: dict[float, np.ndarray] = field(
        init=False, repr=False, default_factory=dict
    )
    _train_times: np.ndarray = field(init=False, repr=False)
    _eval_times: np.ndarray = field(init=False, repr=False)
    _train_counts: dict[float, int] = field(
        init=False, repr=False, default_factory=dict
    )
    _eval_counts: dict[float, int] = field(init=False, repr=False, default_factory=dict)
    _resolved_batch_key: str = field(init=False, repr=False)
    _resolved_time_key: str = field(init=False, repr=False)
    _train_batch_value: object = field(init=False, repr=False)
    _eval_batch_value: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        adata = self._load_adata()
        self._resolved_batch_key = self._resolve_batch_key(adata)
        self._resolved_time_key = self._resolve_time_key(adata)

        time_series = pd.to_numeric(adata.obs[self._resolved_time_key], errors="coerce")
        valid_mask = ~time_series.isna()
        if not bool(valid_mask.all()):
            dropped = int((~valid_mask).sum())
            logger.warning("Dropping %d cells with missing time values", dropped)
            adata = adata[valid_mask].copy()
            time_series = time_series[valid_mask]

        embedding = self._get_embedding(adata)

        batch_series = adata.obs[self._resolved_batch_key]
        train_value, eval_value = self._resolve_batch_values(batch_series)
        same_batch = train_value == eval_value
        if same_batch and not self.split_within_batch:
            raise ValueError(
                "train_batch and eval_batch resolve to the same value; "
                "set split_within_batch=True to split time points within the batch"
            )
        self._train_batch_value = train_value
        self._eval_batch_value = eval_value

        train_mask = batch_series == train_value
        eval_mask = batch_series == eval_value

        train_map, train_times, train_counts = self._build_time_map(
            embedding[train_mask],
            time_series[train_mask].to_numpy(dtype=float),
            min_cells=self.min_cells_per_time,
        )
        eval_map, eval_times, eval_counts = self._build_time_map(
            embedding[eval_mask],
            time_series[eval_mask].to_numpy(dtype=float),
            min_cells=self.min_cells_per_time,
        )

        if same_batch:
            train_map, train_times, train_counts, eval_map, eval_times, eval_counts = (
                self._split_time_maps(train_map, train_times, train_counts)
            )

        if train_times.size == 0:
            raise ValueError("No training time points remain after filtering")
        if eval_times.size == 0:
            raise ValueError("No evaluation time points remain after filtering")

        object.__setattr__(self, "_train_data", train_map)
        object.__setattr__(self, "_eval_data", eval_map)
        object.__setattr__(self, "_train_times", train_times)
        object.__setattr__(self, "_eval_times", eval_times)
        object.__setattr__(self, "_train_counts", train_counts)
        object.__setattr__(self, "_eval_counts", eval_counts)
        object.__setattr__(self, "candidate_times", train_times)
        object.__setattr__(self, "eval_times", eval_times)
        object.__setattr__(self, "t_start", float(train_times[0]))
        object.__setattr__(self, "t_end", float(train_times[-1]))

        if same_batch:
            logger.info(
                "Schiebinger trajectory ready: batch=%s (train_times=%d, eval_times=%d, eval_fraction=%.3f)",
                train_value,
                train_times.size,
                eval_times.size,
                self.eval_time_fraction,
            )
        else:
            logger.info(
                "Schiebinger trajectory ready: train_batch=%s (%d times), eval_batch=%s (%d times)",
                train_value,
                train_times.size,
                eval_value,
                eval_times.size,
            )

    def sample(
        self, t: float, n: int | None, rng: np.random.Generator | None = None
    ) -> EmpiricalMeasure:
        """Sample an empirical measure from the training batch."""
        return self._sample_from_map(self._train_data, self._train_times, t, n, rng)

    def sample_train(
        self,
        t: float,
        n: int | None,
        rng: np.random.Generator | None = None,
    ) -> EmpiricalMeasure:
        """Explicit alias for training-batch sampling."""
        return self.sample(t, n, rng=rng)

    def sample_eval(
        self,
        t: float,
        n: int | None,
        rng: np.random.Generator | None = None,
    ) -> EmpiricalMeasure:
        """Sample an empirical measure from the evaluation batch."""
        return self._sample_from_map(self._eval_data, self._eval_times, t, n, rng)

    def _load_adata(self):
        try:
            import scanpy as sc
        except ImportError as exc:
            raise ImportError(
                "scanpy is required to load the Schiebinger dataset"
            ) from exc
        try:
            import cellrank as cr
        except ImportError as exc:
            raise ImportError(
                "cellrank is required to load the Schiebinger dataset"
            ) from exc

        if self.path is None:
            if not self.allow_download:
                raise ValueError("path must be set when allow_download=False")
            return cr.datasets.reprogramming_schiebinger(
                subset_to_serum=self.subset_to_serum,
            )

        path = Path(self.path)
        if path.exists():
            if self.use_cellrank_loader:
                return cr.datasets.reprogramming_schiebinger(
                    path=str(path),
                    subset_to_serum=self.subset_to_serum,
                )
            return sc.read_h5ad(path)

        if not self.allow_download:
            raise FileNotFoundError(f"Dataset not found at {path}")
        return cr.datasets.reprogramming_schiebinger(
            path=str(path),
            subset_to_serum=self.subset_to_serum,
        )

    def _resolve_batch_key(self, adata) -> str:
        if self.batch_key is not None:
            if self.batch_key not in adata.obs.columns:
                raise ValueError(f"batch_key '{self.batch_key}' not found in adata.obs")
            return self.batch_key
        if "batch" in adata.obs.columns:
            return "batch"
        candidates = []
        for col in adata.obs.columns:
            lowered = col.lower()
            if any(
                key in lowered
                for key in ["rep", "batch", "sample", "donor", "line", "lane"]
            ):
                candidates.append(col)
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            "Unable to infer batch column; set batch_key explicitly (candidates: "
            f"{candidates})"
        )

    def _resolve_time_key(self, adata) -> str:
        if self.time_key is not None:
            if self.time_key not in adata.obs.columns:
                raise ValueError(f"time_key '{self.time_key}' not found in adata.obs")
            return self.time_key
        if "day_numerical" in adata.obs.columns:
            return "day_numerical"
        if "day" in adata.obs.columns:
            return "day"
        if "time" in adata.obs.columns:
            return "time"
        raise ValueError("Unable to infer time column; set time_key explicitly")

    def _resolve_batch_values(self, batch_series: Iterable) -> tuple[object, object]:
        values = pd.unique(pd.Series(batch_series))
        if values.size < 1:
            raise ValueError("No batches found in dataset")
        if values.size < 2 and not self.split_within_batch:
            raise ValueError("Expected at least two batches for train/eval split")

        def match_value(target: str | int | None, index: int) -> object:
            if target is None:
                if index >= values.size:
                    raise ValueError("batch index out of range for available batches")
                return values[index]
            for val in values:
                if val == target:
                    return val
            target_str = str(target)
            for val in values:
                if str(val) == target_str:
                    return val
            raise ValueError(
                f"batch value '{target}' not found among {values.tolist()}"
            )

        train_value = match_value(self.train_batch, self.train_batch_index)
        eval_value = match_value(self.eval_batch, self.eval_batch_index)
        if train_value == eval_value and not self.split_within_batch:
            raise ValueError("train_batch and eval_batch resolve to the same value")
        return train_value, eval_value

    def _get_embedding(self, adata) -> np.ndarray:
        try:
            import scanpy as sc
        except ImportError as exc:
            raise ImportError("scanpy is required for PCA preprocessing") from exc
        try:
            import scipy.sparse as sp
        except ImportError:
            sp = None

        if self.embedding_key == "X_pca":
            rep = adata.obsm.get("X_pca")
            if rep is None or rep.shape[1] < self.n_pcs:
                sc.pp.pca(adata, n_comps=self.n_pcs, random_state=self.pca_random_state)
                rep = adata.obsm["X_pca"]
            if rep.shape[1] > self.n_pcs:
                rep = rep[:, : self.n_pcs]
        elif self.embedding_key == "X":
            rep = adata.X
        else:
            rep = adata.obsm.get(self.embedding_key)
            if rep is None:
                raise ValueError(
                    f"embedding_key '{self.embedding_key}' not found in adata.obsm"
                )

        if sp is not None and sp.issparse(rep):
            rep = rep.toarray()
        rep = np.asarray(rep, dtype=float)
        if self.whiten_pca:
            rep = rep - rep.mean(axis=0, keepdims=True)
            std = rep.std(axis=0, ddof=0, keepdims=True)
            std = np.where(std > 0, std, 1.0)
            rep = rep / std
        return rep

    def _build_time_map(
        self,
        embedding: np.ndarray,
        times: np.ndarray,
        min_cells: int,
    ) -> tuple[dict[float, np.ndarray], np.ndarray, dict[float, int]]:
        time_map: dict[float, np.ndarray] = {}
        counts: dict[float, int] = {}
        unique_times = np.unique(times)
        for t in np.sort(unique_times):
            mask = np.isclose(times, t, atol=self.time_tolerance, rtol=0.0)
            count = int(mask.sum())
            if count < min_cells:
                logger.debug(
                    "Skipping time %.4f with %d cells (min=%d)", t, count, min_cells
                )
                continue
            time_map[float(t)] = embedding[mask]
            counts[float(t)] = count
        ordered_times = np.array(sorted(time_map.keys()), dtype=float)
        return time_map, ordered_times, counts

    def _split_time_maps(
        self,
        full_map: dict[float, np.ndarray],
        full_times: np.ndarray,
        full_counts: dict[float, int],
    ) -> tuple[
        dict[float, np.ndarray],
        np.ndarray,
        dict[float, int],
        dict[float, np.ndarray],
        np.ndarray,
        dict[float, int],
    ]:
        train_times, eval_times = self._split_times(full_times)
        train_map = {t: full_map[t] for t in train_times}
        eval_map = {t: full_map[t] for t in eval_times}
        train_counts = {t: full_counts[t] for t in train_times}
        eval_counts = {t: full_counts[t] for t in eval_times}
        return train_map, train_times, train_counts, eval_map, eval_times, eval_counts

    def _split_times(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=float)
        if times.size < 2:
            raise ValueError(
                "Need at least two time points to split train/eval within a batch"
            )
        fraction = float(self.eval_time_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("eval_time_fraction must be between 0 and 1")
        num_eval = int(np.floor(times.size * fraction))
        num_eval = max(1, num_eval)
        if num_eval >= times.size:
            num_eval = times.size - 1
        eval_indices = np.linspace(0, times.size - 1, num_eval, dtype=int)
        eval_mask = np.zeros(times.size, dtype=bool)
        eval_mask[eval_indices] = True
        eval_times = times[eval_mask]
        train_times = times[~eval_mask]
        return train_times, eval_times

    def _sample_from_map(
        self,
        time_map: dict[float, np.ndarray],
        available_times: np.ndarray,
        t: float,
        n: int | None,
        rng: np.random.Generator | None,
    ) -> EmpiricalMeasure:
        if n is not None and n <= 0:
            raise ValueError("n must be positive")
        rng = rng or np.random.default_rng()
        resolved = self._resolve_time(t, available_times)
        support_pool = time_map[resolved]
        if support_pool.shape[0] == 0:
            raise ValueError("No cells available at the requested time")
        if n is None:
            return EmpiricalMeasure(support=support_pool)
        idx = rng.choice(support_pool.shape[0], size=int(n), replace=True)
        return EmpiricalMeasure(support=support_pool[idx])

    def _resolve_time(self, t: float, available_times: np.ndarray) -> float:
        if available_times.size == 0:
            raise ValueError("No times available for sampling")
        t_val = float(t)
        diffs = np.abs(available_times - t_val)
        idx = int(np.argmin(diffs))
        if diffs[idx] > self.time_tolerance:
            raise ValueError(
                f"Requested time {t_val} not in available times "
                f"(closest={available_times[idx]:.4f})"
            )
        return float(available_times[idx])

    def metadata(self) -> dict:
        """Return dataset metadata for experiment logging."""
        train_counts = {str(k): int(v) for k, v in self._train_counts.items()}
        eval_counts = {str(k): int(v) for k, v in self._eval_counts.items()}
        return {
            "dataset_path": str(self.path) if self.path is not None else None,
            "subset_to_serum": bool(self.subset_to_serum),
            "batch_key": self._resolved_batch_key,
            "time_key": self._resolved_time_key,
            "train_batch": str(self._train_batch_value),
            "eval_batch": str(self._eval_batch_value),
            "split_within_batch": bool(self.split_within_batch),
            "eval_time_fraction": float(self.eval_time_fraction),
            "embedding_key": self.embedding_key,
            "n_pcs": int(self.n_pcs),
            "pca_random_state": int(self.pca_random_state),
            "whiten_pca": bool(self.whiten_pca),
            "min_cells_per_time": int(self.min_cells_per_time),
            "time_tolerance": float(self.time_tolerance),
            "train_times": self._train_times.tolist(),
            "eval_times": self._eval_times.tolist(),
            "train_counts": train_counts,
            "eval_counts": eval_counts,
        }
