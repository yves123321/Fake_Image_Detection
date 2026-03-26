from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RPOStats:
    feature_dim: int
    n_projections: int
    projector_type: str
    normalize: bool
    eps: float
    sparsity: float
    random_state: int


class RPODetector:
    """
    Random Projection Outlyingness (RPO) detector.

    Core idea:
        1. Sample many random unit projection directions.
        2. Project reference features onto each direction.
        3. For each direction, compute robust center/spread:
              median and MAD
        4. For a test sample, score = max over directions of
              |proj - median| / MAD

    Larger score means more anomalous.

    Typical usage:
        detector = RPODetector(
            n_projections=512,
            projector_type="sparse",
            normalize=True,
            sparsity=0.1,
            random_state=42,
        )
        detector.fit(x_ref)
        scores = detector.score(x_test)
    """

    def __init__(
        self,
        n_projections: int = 512,
        projector_type: str = "sparse",
        normalize: bool = True,
        sparsity: float = 0.1,
        eps: float = 1e-6,
        random_state: int = 42,
        aggregation: str = "max",
        use_float64: bool = True,
    ):
        if n_projections < 1:
            raise ValueError("n_projections must be >= 1")
        if projector_type not in {"gaussian", "sparse"}:
            raise ValueError("projector_type must be 'gaussian' or 'sparse'")
        if aggregation not in {"max", "mean"}:
            raise ValueError("aggregation must be 'max' or 'mean'")
        if not (0.0 < sparsity <= 1.0):
            raise ValueError("sparsity must be in (0, 1]")

        self.n_projections = n_projections
        self.projector_type = projector_type
        self.normalize = normalize
        self.sparsity = sparsity
        self.eps = eps
        self.random_state = random_state
        self.aggregation = aggregation
        self.use_float64 = use_float64

        self.stats: Optional[RPOStats] = None
        self.projector: Optional[np.ndarray] = None
        self.medians: Optional[np.ndarray] = None
        self.mads: Optional[np.ndarray] = None

    def fit(self, x_ref: np.ndarray) -> "RPODetector":
        x_ref = self._validate_features(x_ref, name="x_ref")
        dtype = np.float64 if self.use_float64 else np.float32
        x_ref = x_ref.astype(dtype, copy=False)

        if self.normalize:
            x_ref = self._l2_normalize(x_ref)

        n, d = x_ref.shape
        self.projector = self._build_projector(d, dtype=dtype)

        z_ref = x_ref @ self.projector.T  # [N_ref, P]

        self.medians = np.median(z_ref, axis=0)
        self.mads = np.median(np.abs(z_ref - self.medians[None, :]), axis=0)
        self.mads = np.maximum(self.mads, self.eps)

        self.stats = RPOStats(
            feature_dim=d,
            n_projections=self.n_projections,
            projector_type=self.projector_type,
            normalize=self.normalize,
            eps=self.eps,
            sparsity=self.sparsity,
            random_state=self.random_state,
        )
        return self

    def score(
        self,
        x: np.ndarray,
        return_all_deviations: bool = False,
    ):
        self._check_is_fitted()
        x = self._validate_features(x, name="x")
        dtype = np.float64 if self.use_float64 else np.float32
        x = x.astype(dtype, copy=False)

        if x.shape[1] != self.stats.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {x.shape[1]}, expected {self.stats.feature_dim}"
            )

        if self.normalize:
            x = self._l2_normalize(x)

        z = x @ self.projector.T  # [N, P]
        deviations = np.abs(z - self.medians[None, :]) / self.mads[None, :]

        if self.aggregation == "max":
            scores = np.max(deviations, axis=1)
        else:
            scores = np.mean(deviations, axis=1)

        if return_all_deviations:
            return scores, deviations
        return scores

    def fit_score(
        self,
        x_ref: np.ndarray,
        x_test: np.ndarray,
        return_all_deviations: bool = False,
    ):
        self.fit(x_ref)
        return self.score(x_test, return_all_deviations=return_all_deviations)

    def summary(self) -> Dict[str, Any]:
        self._check_is_fitted()
        return {
            "feature_dim": self.stats.feature_dim,
            "n_projections": self.stats.n_projections,
            "projector_type": self.stats.projector_type,
            "normalize": self.stats.normalize,
            "aggregation": self.aggregation,
            "eps": self.stats.eps,
            "sparsity": self.stats.sparsity,
            "random_state": self.stats.random_state,
        }

    def save(self, path: str) -> None:
        self._check_is_fitted()
        np.savez_compressed(
            path,
            projector=self.projector,
            medians=self.medians,
            mads=self.mads,
            feature_dim=np.array([self.stats.feature_dim], dtype=np.int64),
            n_projections=np.array([self.stats.n_projections], dtype=np.int64),
            projector_type=np.array([self.stats.projector_type], dtype=object),
            normalize=np.array([self.stats.normalize], dtype=bool),
            aggregation=np.array([self.aggregation], dtype=object),
            eps=np.array([self.stats.eps], dtype=np.float64),
            sparsity=np.array([self.stats.sparsity], dtype=np.float64),
            random_state=np.array([self.stats.random_state], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str, use_float64: bool = True) -> "RPODetector":
        data = np.load(path, allow_pickle=True)

        detector = cls(
            n_projections=int(data["n_projections"][0]),
            projector_type=str(data["projector_type"][0]),
            normalize=bool(data["normalize"][0]),
            sparsity=float(data["sparsity"][0]),
            eps=float(data["eps"][0]),
            random_state=int(data["random_state"][0]),
            aggregation=str(data["aggregation"][0]),
            use_float64=use_float64,
        )

        detector.projector = data["projector"]
        detector.medians = data["medians"]
        detector.mads = data["mads"]
        detector.stats = RPOStats(
            feature_dim=int(data["feature_dim"][0]),
            n_projections=int(data["n_projections"][0]),
            projector_type=str(data["projector_type"][0]),
            normalize=bool(data["normalize"][0]),
            eps=float(data["eps"][0]),
            sparsity=float(data["sparsity"][0]),
            random_state=int(data["random_state"][0]),
        )
        return detector

    def _build_projector(self, d: int, dtype=np.float64) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)

        if self.projector_type == "gaussian":
            proj = rng.standard_normal((self.n_projections, d)).astype(dtype)

        elif self.projector_type == "sparse":
            proj = np.zeros((self.n_projections, d), dtype=dtype)
            mask = rng.random((self.n_projections, d)) < self.sparsity
            signs = rng.choice(np.array([-1.0, 1.0], dtype=dtype), size=(self.n_projections, d))
            proj[mask] = signs[mask]

            # make sure no all-zero projection row survives
            zero_rows = np.where(np.linalg.norm(proj, axis=1) < self.eps)[0]
            for r in zero_rows:
                idx = rng.integers(low=0, high=d)
                proj[r, idx] = 1.0

        else:
            raise RuntimeError(f"Unsupported projector_type: {self.projector_type}")

        row_norm = np.linalg.norm(proj, axis=1, keepdims=True)
        row_norm = np.maximum(row_norm, self.eps)
        proj = proj / row_norm
        return proj

    def _check_is_fitted(self) -> None:
        if self.stats is None or self.projector is None or self.medians is None or self.mads is None:
            raise RuntimeError("RPODetector is not fitted yet.")

    @staticmethod
    def _validate_features(x: np.ndarray, name: str = "x") -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError(f"{name} must be a 2D array of shape [N, D], got shape {x.shape}")

        if x.shape[0] == 0:
            raise ValueError(f"{name} is empty")

        if not np.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf values")

        return x

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, eps)
        return x / norm