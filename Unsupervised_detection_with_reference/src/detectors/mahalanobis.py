# src/detectors/mahalanobis.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class MahalanobisStats:
    mean: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    feature_dim: int
    num_reference_samples: int
    regularization: float
    covariance_type: str


class MahalanobisDetector:
    """
    Mahalanobis distance based anomaly detector.

    Typical usage:
        detector = MahalanobisDetector(regularization=1e-4)
        detector.fit(x_ref)
        scores = detector.score(x_test)

    Input:
        x_ref:  [N, D] reference features, usually real/nature train features
        x_test: [M, D] test features

    Output:
        scores: [M], larger means more anomalous
    """

    def __init__(
        self,
        regularization: float = 1e-4,
        covariance_type: str = "full",
        center: bool = True,
        use_float64: bool = True,
    ):
        """
        Args:
            regularization: diagonal regularization added to covariance
            covariance_type: 'full' or 'diag'
            center: whether to subtract mean before distance computation
            use_float64: whether to compute stats in float64 for stability
        """
        if covariance_type not in {"full", "diag"}:
            raise ValueError("covariance_type must be 'full' or 'diag'")

        self.regularization = regularization
        self.covariance_type = covariance_type
        self.center = center
        self.use_float64 = use_float64

        self.stats: Optional[MahalanobisStats] = None

    def fit(self, x_ref: np.ndarray) -> "MahalanobisDetector":
        """
        Fit Gaussian statistics from reference features.

        Args:
            x_ref: [N, D]
        """
        x_ref = self._validate_features(x_ref, name="x_ref")

        dtype = np.float64 if self.use_float64 else np.float32
        x_ref = x_ref.astype(dtype, copy=False)

        n, d = x_ref.shape
        if n < 2:
            raise ValueError(f"x_ref must have at least 2 samples, got {n}")

        mean = x_ref.mean(axis=0)

        if self.center:
            centered = x_ref - mean
        else:
            centered = x_ref

        if self.covariance_type == "full":
            cov = np.cov(centered, rowvar=False, bias=False)
            if cov.ndim == 0:
                cov = np.array([[cov]], dtype=dtype)
            cov = cov + self.regularization * np.eye(d, dtype=dtype)
            inv_cov = np.linalg.pinv(cov)

        else:  # diag
            var = centered.var(axis=0, ddof=1)
            var = var + self.regularization
            cov = np.diag(var)
            inv_cov = np.diag(1.0 / var)

        self.stats = MahalanobisStats(
            mean=mean,
            cov=cov,
            inv_cov=inv_cov,
            feature_dim=d,
            num_reference_samples=n,
            regularization=self.regularization,
            covariance_type=self.covariance_type,
        )
        return self

    def score(self, x: np.ndarray, squared: bool = True) -> np.ndarray:
        """
        Compute Mahalanobis scores.

        Args:
            x: [M, D]
            squared: if True, return squared Mahalanobis distance

        Returns:
            scores: [M], larger means more anomalous
        """
        self._check_is_fitted()
        x = self._validate_features(x, name="x")

        dtype = self.stats.mean.dtype
        x = x.astype(dtype, copy=False)

        if x.shape[1] != self.stats.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {x.shape[1]}, "
                f"expected {self.stats.feature_dim}"
            )

        diff = x - self.stats.mean if self.center else x

        # Efficient vectorized quadratic form:
        # dist^2 = (x-mu)^T Sigma^{-1} (x-mu)
        scores = np.sum((diff @ self.stats.inv_cov) * diff, axis=1)

        # Numerical safety
        scores = np.maximum(scores, 0.0)

        if squared:
            return scores
        return np.sqrt(scores)

    def fit_score(
        self,
        x_ref: np.ndarray,
        x_test: np.ndarray,
        squared: bool = True,
    ) -> np.ndarray:
        """
        Convenience wrapper: fit on x_ref then score x_test.
        """
        self.fit(x_ref)
        return self.score(x_test, squared=squared)

    def get_stats(self) -> MahalanobisStats:
        self._check_is_fitted()
        return self.stats

    def save(self, path: str) -> None:
        """
        Save fitted statistics to .npz
        """
        self._check_is_fitted()

        np.savez_compressed(
            path,
            mean=self.stats.mean,
            cov=self.stats.cov,
            inv_cov=self.stats.inv_cov,
            feature_dim=np.array([self.stats.feature_dim], dtype=np.int64),
            num_reference_samples=np.array([self.stats.num_reference_samples], dtype=np.int64),
            regularization=np.array([self.stats.regularization], dtype=np.float64),
            covariance_type=np.array([self.stats.covariance_type], dtype=object),
        )

    @classmethod
    def load(
        cls,
        path: str,
        center: bool = True,
        use_float64: bool = True,
    ) -> "MahalanobisDetector":
        """
        Load fitted statistics from .npz
        """
        data = np.load(path, allow_pickle=True)

        detector = cls(
            regularization=float(data["regularization"][0]),
            covariance_type=str(data["covariance_type"][0]),
            center=center,
            use_float64=use_float64,
        )

        detector.stats = MahalanobisStats(
            mean=data["mean"],
            cov=data["cov"],
            inv_cov=data["inv_cov"],
            feature_dim=int(data["feature_dim"][0]),
            num_reference_samples=int(data["num_reference_samples"][0]),
            regularization=float(data["regularization"][0]),
            covariance_type=str(data["covariance_type"][0]),
        )
        return detector

    def summary(self) -> Dict[str, Any]:
        self._check_is_fitted()
        return {
            "feature_dim": self.stats.feature_dim,
            "num_reference_samples": self.stats.num_reference_samples,
            "regularization": self.stats.regularization,
            "covariance_type": self.stats.covariance_type,
            "center": self.center,
        }

    def _check_is_fitted(self) -> None:
        if self.stats is None:
            raise RuntimeError("MahalanobisDetector is not fitted yet.")

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