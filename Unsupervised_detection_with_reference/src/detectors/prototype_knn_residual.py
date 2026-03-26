from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class PrototypeKNNStats:
    feature_dim: int
    num_reference_samples: int
    normalize: bool
    lambda_proto: float
    k_neighbors: int


class PrototypeKNNResidualDetector:
    """
    Prototype + kNN residual detector.

    Score:
        score(x) = lambda_proto * proto_score(x)
                 + (1 - lambda_proto) * knn_score(x)

    where:
        proto_score(x) = 1 - cosine(x, prototype)
        knn_score(x)   = mean of k smallest cosine distances to reference set

    Larger score means more anomalous.
    """

    def __init__(
        self,
        lambda_proto: float = 0.5,
        k_neighbors: int = 1,
        normalize: bool = True,
        eps: float = 1e-12,
        use_float64: bool = True,
    ):
        if not (0.0 <= lambda_proto <= 1.0):
            raise ValueError("lambda_proto must be in [0, 1]")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")

        self.lambda_proto = lambda_proto
        self.k_neighbors = k_neighbors
        self.normalize = normalize
        self.eps = eps
        self.use_float64 = use_float64

        self.stats: Optional[PrototypeKNNStats] = None
        self.prototype: Optional[np.ndarray] = None
        self.reference: Optional[np.ndarray] = None

    def fit(self, x_ref: np.ndarray) -> "PrototypeKNNResidualDetector":
        x_ref = self._validate_features(x_ref, name="x_ref")
        dtype = np.float64 if self.use_float64 else np.float32
        x_ref = x_ref.astype(dtype, copy=False)

        if self.normalize:
            x_ref = self._l2_normalize(x_ref, self.eps)

        self.reference = x_ref
        prototype = x_ref.mean(axis=0)
        if self.normalize:
            prototype = prototype / max(np.linalg.norm(prototype), self.eps)
        self.prototype = prototype

        self.stats = PrototypeKNNStats(
            feature_dim=x_ref.shape[1],
            num_reference_samples=x_ref.shape[0],
            normalize=self.normalize,
            lambda_proto=self.lambda_proto,
            k_neighbors=min(self.k_neighbors, x_ref.shape[0]),
        )
        return self

    def score(
        self,
        x: np.ndarray,
        return_components: bool = False,
    ):
        self._check_is_fitted()
        x = self._validate_features(x, name="x")
        dtype = self.reference.dtype
        x = x.astype(dtype, copy=False)

        if x.shape[1] != self.stats.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {x.shape[1]}, expected {self.stats.feature_dim}"
            )

        if self.normalize:
            x = self._l2_normalize(x, self.eps)

        proto_score = self._prototype_score(x)
        knn_score = self._knn_score(x)

        score = self.lambda_proto * proto_score + (1.0 - self.lambda_proto) * knn_score

        if return_components:
            return score, proto_score, knn_score
        return score

    def fit_score(
        self,
        x_ref: np.ndarray,
        x_test: np.ndarray,
        return_components: bool = False,
    ):
        self.fit(x_ref)
        return self.score(x_test, return_components=return_components)

    def summary(self) -> Dict[str, Any]:
        self._check_is_fitted()
        return {
            "feature_dim": self.stats.feature_dim,
            "num_reference_samples": self.stats.num_reference_samples,
            "normalize": self.stats.normalize,
            "lambda_proto": self.stats.lambda_proto,
            "k_neighbors": self.stats.k_neighbors,
        }

    def save(self, path: str) -> None:
        self._check_is_fitted()
        np.savez_compressed(
            path,
            prototype=self.prototype,
            reference=self.reference,
            feature_dim=np.array([self.stats.feature_dim], dtype=np.int64),
            num_reference_samples=np.array([self.stats.num_reference_samples], dtype=np.int64),
            normalize=np.array([self.stats.normalize], dtype=bool),
            lambda_proto=np.array([self.stats.lambda_proto], dtype=np.float64),
            k_neighbors=np.array([self.stats.k_neighbors], dtype=np.int64),
            eps=np.array([self.eps], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str, use_float64: bool = True) -> "PrototypeKNNResidualDetector":
        data = np.load(path, allow_pickle=True)
        detector = cls(
            lambda_proto=float(data["lambda_proto"][0]),
            k_neighbors=int(data["k_neighbors"][0]),
            normalize=bool(data["normalize"][0]),
            eps=float(data["eps"][0]),
            use_float64=use_float64,
        )
        detector.prototype = data["prototype"]
        detector.reference = data["reference"]
        detector.stats = PrototypeKNNStats(
            feature_dim=int(data["feature_dim"][0]),
            num_reference_samples=int(data["num_reference_samples"][0]),
            normalize=bool(data["normalize"][0]),
            lambda_proto=float(data["lambda_proto"][0]),
            k_neighbors=int(data["k_neighbors"][0]),
        )
        return detector

    def _prototype_score(self, x: np.ndarray) -> np.ndarray:
        # cosine distance = 1 - cosine similarity
        sim = x @ self.prototype
        return 1.0 - sim

    def _knn_score(self, x: np.ndarray) -> np.ndarray:
        # cosine distance assuming normalized vectors
        sims = x @ self.reference.T  # [N, R]
        dists = 1.0 - sims

        k = self.stats.k_neighbors
        if k == 1:
            return np.min(dists, axis=1)

        part = np.partition(dists, kth=k - 1, axis=1)[:, :k]
        return np.mean(part, axis=1)

    def _check_is_fitted(self) -> None:
        if self.stats is None or self.prototype is None or self.reference is None:
            raise RuntimeError("PrototypeKNNResidualDetector is not fitted yet.")

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