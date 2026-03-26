#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detectors.mahalanobis import MahalanobisDetector
from src.evaluation.metrics import compute_metrics, format_metrics
from src.evaluation.io import save_scores_npz, save_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mahalanobis detector on extracted train/val features."
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Path to train features .npz",
    )
    parser.add_argument(
        "--val-features",
        type=str,
        required=True,
        help="Path to val features .npz",
    )
    parser.add_argument(
        "--feature-key",
        type=str,
        default="cls_features",
        choices=["cls_features", "mean_features"],
        help="Feature key to use from the NPZ files",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="labels",
        help="Label key in NPZ files",
    )
    parser.add_argument(
        "--reference-label",
        type=int,
        default=0,
        help="Reference label in train set, usually 0 for nature",
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="full",
        choices=["full", "diag"],
        help="Covariance type used in Mahalanobis detector",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-4,
        help="Diagonal regularization added to covariance",
    )
    parser.add_argument(
        "--squared",
        action="store_true",
        help="Use squared Mahalanobis distance as score",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="artifacts/mahalanobis",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--save-reference-stats",
        action="store_true",
        help="Whether to save fitted Mahalanobis reference stats",
    )
    return parser.parse_args()


def load_npz_dict(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def main() -> None:
    args = parse_args()

    train_data = load_npz_dict(args.train_features)
    val_data = load_npz_dict(args.val_features)

    if args.feature_key not in train_data:
        raise KeyError(f"{args.feature_key} not found in train features")
    if args.feature_key not in val_data:
        raise KeyError(f"{args.feature_key} not found in val features")
    if args.label_key not in train_data:
        raise KeyError(f"{args.label_key} not found in train features")
    if args.label_key not in val_data:
        raise KeyError(f"{args.label_key} not found in val features")

    x_train = train_data[args.feature_key]
    y_train = train_data[args.label_key].astype(np.int64)

    x_val = val_data[args.feature_key]
    y_val = val_data[args.label_key].astype(np.int64)

    ref_mask = y_train == args.reference_label
    if ref_mask.sum() == 0:
        raise RuntimeError(
            f"No reference samples found in train set for label={args.reference_label}"
        )

    x_ref = x_train[ref_mask]

    print(f"[Info] Train feature shape: {x_train.shape}")
    print(f"[Info] Val feature shape:   {x_val.shape}")
    print(f"[Info] Reference shape:     {x_ref.shape}")
    print(f"[Info] Feature key:         {args.feature_key}")
    print(f"[Info] Covariance type:     {args.covariance_type}")
    print(f"[Info] Regularization:      {args.regularization}")

    x_ref = l2_normalize(x_ref)
    x_val = l2_normalize(x_val)

    detector = MahalanobisDetector(
        regularization=args.regularization,
        covariance_type=args.covariance_type,
        center=True,
        use_float64=True,
    )
    detector.fit(x_ref)

    scores = detector.score(x_val, squared=args.squared)
    metrics = compute_metrics(y_val, scores)

    print("[Results]")
    print(format_metrics(metrics))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    score_path = save_dir / f"scores_{args.feature_key}_{args.covariance_type}.npz"
    metrics_path = save_dir / f"metrics_{args.feature_key}_{args.covariance_type}.json"

    save_scores_npz(
        str(score_path),
        scores=scores,
        labels=y_val,
        paths=val_data["paths"] if "paths" in val_data else np.array([], dtype=object),
        label_names=val_data["label_names"] if "label_names" in val_data else np.array([], dtype=object),
        feature_key=np.array([args.feature_key], dtype=object),
        covariance_type=np.array([args.covariance_type], dtype=object),
        regularization=np.array([args.regularization], dtype=np.float64),
        reference_label=np.array([args.reference_label], dtype=np.int64),
    )

    payload = {
        "feature_key": args.feature_key,
        "covariance_type": args.covariance_type,
        "regularization": args.regularization,
        "reference_label": args.reference_label,
        "num_reference_samples": int(x_ref.shape[0]),
        "train_feature_shape": list(x_train.shape),
        "val_feature_shape": list(x_val.shape),
        "metrics": metrics,
        "detector_summary": detector.summary(),
    }
    save_metrics_json(str(metrics_path), payload)

    print(f"[Info] Saved scores to: {score_path}")
    print(f"[Info] Saved metrics to: {metrics_path}")

    if args.save_reference_stats:
        ref_path = save_dir / f"reference_{args.feature_key}_{args.covariance_type}.npz"
        detector.save(str(ref_path))
        print(f"[Info] Saved reference stats to: {ref_path}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/run_mahalanobis.py \
#   --train-features artifacts/features/dinov2_train.npz \
#   --val-features artifacts/features/dinov2_val.npz \
#   --feature-key cls_features \
#   --covariance-type full \
#   --regularization 1e-5 \
#   --save-dir artifacts/mahalanobis \
#   --save-reference-stats