#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics, format_metrics
from src.evaluation.io import save_scores_npz, save_metrics_json


# ======================
# Utils
# ======================

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def cosine_score(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    x: [N, D] (assumed normalized)
    mu: [D] (assumed normalized)
    return: anomaly score (larger = more abnormal)
    """
    return 1.0 - (x @ mu)


def load_npz_dict(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def build_dataframe(npz_dict: Dict[str, Any], feature_key: str, label_key: str, class_key: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "idx": np.arange(len(npz_dict[label_key])),
        "label": npz_dict[label_key].astype(np.int64),
        "class_key": npz_dict[class_key].astype(str),
    })

    if "paths" in npz_dict:
        df["path"] = npz_dict["paths"].astype(str)
    else:
        df["path"] = ""

    return df


def safe_macro_mean(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


# ======================
# Main
# ======================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class cosine distance baseline")

    parser.add_argument("--train-features", type=str, required=True)
    parser.add_argument("--val-features", type=str, required=True)

    parser.add_argument("--feature-key", type=str, default="cls_features")
    parser.add_argument("--label-key", type=str, default="labels")
    parser.add_argument("--class-key", type=str, default="class_keys")

    parser.add_argument("--reference-label", type=int, default=0)
    parser.add_argument("--fake-label", type=int, default=1)

    parser.add_argument("--num-ref", type=int, default=16)
    parser.add_argument("--num-real-test", type=int, default=4)
    parser.add_argument("--num-fake-test", type=int, default=4)

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--save-dir", type=str, default="artifacts/per_class_cosine")

    return parser.parse_args()


def main():
    args = parse_args()

    rng = np.random.default_rng(args.random_state)

    train_data = load_npz_dict(args.train_features)
    val_data = load_npz_dict(args.val_features)

    x_train = train_data[args.feature_key]
    y_train = train_data[args.label_key].astype(np.int64)

    x_val = val_data[args.feature_key]
    y_val = val_data[args.label_key].astype(np.int64)

    if args.normalize:
        x_train = l2_normalize(x_train)
        x_val = l2_normalize(x_val)

    train_df = build_dataframe(train_data, args.feature_key, args.label_key, args.class_key)
    val_df = build_dataframe(val_data, args.feature_key, args.label_key, args.class_key)

    train_real_df = train_df[train_df["label"] == args.reference_label]
    val_fake_df = val_df[val_df["label"] == args.fake_label]

    candidate_classes = sorted(
        set(train_real_df["class_key"]) & set(val_fake_df["class_key"])
    )

    print(f"[Info] candidate classes: {len(candidate_classes)}")

    per_class_rows = []
    all_scores = []
    all_labels = []

    for class_key in candidate_classes:

        train_real_c = train_real_df[train_real_df["class_key"] == class_key]
        val_fake_c = val_fake_df[val_fake_df["class_key"] == class_key]

        if len(train_real_c) < args.num_ref + args.num_real_test:
            continue
        if len(val_fake_c) < args.num_fake_test:
            continue

        real_idx = rng.permutation(train_real_c["idx"].to_numpy())
        fake_idx = rng.permutation(val_fake_c["idx"].to_numpy())

        ref_idx = real_idx[:args.num_ref]
        real_test_idx = real_idx[args.num_ref: args.num_ref + args.num_real_test]
        fake_test_idx = fake_idx[:args.num_fake_test]

        x_ref = x_train[ref_idx]
        x_real_test = x_train[real_test_idx]
        x_fake_test = x_val[fake_test_idx]

        # ===== cosine core =====
        mu = x_ref.mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)

        x_test = np.concatenate([x_real_test, x_fake_test], axis=0)
        x_test = l2_normalize(x_test)

        scores = cosine_score(x_test, mu)

        y_test = np.concatenate([
            np.zeros(len(x_real_test)),
            np.ones(len(x_fake_test))
        ])

        metrics = compute_metrics(y_test, scores)

        per_class_rows.append({
            "class_key": class_key,
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "fpr95": metrics["fpr95"],
        })

        all_scores.append(scores)
        all_labels.append(y_test)

    df = pd.DataFrame(per_class_rows)

    macro = {
        "auroc": safe_macro_mean(df["auroc"].tolist()),
        "auprc": safe_macro_mean(df["auprc"].tolist()),
        "fpr95": safe_macro_mean(df["fpr95"].tolist()),
    }

    print("[Macro Results]")
    print(format_metrics(macro))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_dir / "per_class_cosine.csv", index=False)

    save_metrics_json(
        str(save_dir / "macro_cosine.json"),
        macro
    )


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/run_per_class_cosine.py \
#   --train-features artifacts/features/dinov2_train_with_classkey.npz \
#   --val-features artifacts/features/dinov2_val_with_classkey.npz \
#   --feature-key cls_features \
#   --class-key class_keys \
#   --num-ref 16 \
#   --num-real-test 4 \
#   --num-fake-test 4 \
#   --normalize