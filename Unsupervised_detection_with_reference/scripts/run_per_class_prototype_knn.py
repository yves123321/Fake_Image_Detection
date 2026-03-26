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

from src.detectors.prototype_knn_residual import PrototypeKNNResidualDetector
from src.evaluation.metrics import compute_metrics, format_metrics
from src.evaluation.io import save_scores_npz, save_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-class Prototype + kNN residual detection."
    )
    parser.add_argument("--train-features", type=str, required=True, help="Path to train features .npz")
    parser.add_argument("--val-features", type=str, required=True, help="Path to val features .npz")
    parser.add_argument(
        "--feature-key",
        type=str,
        default="cls_features",
        #choices=["cls_features", "mean_features"],
        help="Feature key in NPZ",
    )
    parser.add_argument("--label-key", type=str, default="labels", help="Label key in NPZ")
    parser.add_argument("--class-key", type=str, default="class_keys", help="Class key field in NPZ")
    parser.add_argument("--reference-label", type=int, default=0, help="Real/nature label, usually 0")
    parser.add_argument("--fake-label", type=int, default=1, help="Fake/ai label, usually 1")

    parser.add_argument("--num-ref", type=int, default=16, help="Reference real samples per class")
    parser.add_argument("--num-real-test", type=int, default=4, help="Held-out real test samples per class")
    parser.add_argument("--num-fake-test", type=int, default=4, help="Fake test samples per class from val/ai")
    parser.add_argument(
        "--min-ref-samples",
        type=int,
        default=None,
        help="Minimum real samples required per class in train. Default: num_ref + num_real_test",
    )

    parser.add_argument(
        "--lambda-proto",
        type=float,
        default=0.5,
        help="Weight for prototype score in final score",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=1,
        help="Number of nearest reference neighbors for local score",
    )
    parser.add_argument("--normalize", action="store_true", help="Apply L2 normalization inside detector")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for per-class split")
    parser.add_argument("--save-dir", type=str, default="artifacts/per_class_prototype_knn", help="Directory to save outputs")
    parser.add_argument("--save-reference-stats", action="store_true", help="Whether to save one example fitted detector")
    return parser.parse_args()


def load_npz_dict(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def build_dataframe(npz_dict: Dict[str, Any], feature_key: str, label_key: str, class_key: str) -> pd.DataFrame:
    required = [feature_key, label_key, class_key]
    for k in required:
        if k not in npz_dict:
            raise KeyError(f"{k} not found in NPZ")

    n = len(npz_dict[label_key])
    df = pd.DataFrame({
        "idx": np.arange(n),
        "label": npz_dict[label_key].astype(np.int64),
        "class_key": npz_dict[class_key].astype(str),
    })

    if "paths" in npz_dict:
        df["path"] = npz_dict["paths"].astype(str)
    else:
        df["path"] = ""

    if "label_names" in npz_dict:
        df["label_name"] = npz_dict["label_names"].astype(str)
    else:
        df["label_name"] = ""

    return df


def safe_macro_mean(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def main() -> None:
    args = parse_args()

    min_ref_samples = args.min_ref_samples
    if min_ref_samples is None:
        min_ref_samples = args.num_ref + args.num_real_test

    rng = np.random.default_rng(args.random_state)

    train_data = load_npz_dict(args.train_features)
    val_data = load_npz_dict(args.val_features)

    x_train = train_data[args.feature_key]
    y_train = train_data[args.label_key].astype(np.int64)

    x_val = val_data[args.feature_key]
    y_val = val_data[args.label_key].astype(np.int64)

    train_df = build_dataframe(train_data, args.feature_key, args.label_key, args.class_key)
    val_df = build_dataframe(val_data, args.feature_key, args.label_key, args.class_key)

    train_real_df = train_df[train_df["label"] == args.reference-label].copy() if False else train_df[train_df["label"] == args.reference_label].copy()
    val_fake_df = val_df[val_df["label"] == args.fake_label].copy()

    print(f"[Info] Train feature shape: {x_train.shape}")
    print(f"[Info] Val feature shape:   {x_val.shape}")
    print(f"[Info] Train real samples:  {len(train_real_df)}")
    print(f"[Info] Val fake samples:    {len(val_fake_df)}")
    print(f"[Info] Feature key:         {args.feature_key}")
    print(f"[Info] Class key:           {args.class_key}")
    print(f"[Info] num_ref:             {args.num_ref}")
    print(f"[Info] num_real_test:       {args.num_real_test}")
    print(f"[Info] num_fake_test:       {args.num_fake_test}")
    print(f"[Info] lambda_proto:        {args.lambda_proto}")
    print(f"[Info] k_neighbors:         {args.k_neighbors}")
    print(f"[Info] normalize:           {args.normalize}")

    candidate_classes = sorted(
        set(train_real_df["class_key"].unique()) & set(val_fake_df["class_key"].unique())
    )

    print(f"[Info] Candidate classes in intersection: {len(candidate_classes)}")

    per_class_rows: List[Dict[str, Any]] = []
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_proto_scores: List[np.ndarray] = []
    all_knn_scores: List[np.ndarray] = []
    all_class_keys: List[np.ndarray] = []
    all_paths: List[np.ndarray] = []

    used_classes = 0
    skipped_classes = 0
    saved_one_reference = False

    for class_key in candidate_classes:
        train_real_c = train_real_df[train_real_df["class_key"] == class_key]
        val_fake_c = val_fake_df[val_fake_df["class_key"] == class_key]

        n_real_train = len(train_real_c)
        n_fake_val = len(val_fake_c)

        if n_real_train < min_ref_samples:
            skipped_classes += 1
            continue

        if n_fake_val < args.num_fake_test:
            skipped_classes += 1
            continue

        real_indices_all = train_real_c["idx"].to_numpy()
        fake_indices_all = val_fake_c["idx"].to_numpy()

        real_indices_all = rng.permutation(real_indices_all)
        fake_indices_all = rng.permutation(fake_indices_all)

        ref_indices = real_indices_all[:args.num_ref]
        real_test_indices = real_indices_all[args.num_ref: args.num_ref + args.num_real_test]
        fake_test_indices = fake_indices_all[:args.num_fake_test]

        if len(ref_indices) < args.num_ref or len(real_test_indices) < args.num_real_test:
            skipped_classes += 1
            continue

        x_ref = x_train[ref_indices]
        x_real_test = x_train[real_test_indices]
        x_fake_test = x_val[fake_test_indices]

        y_real_test = np.zeros(len(x_real_test), dtype=np.int64)
        y_fake_test = np.ones(len(x_fake_test), dtype=np.int64)

        x_test = np.concatenate([x_real_test, x_fake_test], axis=0)
        y_test = np.concatenate([y_real_test, y_fake_test], axis=0)

        path_real_test = train_df.iloc[real_test_indices]["path"].to_numpy(dtype=object)
        path_fake_test = val_df.iloc[fake_test_indices]["path"].to_numpy(dtype=object)
        path_test = np.concatenate([path_real_test, path_fake_test], axis=0)

        detector = PrototypeKNNResidualDetector(
            lambda_proto=args.lambda_proto,
            k_neighbors=args.k_neighbors,
            normalize=args.normalize,
            use_float64=True,
        )
        detector.fit(x_ref)
        scores, proto_scores, knn_scores = detector.score(x_test, return_components=True)

        metrics = compute_metrics(y_test, scores)

        per_class_rows.append(
            {
                "class_key": class_key,
                "n_ref": int(len(ref_indices)),
                "n_real_test": int(len(real_test_indices)),
                "n_fake_test": int(len(fake_test_indices)),
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "fpr95": metrics["fpr95"],
            }
        )

        all_scores.append(scores)
        all_labels.append(y_test)
        all_proto_scores.append(proto_scores)
        all_knn_scores.append(knn_scores)
        all_class_keys.append(np.array([class_key] * len(y_test), dtype=object))
        all_paths.append(path_test)

        used_classes += 1

        if args.save_reference_stats and not saved_one_reference:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            ref_example_path = save_dir / (
                f"reference_example_{args.feature_key}_lp{args.lambda_proto}_k{args.k_neighbors}.npz"
            )
            detector.save(str(ref_example_path))
            saved_one_reference = True

    if used_classes == 0:
        raise RuntimeError(
            "No valid classes found for per-class Prototype+kNN evaluation. "
            "Check class_key mapping and sample counts."
        )

    per_class_df = pd.DataFrame(per_class_rows).sort_values("class_key").reset_index(drop=True)

    macro_metrics = {
        "auroc": safe_macro_mean(per_class_df["auroc"].tolist()),
        "auprc": safe_macro_mean(per_class_df["auprc"].tolist()),
        "fpr95": safe_macro_mean(per_class_df["fpr95"].tolist()),
    }

    print("[Macro Results]")
    print(format_metrics(macro_metrics))
    print(f"[Info] Used classes:    {used_classes}")
    print(f"[Info] Skipped classes: {skipped_classes}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"{args.feature_key}_lp{args.lambda_proto}_k{args.k_neighbors}_"
        f"ref{args.num_ref}_rtest{args.num_real_test}_ftest{args.num_fake_test}"
    )
    if args.normalize:
        tag += "_norm"

    per_class_csv_path = save_dir / f"per_class_metrics_{tag}.csv"
    metrics_json_path = save_dir / f"macro_metrics_{tag}.json"
    scores_npz_path = save_dir / f"scores_{tag}.npz"

    per_class_df.to_csv(per_class_csv_path, index=False)

    all_scores_arr = np.concatenate(all_scores, axis=0)
    all_labels_arr = np.concatenate(all_labels, axis=0)
    all_proto_scores_arr = np.concatenate(all_proto_scores, axis=0)
    all_knn_scores_arr = np.concatenate(all_knn_scores, axis=0)
    all_class_keys_arr = np.concatenate(all_class_keys, axis=0)
    all_paths_arr = np.concatenate(all_paths, axis=0)

    save_scores_npz(
        str(scores_npz_path),
        scores=all_scores_arr,
        proto_scores=all_proto_scores_arr,
        knn_scores=all_knn_scores_arr,
        labels=all_labels_arr,
        class_keys=all_class_keys_arr,
        paths=all_paths_arr,
        feature_key=np.array([args.feature_key], dtype=object),
        lambda_proto=np.array([args.lambda_proto], dtype=np.float64),
        k_neighbors=np.array([args.k_neighbors], dtype=np.int64),
        normalize=np.array([args.normalize], dtype=bool),
        num_ref=np.array([args.num_ref], dtype=np.int64),
        num_real_test=np.array([args.num_real_test], dtype=np.int64),
        num_fake_test=np.array([args.num_fake_test], dtype=np.int64),
    )

    payload = {
        "feature_key": args.feature_key,
        "class_key": args.class_key,
        "lambda_proto": args.lambda_proto,
        "k_neighbors": args.k_neighbors,
        "normalize": args.normalize,
        "num_ref": args.num_ref,
        "num_real_test": args.num_real_test,
        "num_fake_test": args.num_fake_test,
        "min_ref_samples": min_ref_samples,
        "used_classes": used_classes,
        "skipped_classes": skipped_classes,
        "macro_metrics": macro_metrics,
    }
    save_metrics_json(str(metrics_json_path), payload)

    print(f"[Info] Saved per-class table to: {per_class_csv_path}")
    print(f"[Info] Saved macro metrics to:   {metrics_json_path}")
    print(f"[Info] Saved scores to:          {scores_npz_path}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/run_per_class_prototype_knn.py \
#   --train-features artifacts/features/dinov2_train.npz \
#   --val-features artifacts/features/dinov2_val.npz \
#   --feature-key cls_features \
#   --class-key class_keys \
#   --num-ref 16 \
#   --num-real-test 4 \
#   --num-fake-test 4 \
#   --lambda-proto 0.5 \
#   --k-neighbors 1 \
#   --normalize \
#   --save-dir artifacts/per_class_prototype_knn