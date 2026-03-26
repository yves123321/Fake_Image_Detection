#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_command(cmd: List[str], log_path: Path) -> None:
    print("[Running]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as f:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def find_macro_json(save_dir: Path) -> Path:
    files = sorted(save_dir.glob("macro_metrics_*.json"))
    if len(files) == 0:
        raise FileNotFoundError(f"No macro_metrics_*.json found in {save_dir}")
    return files[-1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_experiment_name(params: Dict[str, Any]) -> str:
    parts = [
        f"feat-{params['feature_key']}",
        f"ref-{params['num_ref']}",
        f"rtest-{params['num_real_test']}",
        f"ftest-{params['num_fake_test']}",
        f"lp-{params['lambda_proto']}",
        f"k-{params['k_neighbors']}",
        f"norm-{int(params['normalize'])}",
    ]
    return "__".join(parts)


def extract_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload["macro_metrics"]
    return {
        "used_classes": payload.get("used_classes"),
        "skipped_classes": payload.get("skipped_classes"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "fpr95": metrics.get("fpr95"),
    }


def main() -> None:
    train_features = "artifacts/features/dinov2_multilayer_train.npz"
    val_features = "artifacts/features/dinov2_multilayer_val.npz"
    class_key = "class_keys"

    grid = {
        "feature_key": ["cls_layer_12"], 
        "num_ref": [16],
        "num_real_test": [4],
        "num_fake_test": [4],
        "lambda_proto": [0.1, 0.3, 0.5, 0.7, 0.9],
        "k_neighbors": [1],
        "normalize": [True],
    }

    output_root = ROOT / "artifacts" / "sweeps" / "per_class_prototype_knn_dinov2_diff_lambda"
    output_root.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    records: List[Dict[str, Any]] = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        exp_name = make_experiment_name(params)
        save_dir = output_root / exp_name
        save_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            PYTHON,
            "scripts/run_per_class_prototype_knn.py",
            "--train-features", train_features,
            "--val-features", val_features,
            "--feature-key", str(params["feature_key"]),
            "--class-key", class_key,
            "--num-ref", str(params["num_ref"]),
            "--num-real-test", str(params["num_real_test"]),
            "--num-fake-test", str(params["num_fake_test"]),
            "--lambda-proto", str(params["lambda_proto"]),
            "--k-neighbors", str(params["k_neighbors"]),
            "--save-dir", str(save_dir),
        ]

        if params["normalize"]:
            cmd.append("--normalize")

        log_path = save_dir / "run.log"

        try:
            run_command(cmd, log_path)
            macro_json = find_macro_json(save_dir)
            payload = load_json(macro_json)
            metric_info = extract_metrics(payload)

            record = {
                "exp_name": exp_name,
                **params,
                **metric_info,
                "save_dir": str(save_dir),
            }
            records.append(record)

        except Exception as e:
            print(f"[Error] Experiment failed: {exp_name}")
            print(e)
            records.append({
                "exp_name": exp_name,
                **params,
                "used_classes": None,
                "skipped_classes": None,
                "auroc": None,
                "auprc": None,
                "fpr95": None,
                "save_dir": str(save_dir),
                "error": str(e),
            })

    df = pd.DataFrame(records)
    if "auroc" in df.columns:
        df = df.sort_values(
            by=["auroc", "auprc"],
            ascending=[False, False],
            na_position="last",
        )

    summary_csv = output_root / "summary.csv"
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"[Done] Summary saved to: {summary_csv}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()