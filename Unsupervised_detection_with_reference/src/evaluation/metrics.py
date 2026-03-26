# src/evaluation/metrics.py

from __future__ import annotations
import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_fpr95(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "fpr95": float(compute_fpr95(y_true, y_score)),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"AUROC: {metrics['auroc']:.6f}\n"
        f"AUPRC: {metrics['auprc']:.6f}\n"
        f"FPR95: {metrics['fpr95']:.6f}"
    )