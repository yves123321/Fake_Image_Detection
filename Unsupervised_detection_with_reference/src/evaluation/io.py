# src/evaluation/io.py

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any


def save_scores_npz(output_path: str, **kwargs) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **kwargs)


def save_metrics_json(output_path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)