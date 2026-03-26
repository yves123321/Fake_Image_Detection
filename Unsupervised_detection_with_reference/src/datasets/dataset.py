# src/datasets/dataset.py

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from PIL import Image
from torch.utils.data import Dataset


class BinaryImageDataset(Dataset):
    """
    Dataset for binary classification / anomaly detection on VQDM subset.

    Expected CSV columns:
        - path
        - split
        - label
        - label_name
        - filename

    Optional semantic columns:
        - semantic_raw
        - fake_class_id
        - real_synset
        - class_key
        - class_key_source
    """

    def __init__(
        self,
        index_file: str,
        root: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_meta: bool = True,
    ):
        self.index_file = Path(index_file)
        self.root = Path(root) if root is not None else None
        self.split = split
        self.transform = transform
        self.return_meta = return_meta

        self.df = pd.read_csv(self.index_file)

        required_columns = {"path", "split", "label", "label_name", "filename"}
        missing = required_columns - set(self.df.columns)
        if len(missing) > 0:
            raise ValueError(
                f"Missing required columns in index file: {sorted(missing)}"
            )

        if self.split != "all":
            self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"No samples found for split={self.split}")

        self.optional_columns: List[str] = [
            "semantic_raw",
            "fake_class_id",
            "real_synset",
            "class_key",
            "class_key_source",
        ]

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)

        if not path.is_absolute():
            if self.root is None:
                raise ValueError(
                    f"Relative path detected but root is None: {path}"
                )
            path = self.root / path

        return path

    @staticmethod
    def _safe_get(row: pd.Series, key: str, default: str = "") -> str:
        if key not in row.index:
            return default
        value = row[key]
        if pd.isna(value):
            return default
        return str(value)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_path = self._resolve_path(str(row["path"]))

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])

        sample: Dict[str, Any] = {
            "image": image,
            "label": label,
        }

        if self.return_meta:
            sample.update(
                {
                    "path": str(img_path),
                    "label_name": str(row["label_name"]),
                    "filename": str(row["filename"]),
                    "split": str(row["split"]),
                }
            )

            for key in self.optional_columns:
                sample[key] = self._safe_get(row, key, default="")

        return sample


def build_dataset(
    index_file: str,
    root: str,
    split: str,
    transform: Optional[Callable] = None,
) -> BinaryImageDataset:
    return BinaryImageDataset(
        index_file=index_file,
        root=root,
        split=split,
        transform=transform,
        return_meta=True,
    )