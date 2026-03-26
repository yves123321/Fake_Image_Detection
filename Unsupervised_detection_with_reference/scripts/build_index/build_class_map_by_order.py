#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Set

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FAKE_FILENAME_PATTERN = re.compile(
    r"^VQDM_\d+_\d+_\d+_(\d{3})_vqdm_\d+\.[A-Za-z0-9]+$"
)
REAL_FILENAME_PATTERN = re.compile(
    r"^(n\d{8})_.+\.[A-Za-z0-9]+$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate class map by order assumption."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root, e.g. data/vqdm_subset_raw",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/index/class_map_by_order.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def parse_fake_class_id(filename: str) -> str:
    m = FAKE_FILENAME_PATTERN.match(filename)
    if m is None:
        raise ValueError(f"Cannot parse fake class id from filename: {filename}")
    return m.group(1)


def parse_real_synset(filename: str) -> str:
    m = REAL_FILENAME_PATTERN.match(filename)
    if m is None:
        raise ValueError(f"Cannot parse real synset from filename: {filename}")
    return m.group(1)


def collect_unique_fake_ids(ai_dir: Path) -> List[str]:
    fake_ids: Set[str] = set()
    for p in ai_dir.rglob("*"):
        if not is_image_file(p):
            continue
        fake_ids.add(parse_fake_class_id(p.name))
    return sorted(fake_ids)


def collect_unique_real_synsets(nature_dir: Path) -> List[str]:
    synsets: Set[str] = set()
    for p in nature_dir.rglob("*"):
        if not is_image_file(p):
            continue
        synsets.add(parse_real_synset(p.name))
    return sorted(synsets)


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    train_ai_dir = root / "train" / "ai"
    train_nature_dir = root / "train" / "nature"

    if not train_ai_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_ai_dir}")
    if not train_nature_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_nature_dir}")

    fake_ids = collect_unique_fake_ids(train_ai_dir)
    real_synsets = collect_unique_real_synsets(train_nature_dir)

    print(f"[Info] Unique fake class ids: {len(fake_ids)}")
    print(f"[Info] Unique real synsets:  {len(real_synsets)}")

    if len(fake_ids) != len(real_synsets):
        raise RuntimeError(
            "The number of unique fake class ids and real synsets does not match, "
            "so mapping by order is unsafe."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fake_class_id", "real_synset", "class_key"],
        )
        writer.writeheader()

        for fake_id, real_synset in zip(fake_ids, real_synsets):
            writer.writerow(
                {
                    "fake_class_id": fake_id,
                    "real_synset": real_synset,
                    "class_key": fake_id,
                }
            )

    print(f"[Info] Candidate class map saved to: {output_path}")
    # print("[Info] First 10 mappings:")
    # for fake_id, real_synset in list(zip(fake_ids, real_synsets))[:10]:
    #     print(f"  {fake_id} -> {real_synset}")


if __name__ == "__main__":
    main()