#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a CSV index for the VQDM subset dataset.

Output CSV columns:
- path
- split
- label
- label_name
- filename
- semantic_raw
- fake_class_id
- real_synset
- class_key
- class_key_source

Notes:
1. For ai images:
   example filename: VQDM_1000_200_00_001_vqdm_00035.png
   -> fake_class_id = "001"

2. For nature images:
   example filename: n01440764_10027.JPEG
   -> real_synset = "n01440764"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_MAP = {
    "nature": 0,
    "ai": 1,
}

FAKE_FILENAME_PATTERN = re.compile(
    r"^VQDM_\d+_\d+_\d+_(\d{3})_vqdm_\d+\.[A-Za-z0-9]+$"
)
REAL_FILENAME_PATTERN = re.compile(
    r"^(n\d{8})_.+\.[A-Za-z0-9]+$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset index CSV with semantic class information.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of dataset, e.g. data/vqdm_subset_raw",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/index/index.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Store relative paths to --root instead of absolute paths",
    )
    parser.add_argument(
        "--check-exists",
        action="store_true",
        help="Extra sanity check for file existence before writing",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help=(
            "Mapping file (CSV or JSON) to unify ai fake_class_id and "
            "nature real_synset into one shared class_key."
        ),
    )
    parser.add_argument(
        "--strict-parse",
        action="store_true",
        help="Raise error if semantic parsing fails. Otherwise fill empty fields and warn.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def parse_fake_class_id(filename: str) -> str:
    """
    Example:
        VQDM_1000_200_00_001_vqdm_00035.png -> 001
    """
    m = FAKE_FILENAME_PATTERN.match(filename)
    if m is None:
        raise ValueError(f"Cannot parse fake class id from filename: {filename}")
    return m.group(1)


def parse_real_synset(filename: str) -> str:
    """
    Example:
        n01440764_10027.JPEG -> n01440764
    """
    m = REAL_FILENAME_PATTERN.match(filename)
    if m is None:
        raise ValueError(f"Cannot parse real synset from filename: {filename}")
    return m.group(1)


def load_class_map(path: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Return:
        fake_to_class_key
        real_to_class_key
        fake_to_real_synset

    Supported formats:
    1. CSV with columns:
       fake_class_id, real_synset, class_key
    2. JSON:
       {
         "000": {"real_synset": "n01440764", "class_key": "000"},
         "001": {"real_synset": "n01443537", "class_key": "001"}
       }

       or
       [
         {"fake_class_id": "000", "real_synset": "n01440764", "class_key": "000"},
         ...
       ]
    """
    if path is None:
        return {}, {}, {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Class map file not found: {p}")

    fake_to_class_key: Dict[str, str] = {}
    real_to_class_key: Dict[str, str] = {}
    fake_to_real_synset: Dict[str, str] = {}

    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"fake_class_id", "real_synset", "class_key"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV class map must contain columns {required}, got {reader.fieldnames}"
                )

            for row in reader:
                fake_id = str(row["fake_class_id"]).strip()
                real_synset = str(row["real_synset"]).strip()
                class_key = str(row["class_key"]).strip()

                fake_to_class_key[fake_id] = class_key
                real_to_class_key[real_synset] = class_key
                fake_to_real_synset[fake_id] = real_synset

    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for fake_id, item in data.items():
                real_synset = str(item["real_synset"]).strip()
                class_key = str(item.get("class_key", fake_id)).strip()
                fake_id = str(fake_id).strip()

                fake_to_class_key[fake_id] = class_key
                real_to_class_key[real_synset] = class_key
                fake_to_real_synset[fake_id] = real_synset

        elif isinstance(data, list):
            for item in data:
                fake_id = str(item["fake_class_id"]).strip()
                real_synset = str(item["real_synset"]).strip()
                class_key = str(item.get("class_key", fake_id)).strip()

                fake_to_class_key[fake_id] = class_key
                real_to_class_key[real_synset] = class_key
                fake_to_real_synset[fake_id] = real_synset
        else:
            raise ValueError("Unsupported JSON class map format")

    else:
        raise ValueError(f"Unsupported class map format: {p.suffix}")

    return fake_to_class_key, real_to_class_key, fake_to_real_synset


def resolve_semantic_info(
    label_name: str,
    filename: str,
    fake_to_class_key: Dict[str, str],
    real_to_class_key: Dict[str, str],
    strict_parse: bool,
) -> Dict[str, str]:
    """
    Returns:
        semantic_raw
        fake_class_id
        real_synset
        class_key
        class_key_source
    """
    semantic_raw = ""
    fake_class_id = ""
    real_synset = ""
    class_key = ""
    class_key_source = ""

    try:
        if label_name == "ai":
            fake_class_id = parse_fake_class_id(filename)
            semantic_raw = fake_class_id

            if fake_class_id in fake_to_class_key:
                class_key = fake_to_class_key[fake_class_id]
                class_key_source = "mapped_from_fake_class_id"
            else:
                class_key = fake_class_id
                class_key_source = "fake_class_id_raw"

        elif label_name == "nature":
            real_synset = parse_real_synset(filename)
            semantic_raw = real_synset

            if real_synset in real_to_class_key:
                class_key = real_to_class_key[real_synset]
                class_key_source = "mapped_from_real_synset"
            else:
                class_key = real_synset
                class_key_source = "real_synset_raw"

        else:
            raise ValueError(f"Unsupported label_name: {label_name}")

    except Exception as e:
        if strict_parse:
            raise
        print(f"[Warning] {e}", file=sys.stderr)

    return {
        "semantic_raw": semantic_raw,
        "fake_class_id": fake_class_id,
        "real_synset": real_synset,
        "class_key": class_key,
        "class_key_source": class_key_source,
    }


def collect_split(
    root: Path,
    split: str,
    label_map: Dict[str, int],
    use_relative: bool,
    check_exists: bool,
    fake_to_class_key: Dict[str, str],
    real_to_class_key: Dict[str, str],
    strict_parse: bool,
) -> List[Dict[str, str]]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    rows: List[Dict[str, str]] = []

    for label_name, label_id in label_map.items():
        class_dir = split_dir / label_name
        if not class_dir.exists():
            print(f"[Warning] Missing directory: {class_dir}", file=sys.stderr)
            continue

        image_paths = sorted([p for p in class_dir.rglob("*") if is_image_file(p)])

        if len(image_paths) == 0:
            print(f"[Warning] No images found in: {class_dir}", file=sys.stderr)

        for img_path in image_paths:
            if check_exists and not img_path.exists():
                print(f"[Warning] File does not exist: {img_path}", file=sys.stderr)
                continue

            stored_path = img_path.relative_to(root) if use_relative else img_path.resolve()

            semantic_info = resolve_semantic_info(
                label_name=label_name,
                filename=img_path.name,
                fake_to_class_key=fake_to_class_key,
                real_to_class_key=real_to_class_key,
                strict_parse=strict_parse,
            )

            rows.append(
                {
                    "path": str(stored_path),
                    "split": split,
                    "label": str(label_id),
                    "label_name": label_name,
                    "filename": img_path.name,
                    "semantic_raw": semantic_info["semantic_raw"],
                    "fake_class_id": semantic_info["fake_class_id"],
                    "real_synset": semantic_info["real_synset"],
                    "class_key": semantic_info["class_key"],
                    "class_key_source": semantic_info["class_key_source"],
                }
            )

    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "split",
        "label",
        "label_name",
        "filename",
        "semantic_raw",
        "fake_class_id",
        "real_synset",
        "class_key",
        "class_key_source",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, str]]) -> None:
    total = len(rows)
    print(f"[Info] Total indexed images: {total}")

    by_split: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    by_split_label: Dict[Tuple[str, str], int] = {}
    class_keys_ai = set()
    class_keys_real = set()
    missing_class_key = 0

    for row in rows:
        split = row["split"]
        label_name = row["label_name"]
        class_key = row["class_key"]

        by_split[split] = by_split.get(split, 0) + 1
        by_label[label_name] = by_label.get(label_name, 0) + 1
        by_split_label[(split, label_name)] = by_split_label.get((split, label_name), 0) + 1

        if not class_key:
            missing_class_key += 1

        if label_name == "ai" and class_key:
            class_keys_ai.add(class_key)
        if label_name == "nature" and class_key:
            class_keys_real.add(class_key)

    print("[Info] Count by split:")
    for split, count in sorted(by_split.items()):
        print(f"  - {split}: {count}")

    print("[Info] Count by label:")
    for label_name, count in sorted(by_label.items()):
        print(f"  - {label_name}: {count}")

    print("[Info] Count by split and label:")
    for (split, label_name), count in sorted(by_split_label.items()):
        print(f"  - {split} / {label_name}: {count}")

    print("[Info] Semantic summary:")
    print(f"  - unique class_key in ai:     {len(class_keys_ai)}")
    print(f"  - unique class_key in nature: {len(class_keys_real)}")
    print(f"  - intersection:               {len(class_keys_ai & class_keys_real)}")
    print(f"  - missing class_key rows:     {missing_class_key}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    fake_to_class_key, real_to_class_key, fake_to_real_synset = load_class_map(args.class_map)

    if args.class_map is not None:
        print(f"[Info] Loaded class map from: {args.class_map}")
        print(f"[Info] fake ids mapped:   {len(fake_to_class_key)}")
        print(f"[Info] real synsets mapped: {len(real_to_class_key)}")

    all_rows: List[Dict[str, str]] = []
    for split in ["train", "val"]:
        rows = collect_split(
            root=root,
            split=split,
            label_map=LABEL_MAP,
            use_relative=args.relative,
            check_exists=args.check_exists,
            fake_to_class_key=fake_to_class_key,
            real_to_class_key=real_to_class_key,
            strict_parse=args.strict_parse,
        )
        all_rows.extend(rows)

    if len(all_rows) == 0:
        raise RuntimeError("No images were indexed. Please check dataset structure and file extensions.")

    write_csv(all_rows, Path(args.output))
    print_summary(all_rows)
    print(f"[Info] CSV saved to: {args.output}")


if __name__ == "__main__":
    main()