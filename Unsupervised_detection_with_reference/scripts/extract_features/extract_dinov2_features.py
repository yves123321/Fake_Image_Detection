#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dataset import BinaryImageDataset
from src.datasets.transforms import build_basic_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features and save them to NPZ."
    )
    parser.add_argument(
        "--index-file",
        type=str,
        required=True,
        help="Path to index.csv",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root directory, e.g. data/vqdm_subset_raw",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="Which split to extract",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz path",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--save-fp16",
        action="store_true",
        help="Save features as float16 to reduce disk usage",
    )
    return parser.parse_args()


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    def collect(key: str) -> List[str]:
        return [str(item.get(key, "")) for item in batch]

    return {
        "image": images,
        "label": labels,
        "path": collect("path"),
        "filename": collect("filename"),
        "split": collect("split"),
        "label_name": collect("label_name"),
        "semantic_raw": collect("semantic_raw"),
        "fake_class_id": collect("fake_class_id"),
        "real_synset": collect("real_synset"),
        "class_key": collect("class_key"),
        "class_key_source": collect("class_key_source"),
    }


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    all_cls_features: List[np.ndarray] = []
    all_mean_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    all_paths: List[str] = []
    all_filenames: List[str] = []
    all_splits: List[str] = []
    all_label_names: List[str] = []

    all_semantic_raw: List[str] = []
    all_fake_class_id: List[str] = []
    all_real_synset: List[str] = []
    all_class_key: List[str] = []
    all_class_key_source: List[str] = []

    model.eval()
    use_amp = device.type == "cuda"

    for batch in tqdm(loader, desc="Extracting"):
        images = batch["image"].to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=images)
        else:
            outputs = model(pixel_values=images)

        last_hidden = outputs.last_hidden_state  # [B, N, D]

        cls_feat = last_hidden[:, 0, :]     # [B, D]
        mean_feat = last_hidden.mean(dim=1) # [B, D]

        all_cls_features.append(cls_feat.float().cpu().numpy())
        all_mean_features.append(mean_feat.float().cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

        all_paths.extend(batch["path"])
        all_filenames.extend(batch["filename"])
        all_splits.extend(batch["split"])
        all_label_names.extend(batch["label_name"])

        all_semantic_raw.extend(batch["semantic_raw"])
        all_fake_class_id.extend(batch["fake_class_id"])
        all_real_synset.extend(batch["real_synset"])
        all_class_key.extend(batch["class_key"])
        all_class_key_source.extend(batch["class_key_source"])

    cls_features = np.concatenate(all_cls_features, axis=0)
    mean_features = np.concatenate(all_mean_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return {
        "cls_features": cls_features,
        "mean_features": mean_features,
        "labels": labels,
        "paths": np.array(all_paths, dtype=object),
        "filenames": np.array(all_filenames, dtype=object),
        "splits": np.array(all_splits, dtype=object),
        "label_names": np.array(all_label_names, dtype=object),
        "semantic_raw": np.array(all_semantic_raw, dtype=object),
        "fake_class_id": np.array(all_fake_class_id, dtype=object),
        "real_synset": np.array(all_real_synset, dtype=object),
        "class_keys": np.array(all_class_key, dtype=object),
        "class_key_source": np.array(all_class_key_source, dtype=object),
    }


def save_npz(
    data: Dict[str, np.ndarray],
    output_path: Path,
    save_fp16: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = dict(data)

    if save_fp16:
        data_to_save["cls_features"] = data_to_save["cls_features"].astype(np.float16)
        data_to_save["mean_features"] = data_to_save["mean_features"].astype(np.float16)

    np.savez_compressed(output_path, **data_to_save)


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    print(f"[Info] Using device: {device}")
    if device.type == "cuda":
        print(f"[Info] GPU: {torch.cuda.get_device_name(0)}")

    transform = build_basic_transform(img_size=args.img_size)

    dataset = BinaryImageDataset(
        index_file=args.index_file,
        root=args.root,
        split=args.split,
        transform=transform,
        return_meta=True,
    )

    print(f"[Info] Dataset size ({args.split}): {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    print(f"[Info] Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    data = extract_features(model=model, loader=loader, device=device)

    output_path = Path(args.output)
    save_npz(data, output_path, save_fp16=args.save_fp16)

    print("[Info] Saved features:")
    print(f"  - cls_features: {data['cls_features'].shape}")
    print(f"  - mean_features: {data['mean_features'].shape}")
    print(f"  - labels: {data['labels'].shape}")
    print(f"  - class_keys: {data['class_keys'].shape}")
    print(f"[Info] Output file: {output_path}")


if __name__ == "__main__":
    main()