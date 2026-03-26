#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dataset import BinaryImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SigLIP image features and save them to NPZ."
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
        default="google/siglip-base-patch16-224",
        help="Hugging Face model name",
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
        default=0,
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
        "--l2-normalize",
        action="store_true",
        help="Apply L2 normalization before saving features",
    )
    parser.add_argument(
        "--save-fp16",
        action="store_true",
        help="Save features as float16 to reduce disk usage",
    )
    return parser.parse_args()


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [item["image"] for item in batch]

    def collect(key: str) -> List[str]:
        return [str(item.get(key, "")) for item in batch]

    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)

    return {
        "images": images,
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


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    processor: AutoProcessor,
    loader: DataLoader,
    device: torch.device,
    l2_normalize: bool,
) -> Dict[str, np.ndarray]:
    all_features: List[np.ndarray] = []
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

    for batch in tqdm(loader, desc="Extracting SigLIP"):
        pil_images = batch["images"]

        inputs = processor(
            images=pil_images,
            return_tensors="pt",
        )

        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model.get_image_features(**inputs)
        else:
            features = model.get_image_features(**inputs)

        features = features.pooler_output.float().cpu().numpy()

        if l2_normalize:
            features = l2_normalize_np(features)

        all_features.append(features)
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

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return {
        "siglip_features": features,
        "cls_features": features,
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
        for key in ["siglip_features", "cls_features"]:
            if key in data_to_save:
                data_to_save[key] = data_to_save[key].astype(np.float16)

    np.savez_compressed(output_path, **data_to_save)


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    print(f"[Info] Using device: {device}")
    if device.type == "cuda":
        print(f"[Info] GPU: {torch.cuda.get_device_name(0)}")

    dataset = BinaryImageDataset(
        index_file=args.index_file,
        root=args.root,
        split=args.split,
        transform=None,
        return_meta=True,
    )

    print(f"[Info] Dataset size ({args.split}): {len(dataset)}")
    print(f"[Info] Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)

    print(f"[Info] Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    data = extract_features(
        model=model,
        processor=processor,
        loader=loader,
        device=device,
        l2_normalize=args.l2_normalize,
    )

    output_path = Path(args.output)
    save_npz(data, output_path, save_fp16=args.save_fp16)

    print("[Info] Saved features:")
    print(f"  - siglip_features: {data['siglip_features'].shape}")
    print(f"  - cls_features:    {data['cls_features'].shape}")
    print(f"  - labels:          {data['labels'].shape}")
    print(f"  - class_keys:      {data['class_keys'].shape}")
    print(f"[Info] Output file: {output_path}")


if __name__ == "__main__":
    main()