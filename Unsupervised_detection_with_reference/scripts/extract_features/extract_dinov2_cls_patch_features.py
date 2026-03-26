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
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dataset import BinaryImageDataset
from src.datasets.transforms import build_basic_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 CLS + patch-token pooled features and save them to NPZ."
    )
    parser.add_argument("--index-file", type=str, required=True, help="Path to index.csv")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="Which split to extract",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face model name",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Transformer layer index to extract from, 1-based. For dinov2-base, last layer is usually 12.",
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="Apply L2 normalization to saved features",
    )
    parser.add_argument(
        "--save-fp16",
        action="store_true",
        help="Save feature arrays as float16",
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


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer: int,
    l2_normalize: bool,
) -> Dict[str, np.ndarray]:
    all_cls: List[np.ndarray] = []
    all_patch_mean: List[np.ndarray] = []
    all_patch_max: List[np.ndarray] = []
    all_concat_mean: List[np.ndarray] = []
    all_concat_max: List[np.ndarray] = []
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

    for batch in tqdm(loader, desc="Extracting DINOv2 CLS+Patch"):
        images = batch["image"].to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=images, output_hidden_states=True)
        else:
            outputs = model(pixel_values=images, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")

        if layer < 1 or layer >= len(hidden_states):
            raise ValueError(
                f"Requested layer={layer}, but available transformer layers are 1..{len(hidden_states)-1}"
            )

        h = hidden_states[layer]      # [B, N, D]
        cls = h[:, 0, :]              # [B, D]
        patch = h[:, 1:, :]           # [B, P, D]

        patch_mean = patch.mean(dim=1)            # [B, D]
        patch_max = patch.max(dim=1).values       # [B, D]

        cls_np = cls.float().cpu().numpy()
        patch_mean_np = patch_mean.float().cpu().numpy()
        patch_max_np = patch_max.float().cpu().numpy()

        concat_mean_np = np.concatenate([cls_np, patch_mean_np], axis=1)
        concat_max_np = np.concatenate([cls_np, patch_max_np], axis=1)

        if l2_normalize:
            cls_np = l2_normalize_np(cls_np)
            patch_mean_np = l2_normalize_np(patch_mean_np)
            patch_max_np = l2_normalize_np(patch_max_np)
            concat_mean_np = l2_normalize_np(concat_mean_np)
            concat_max_np = l2_normalize_np(concat_max_np)

        all_cls.append(cls_np)
        all_patch_mean.append(patch_mean_np)
        all_patch_max.append(patch_max_np)
        all_concat_mean.append(concat_mean_np)
        all_concat_max.append(concat_max_np)
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

    return {
        f"cls_layer_{layer}": np.concatenate(all_cls, axis=0),
        f"patch_mean_layer_{layer}": np.concatenate(all_patch_mean, axis=0),
        f"patch_max_layer_{layer}": np.concatenate(all_patch_max, axis=0),
        f"concat_cls_patchmean_layer_{layer}": np.concatenate(all_concat_mean, axis=0),
        f"concat_cls_patchmax_layer_{layer}": np.concatenate(all_concat_max, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
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


def save_npz(data: Dict[str, np.ndarray], output_path: Path, save_fp16: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_to_save = dict(data)

    if save_fp16:
        for key, value in list(data_to_save.items()):
            if isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64):
                if (
                    key.startswith("cls_layer_")
                    or key.startswith("patch_mean_layer_")
                    or key.startswith("patch_max_layer_")
                    or key.startswith("concat_cls_patchmean_layer_")
                    or key.startswith("concat_cls_patchmax_layer_")
                ):
                    data_to_save[key] = value.astype(np.float16)

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
    print(f"[Info] Extract layer: {args.layer}")

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

    data = extract_features(
        model=model,
        loader=loader,
        device=device,
        layer=args.layer,
        l2_normalize=args.l2_normalize,
    )

    output_path = Path(args.output)
    save_npz(data, output_path, save_fp16=args.save_fp16)

    print("[Info] Saved features:")
    print(f"  - cls_layer_{args.layer}: {data[f'cls_layer_{args.layer}'].shape}")
    print(f"  - patch_mean_layer_{args.layer}: {data[f'patch_mean_layer_{args.layer}'].shape}")
    print(f"  - patch_max_layer_{args.layer}: {data[f'patch_max_layer_{args.layer}'].shape}")
    print(f"  - concat_cls_patchmean_layer_{args.layer}: {data[f'concat_cls_patchmean_layer_{args.layer}'].shape}")
    print(f"  - concat_cls_patchmax_layer_{args.layer}: {data[f'concat_cls_patchmax_layer_{args.layer}'].shape}")
    print(f"  - labels: {data['labels'].shape}")
    print(f"[Info] Output file: {output_path}")


if __name__ == "__main__":
    main()