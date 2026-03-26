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
        description="Extract multi-layer DINOv2 CLS features and save them to NPZ."
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
        "--layers",
        type=int,
        nargs="+",
        default=[6, 9, 12],
        help=(
            "Transformer layer indices to extract CLS from. "
            "For DINOv2-base, typical choices are 6 9 12. "
            "These are 1-based hidden layer numbers (excluding embedding output)."
        ),
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
        help="Apply L2 normalization to all saved features",
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


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layers: List[int],
    l2_normalize: bool,
) -> Dict[str, np.ndarray]:
    """
    We use hidden_states from Hugging Face outputs.

    hidden_states[0]  : embedding output
    hidden_states[1]  : layer 1 output
    hidden_states[2]  : layer 2 output
    ...
    hidden_states[L]  : layer L output

    User-facing layer index is therefore 1-based for transformer blocks.
    """
    all_layer_features: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    all_concat_features: List[np.ndarray] = []
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

    for batch in tqdm(loader, desc="Extracting multi-layer DINOv2"):
        images = batch["image"].to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=images, output_hidden_states=True)
        else:
            outputs = model(pixel_values=images, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states. Check output_hidden_states=True.")

        batch_layer_features: List[np.ndarray] = []

        for layer in layers:
            if layer < 1 or layer >= len(hidden_states):
                raise ValueError(
                    f"Requested layer={layer}, but available transformer layers are 1..{len(hidden_states)-1}"
                )

            h = hidden_states[layer]          # [B, N, D]
            cls = h[:, 0, :]                  # [B, D]
            cls_np = cls.float().cpu().numpy()

            if l2_normalize:
                cls_np = l2_normalize_np(cls_np)

            all_layer_features[layer].append(cls_np)
            batch_layer_features.append(cls_np)

        concat_np = np.concatenate(batch_layer_features, axis=1)
        mean_np = np.mean(np.stack(batch_layer_features, axis=0), axis=0)

        if l2_normalize:
            concat_np = l2_normalize_np(concat_np)
            mean_np = l2_normalize_np(mean_np)

        all_concat_features.append(concat_np)
        all_mean_features.append(mean_np)
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

    result: Dict[str, np.ndarray] = {}

    for layer in layers:
        result[f"cls_layer_{layer}"] = np.concatenate(all_layer_features[layer], axis=0)

    result["concat_cls_layers"] = np.concatenate(all_concat_features, axis=0)
    result["mean_cls_layers"] = np.concatenate(all_mean_features, axis=0)
    result["labels"] = np.concatenate(all_labels, axis=0)

    result["paths"] = np.array(all_paths, dtype=object)
    result["filenames"] = np.array(all_filenames, dtype=object)
    result["splits"] = np.array(all_splits, dtype=object)
    result["label_names"] = np.array(all_label_names, dtype=object)

    result["semantic_raw"] = np.array(all_semantic_raw, dtype=object)
    result["fake_class_id"] = np.array(all_fake_class_id, dtype=object)
    result["real_synset"] = np.array(all_real_synset, dtype=object)
    result["class_keys"] = np.array(all_class_key, dtype=object)
    result["class_key_source"] = np.array(all_class_key_source, dtype=object)

    return result


def save_npz(
    data: Dict[str, np.ndarray],
    output_path: Path,
    save_fp16: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = dict(data)

    if save_fp16:
        for key, value in list(data_to_save.items()):
            if isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64):
                if key.startswith("cls_layer_") or key in {"concat_cls_layers", "mean_cls_layers"}:
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
    print(f"[Info] Layers to extract: {args.layers}")

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
        layers=args.layers,
        l2_normalize=args.l2_normalize,
    )

    output_path = Path(args.output)
    save_npz(data, output_path, save_fp16=args.save_fp16)

    print("[Info] Saved features:")
    for layer in args.layers:
        key = f"cls_layer_{layer}"
        print(f"  - {key}: {data[key].shape}")
    print(f"  - concat_cls_layers: {data['concat_cls_layers'].shape}")
    print(f"  - mean_cls_layers:   {data['mean_cls_layers'].shape}")
    print(f"  - labels:            {data['labels'].shape}")
    print(f"  - class_keys:        {data['class_keys'].shape}")
    print(f"[Info] Output file: {output_path}")


if __name__ == "__main__":
    main()