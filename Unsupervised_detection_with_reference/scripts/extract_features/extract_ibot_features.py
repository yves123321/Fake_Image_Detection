#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.dataset import BinaryImageDataset
from src.datasets.transforms import build_basic_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract iBOT features and save them to NPZ."
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
        "--arch",
        type=str,
        default="vit_base_patch16_224",
        help="ViT backbone arch in timm, e.g. vit_small_patch16_224 / vit_base_patch16_224",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to iBOT checkpoint (.pth). Prefer official extracted backbone weights.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "teacher", "student", "model"],
        help="Which key to read from checkpoint. Use 'auto' in most cases.",
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
    parser.add_argument(
        "--save-patch-tokens",
        action="store_true",
        help="Also save patch tokens [N, P, D] (can be very large on disk)",
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


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in prefixes:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        out[new_k] = v
    return out


def _extract_state_dict(ckpt: Dict[str, Any], checkpoint_key: str) -> Dict[str, torch.Tensor]:
    # Most common cases for official / converted checkpoints
    if checkpoint_key != "auto":
        if checkpoint_key in ckpt and isinstance(ckpt[checkpoint_key], dict):
            return ckpt[checkpoint_key]
        if checkpoint_key == "model" and isinstance(ckpt, dict):
            return ckpt

    # auto mode
    candidate_keys = ["teacher", "student", "model", "state_dict"]
    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    # already a plain state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise ValueError(
        "Cannot locate a valid state_dict in checkpoint. "
        "Try --checkpoint-key teacher or --checkpoint-key student."
    )


def load_ibot_model(
    arch: str,
    checkpoint_path: str,
    checkpoint_key: str,
    device: torch.device,
) -> torch.nn.Module:
    model = timm.create_model(
        arch,
        pretrained=False,
        num_classes=0,
        img_size=224,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt, checkpoint_key)

    # Common prefixes seen in self-supervised checkpoints
    state_dict = _strip_prefix_if_present(
        state_dict,
        prefixes=[
            "module.",
            "backbone.",
            "encoder.",
        ],
    )

    # Remove head / projector params if present
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("head."):
            continue
        if k.startswith("fc."):
            continue
        if "projection_head" in k:
            continue
        if "prototypes" in k:
            continue
        if "ibot_head" in k:
            continue
        if "classifier" in k:
            continue
        filtered_state_dict[k] = v

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    print("[Info] Loaded checkpoint.")
    print(f"[Info] Missing keys: {len(missing)}")
    print(f"[Info] Unexpected keys: {len(unexpected)}")

    if len(missing) > 20:
        print("[Warn] Too many missing keys. Please double-check arch/checkpoint compatibility.")

    model.to(device)
    model.eval()
    return model


def forward_tokens_vit(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Forward images through a timm ViT and return token sequence [B, N, D].
    This is more suitable than model(images) because we want cls / mean / patch tokens.
    """
    x = model.patch_embed(images)

    if hasattr(model, "_pos_embed"):
        x = model._pos_embed(x)
    else:
        # fallback for some timm versions
        if hasattr(model, "cls_token") and model.cls_token is not None:
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            x = x + model.pos_embed

    if hasattr(model, "patch_drop"):
        x = model.patch_drop(x)

    if hasattr(model, "norm_pre"):
        x = model.norm_pre(x)

    x = model.blocks(x)

    if hasattr(model, "norm"):
        x = model.norm(x)

    return x


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_patch_tokens: bool,
) -> Dict[str, np.ndarray]:
    all_cls_features: List[np.ndarray] = []
    all_mean_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    all_patch_tokens: List[np.ndarray] = []

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
                tokens = forward_tokens_vit(model, images)
        else:
            tokens = forward_tokens_vit(model, images)

        # tokens: [B, N, D], first token is cls
        cls_feat = tokens[:, 0, :]          # [B, D]
        mean_feat = tokens.mean(dim=1)      # [B, D]
        patch_feat = tokens[:, 1:, :]       # [B, P, D]

        all_cls_features.append(cls_feat.float().cpu().numpy())
        all_mean_features.append(mean_feat.float().cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

        if save_patch_tokens:
            all_patch_tokens.append(patch_feat.float().cpu().numpy())

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

    result = {
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

    if save_patch_tokens:
        result["patch_tokens"] = np.concatenate(all_patch_tokens, axis=0)

    return result


def save_npz(
    data: Dict[str, np.ndarray],
    output_path: Path,
    save_fp16: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = dict(data)

    if save_fp16:
        if "cls_features" in data_to_save:
            data_to_save["cls_features"] = data_to_save["cls_features"].astype(np.float16)
        if "mean_features" in data_to_save:
            data_to_save["mean_features"] = data_to_save["mean_features"].astype(np.float16)
        if "patch_tokens" in data_to_save:
            data_to_save["patch_tokens"] = data_to_save["patch_tokens"].astype(np.float16)

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

    print(f"[Info] Loading iBOT backbone: arch={args.arch}")
    print(f"[Info] Checkpoint: {args.checkpoint}")

    model = load_ibot_model(
        arch=args.arch,
        checkpoint_path=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        device=device,
    )

    data = extract_features(
        model=model,
        loader=loader,
        device=device,
        save_patch_tokens=args.save_patch_tokens,
    )

    output_path = Path(args.output)
    save_npz(data, output_path, save_fp16=args.save_fp16)

    print("[Info] Saved features:")
    print(f"  - cls_features: {data['cls_features'].shape}")
    print(f"  - mean_features: {data['mean_features'].shape}")
    if "patch_tokens" in data:
        print(f"  - patch_tokens: {data['patch_tokens'].shape}")
    print(f"  - labels: {data['labels'].shape}")
    print(f"  - class_keys: {data['class_keys'].shape}")
    print(f"[Info] Output file: {output_path}")


if __name__ == "__main__":
    main()