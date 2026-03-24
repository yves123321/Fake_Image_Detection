import os
import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms, models


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() in exts:
            files.append(p)
    return files


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        x = self.net(x)
        return x.flatten(1)


def load_and_preprocess(path, tf):
    img = Image.open(path).convert("RGB")
    return tf(img)


def make_panel(gen_path, ref_paths, out_path, thumb=128):
    imgs = [Image.open(gen_path).convert("RGB").resize((thumb, thumb))]
    imgs += [Image.open(p).convert("RGB").resize((thumb, thumb)) for p in ref_paths]

    canvas = Image.new("RGB", (thumb * len(imgs), thumb), color=(255, 255, 255))
    for i, img in enumerate(imgs):
        canvas.paste(img, (i * thumb, 0))
    canvas.save(out_path)


def batched_features(paths, model, tf, device, batch_size=128):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        xs = torch.stack([load_and_preprocess(p, tf) for p in batch_paths], dim=0).to(device)
        with torch.no_grad():
            f = model(xs)
            f = F.normalize(f, dim=1)
        feats.append(f.cpu())
        print(f"features: {min(i+batch_size, len(paths))}/{len(paths)}")
    return torch.cat(feats, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num_queries", type=int, default=20)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gen_paths = list_images(args.gen_dir)
    ref_paths = list_images(args.ref_dir)

    if len(gen_paths) == 0 or len(ref_paths) == 0:
        raise ValueError("Empty gen_dir or ref_dir")

    gen_paths = gen_paths[:args.num_queries]

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = FeatureExtractor().to(device)
    model.eval()

    print("Extracting reference features ...")
    ref_feats = batched_features(ref_paths, model, tf, device, batch_size=args.batch_size)

    print("Extracting generated features ...")
    gen_feats = batched_features(gen_paths, model, tf, device, batch_size=args.batch_size)

    sims = gen_feats @ ref_feats.T  # cosine similarity because normalized

    csv_path = outdir / "nearest_neighbors.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen_image", "rank", "ref_image", "cosine_similarity"])

        for i, gen_path in enumerate(gen_paths):
            sim = sims[i]
            topv, topi = torch.topk(sim, k=args.topk, largest=True)

            ref_match_paths = [ref_paths[idx] for idx in topi.tolist()]
            panel_path = outdir / f"panel_{i:03d}.png"
            make_panel(gen_path, ref_match_paths, panel_path)

            for rank, (idx, val) in enumerate(zip(topi.tolist(), topv.tolist()), start=1):
                writer.writerow([str(gen_path), rank, str(ref_paths[idx]), float(val)])

    print(f"saved csv to: {csv_path}")
    print(f"saved panels to: {outdir}")


if __name__ == "__main__":
    main()