import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


LABEL_TO_ID = {"ai": 0, "nature": 1}
ID_TO_LABEL = {0: "ai", 1: "nature"}


class FlatImageFolder(Dataset):
    def __init__(self, folder, img_size=128):
        self.files = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in sorted(Path(folder).glob("*")):
            if p.suffix.lower() in exts:
                self.files.append(p)

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = self.tf(img)
        return img, str(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--detector_ckpt", type=str, required=True)
    parser.add_argument("--expected_label", type=str, default=None, choices=["ai", "nature"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = FlatImageFolder(args.img_dir, img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.detector_ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    preds = []
    files = []

    with torch.no_grad():
        for x, paths in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            files.extend(paths)

    preds = np.array(preds)
    pred_ai_ratio = float((preds == 0).mean())
    pred_nature_ratio = float((preds == 1).mean())

    print(f"num_images: {len(preds)}")
    print(f"pred_ai_ratio: {pred_ai_ratio:.4f}")
    print(f"pred_nature_ratio: {pred_nature_ratio:.4f}")

    if args.expected_label is not None:
        target = LABEL_TO_ID[args.expected_label]
        consistency = float((preds == target).mean())
        print(f"expected_label: {args.expected_label}")
        print(f"conditional_consistency: {consistency:.4f}")

    # 保存逐图结果
    out_txt = Path(args.img_dir) / "detector_predictions.txt"
    with open(out_txt, "w") as f:
        for p, y in zip(files, preds):
            f.write(f"{p}\t{ID_TO_LABEL[int(y)]}\n")

    print(f"saved to: {out_txt}")


if __name__ == "__main__":
    main()