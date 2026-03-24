import os
import csv
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True,
                        help="generated images root with class subfolders")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_csv", type=str, default="multiclass_consistency.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = datasets.ImageFolder(
        root=os.path.expanduser(args.img_root),
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    print("class_to_idx:", ds.class_to_idx)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    n_classes = len(ds.class_to_idx)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(torch.load(os.path.expanduser(args.classifier_ckpt), map_location=device))
    model = model.to(device)
    model.eval()

    total = 0
    correct = 0
    per_cls_total = [0] * n_classes
    per_cls_correct = [0] * n_classes

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            total += y.size(0)
            correct += (pred == y).sum().item()

            for c in range(n_classes):
                mask = (y == c)
                per_cls_total[c] += mask.sum().item()
                per_cls_correct[c] += ((pred == y) & mask).sum().item()

    overall = correct / max(total, 1)
    print("overall_conditional_consistency:", overall)

    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "accuracy", "correct", "total"])
        for c in range(n_classes):
            acc = per_cls_correct[c] / max(per_cls_total[c], 1)
            print(idx_to_class[c], acc, per_cls_correct[c], per_cls_total[c])
            writer.writerow([idx_to_class[c], acc, per_cls_correct[c], per_cls_total[c]])

    print("saved:", args.out_csv)


if __name__ == "__main__":
    main()