import os
import json
import time
import copy
import random
import argparse

import numpy as np
from PIL import ImageFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 128 -> 64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name, num_classes=2, pretrained=False):
    model_name = model_name.lower()

    if model_name == "smallcnn":
        return SmallCNN(num_classes=num_classes)

    elif model_name == "resnet18":
        try:
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(weights=None)

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def maybe_subset(dataset, max_samples, seed=42):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:max_samples]
    return Subset(dataset, indices)


def get_loaders(data_root, img_size, batch_size, num_workers, max_train, max_val, seed):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dir = os.path.join(os.path.expanduser(data_root), "train")
    val_dir = os.path.join(os.path.expanduser(data_root), "val")

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)

    print("class_to_idx:", train_set.class_to_idx)
    print("train size before subset:", len(train_set))
    print("val size before subset:", len(val_set))

    train_set = maybe_subset(train_set, max_train, seed=seed)
    val_set = maybe_subset(val_set, max_val, seed=seed)

    print("train size after subset:", len(train_set))
    print("val size after subset:", len(val_set))

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_set


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    if train:
        context = torch.enable_grad()
    else:
        context = torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_n += images.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./detector_ckpt_gpu")
    parser.add_argument("--model", type=str, default="resnet18", choices=["smallcnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and (not args.force_cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    train_loader, val_loader, train_set = get_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed
    )

    model = build_model(args.model, num_classes=2, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        elapsed = time.time() - start

        print(
            f"Epoch [{epoch}/{args.epochs}] | time {elapsed:.1f}s | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

            best_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            torch.save(best_state, best_path)
            print("Saved best model to:", best_path)

    last_path = os.path.join(args.save_dir, f"last_{args.model}.pth")
    torch.save(model.state_dict(), last_path)
    print("Saved last model to:", last_path)

    meta = {
        "class_to_idx": train_set.dataset.class_to_idx if isinstance(train_set, Subset) else train_set.class_to_idx,
        "best_val_acc": best_val_acc,
        "img_size": args.img_size,
        "model": args.model
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()