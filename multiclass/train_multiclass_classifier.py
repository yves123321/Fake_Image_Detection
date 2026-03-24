import os
import json
import time
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


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
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            total_correct += (pred == y).sum().item()
            total_n += x.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./cls_ai10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = datasets.ImageFolder(
        root=os.path.join(os.path.expanduser(args.data_root), "train"),
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    val_set = datasets.ImageFolder(
        root=os.path.join(os.path.expanduser(args.data_root), "val"),
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )

    print("class_to_idx:", train_set.class_to_idx)
    print("train size:", len(train_set))
    print("val size:", len(val_set))

    with open(os.path.join(args.save_dir, "class_to_idx.json"), "w") as f:
        json.dump(train_set.class_to_idx, f, indent=2)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    try:
        if args.pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, len(train_set.class_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print("Epoch [{}/{}] time {:.1f}s train_loss {:.4f} train_acc {:.4f} val_loss {:.4f} val_acc {:.4f}".format(
            epoch, args.epochs, time.time() - start, train_loss, train_acc, val_loss, val_acc
        ))

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(args.save_dir, "best_resnet18.pth"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "last_resnet18.pth"))
    print("best val acc:", best_acc)


if __name__ == "__main__":
    main()