import os
import csv
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

from pytorch_fid.fid_score import calculate_fid_given_paths


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def append_csv(row_dict, csv_path):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    files = []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix in exts:
            files.append(p)
    return files


class FolderDataset(Dataset):
    def __init__(self, folder, img_size=64, data_frac=1.0, seed=42):
        files = list_images(folder)
        if data_frac < 0.999:
            rng = random.Random(seed)
            rng.shuffle(files)
            k = max(1, int(round(len(files) * data_frac)))
            files = files[:k]
        self.files = files
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.tf(img)
        return img


def get_channels(img_size):
    if img_size == 32:
        return dict(g_start=256, g_mid=[128, 64], d_mid=[64, 128, 256], feat=256)
    elif img_size == 64:
        return dict(g_start=512, g_mid=[256, 128, 64], d_mid=[64, 128, 256, 512], feat=512)
    else:
        raise ValueError("img_size must be 32 or 64")


class Generator(nn.Module):
    def __init__(self, img_size=64, nz=100):
        super().__init__()
        cfg = get_channels(img_size)
        self.fc = nn.Linear(nz, cfg["g_start"] * 4 * 4)

        layers = [nn.BatchNorm2d(cfg["g_start"]), nn.ReLU(True)]
        in_ch = cfg["g_start"]
        for out_ch in cfg["g_mid"]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
            in_ch = out_ch

        layers += [nn.ConvTranspose2d(in_ch, 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        cfg = get_channels(img_size)

        layers = []
        in_ch = 3
        for i, out_ch in enumerate(cfg["d_mid"]):
            if i == 0:
                layers += [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(cfg["feat"], 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.features(x)).view(-1)


def train_one_domain(train_dir, save_dir, img_size, data_frac, epochs, batch_size, nz, lr, num_workers, seed, device):
    ensure_dir(save_dir)
    ds = FolderDataset(train_dir, img_size=img_size, data_frac=data_frac, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    G = Generator(img_size=img_size, nz=nz).to(device)
    D = Discriminator(img_size=img_size).to(device)

    criterion = nn.BCELoss()
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_z = torch.randn(32, nz, device=device)
    history = []

    for epoch in range(1, epochs + 1):
        d_avg, g_avg = 0.0, 0.0

        for real in loader:
            real = real.to(device)
            bsz = real.size(0)

            ones = torch.ones(bsz, device=device)
            zeros = torch.zeros(bsz, device=device)

            # D
            optD.zero_grad()
            out_real = D(real)
            loss_real = criterion(out_real, ones)

            z = torch.randn(bsz, nz, device=device)
            fake = G(z)
            out_fake = D(fake.detach())
            loss_fake = criterion(out_fake, zeros)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            optD.step()

            # G
            optG.zero_grad()
            out_fake2 = D(fake)
            g_loss = criterion(out_fake2, ones)
            g_loss.backward()
            optG.step()

            d_avg += d_loss.item()
            g_avg += g_loss.item()

        d_avg /= len(loader)
        g_avg /= len(loader)
        history.append({"epoch": epoch, "d_loss": d_avg, "g_loss": g_avg})
        print(f"[{save_dir}] epoch {epoch}/{epochs} D={d_avg:.4f} G={g_avg:.4f}")

        if epoch == epochs or epoch % 5 == 0:
            with torch.no_grad():
                fake = G(fixed_z).cpu()
            utils.save_image(fake, os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png"), normalize=True, nrow=8)

    torch.save(G.state_dict(), os.path.join(save_dir, "G_last.pth"))
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return G, len(ds)


def save_samples(G, outdir, n, batch_size, nz, device):
    ensure_dir(outdir)
    generated = 0
    with torch.no_grad():
        while generated < n:
            cur = min(batch_size, n - generated)
            z = torch.randn(cur, nz, device=device)
            fake = (G(z).cpu() * 0.5 + 0.5).clamp(0, 1)
            for i in range(cur):
                utils.save_image(fake[i], os.path.join(outdir, f"{generated + i:06d}.png"))
            generated += cur


def prepare_fid_ref_dir(src_dir, dst_dir, img_size):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    ensure_dir(dst_dir)
    if len(list(dst_dir.glob("*.png"))) > 0:
        return str(dst_dir)

    files = list_images(src_dir)
    for p in files:
        img = Image.open(p).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img.save(dst_dir / f"{p.stem}.png", format="PNG")
    return str(dst_dir)


class FlatImageFolder(Dataset):
    def __init__(self, folder, img_size=128):
        self.files = list_images(folder)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.tf(img)
        return img


def detector_consistency(img_dir, detector_ckpt, expected_label, detector_img_size, batch_size, num_workers, device):
    ds = FlatImageFolder(img_dir, detector_img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    try:
        model = models.resnet18(weights=None)
    except Exception:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(detector_ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())

    preds = np.array(preds)
    target = 0 if expected_label == "ai" else 1
    return float((preds == target).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=64, choices=[32, 64])
    parser.add_argument("--data_frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_n", type=int, default=4000)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--detector_ckpt", type=str, default="")
    parser.add_argument("--detector_img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(os.path.expanduser(args.save_root)) / args.run_name
    ensure_dir(run_dir)

    # train two unconditional models
    G_ai, train_ai_size = train_one_domain(
        train_dir=os.path.join(os.path.expanduser(args.data_root), "train", "ai"),
        save_dir=run_dir / "dcgan_ai",
        img_size=args.img_size,
        data_frac=args.data_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        nz=args.nz,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    G_nat, train_nat_size = train_one_domain(
        train_dir=os.path.join(os.path.expanduser(args.data_root), "train", "nature"),
        save_dir=run_dir / "dcgan_nature",
        img_size=args.img_size,
        data_frac=args.data_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        nz=args.nz,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    # sample
    save_samples(G_ai, run_dir / "eval_samples" / "ai", args.sample_n, args.sample_batch_size, args.nz, device)
    save_samples(G_nat, run_dir / "eval_samples" / "nature", args.sample_n, args.sample_batch_size, args.nz, device)

    # refs
    ref_ai = prepare_fid_ref_dir(
        os.path.join(os.path.expanduser(args.data_root), "val", "ai"),
        os.path.join(os.path.expanduser(args.save_root), "_fid_refs", f"ai_{args.img_size}"),
        args.img_size,
    )
    ref_nat = prepare_fid_ref_dir(
        os.path.join(os.path.expanduser(args.data_root), "val", "nature"),
        os.path.join(os.path.expanduser(args.save_root), "_fid_refs", f"nature_{args.img_size}"),
        args.img_size,
    )

    fid_ai = calculate_fid_given_paths(
        [ref_ai, str(run_dir / "eval_samples" / "ai")],
        batch_size=128, device=device, dims=2048, num_workers=max(1, args.num_workers)
    )
    fid_nat = calculate_fid_given_paths(
        [ref_nat, str(run_dir / "eval_samples" / "nature")],
        batch_size=128, device=device, dims=2048, num_workers=max(1, args.num_workers)
    )

    consistency_ai = ""
    consistency_nat = ""
    if args.detector_ckpt:
        consistency_ai = detector_consistency(
            run_dir / "eval_samples" / "ai", os.path.expanduser(args.detector_ckpt),
            "ai", args.detector_img_size, 128, args.num_workers, device
        )
        consistency_nat = detector_consistency(
            run_dir / "eval_samples" / "nature", os.path.expanduser(args.detector_ckpt),
            "nature", args.detector_img_size, 128, args.num_workers, device
        )

    row = {
        "run_name": args.run_name,
        "model": "dcgan",
        "img_size": args.img_size,
        "data_frac": args.data_frac,
        "epochs": args.epochs,
        "train_size": train_ai_size + train_nat_size,
        "fid_ai": float(fid_ai),
        "fid_nature": float(fid_nat),
        "consistency_ai": consistency_ai,
        "consistency_nature": consistency_nat,
        "seed": args.seed
    }
    append_csv(row, os.path.expanduser(args.results_csv))
    print(row)


if __name__ == "__main__":
    main()