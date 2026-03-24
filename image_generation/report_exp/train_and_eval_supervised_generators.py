import os
import csv
import math
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, utils, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except Exception as e:
    raise ImportError(
        "Please install pytorch-fid first: python -m pip install pytorch-fid"
    ) from e


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


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
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in exts])


def prepare_fid_ref_dir(src_dir, dst_dir, img_size):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    ensure_dir(dst_dir)

    existing = list(dst_dir.glob("*.png"))
    if len(existing) > 0:
        return str(dst_dir)

    files = list_images(src_dir)
    print(f"[prepare_fid_ref_dir] {src_dir} -> {dst_dir}, num_files={len(files)}")

    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        out = dst_dir / f"{p.stem}.png"
        img.save(out, format="PNG")
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(files)}")

    return str(dst_dir)


def stratified_subset_indices(samples, frac, seed=42):
    if frac >= 0.999:
        return list(range(len(samples)))

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for idx, (_, y) in enumerate(samples):
        by_class[y].append(idx)

    chosen = []
    for y, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * frac)))
        chosen.extend(idxs[:k])

    rng.shuffle(chosen)
    return chosen


# =========================================================
# Dataset
# =========================================================
def get_train_dataset(data_root, img_size, data_frac, seed):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds = datasets.ImageFolder(
        root=os.path.join(os.path.expanduser(data_root), "train"),
        transform=tf
    )
    indices = stratified_subset_indices(ds.samples, data_frac, seed)
    ds = Subset(ds, indices)
    return ds


# =========================================================
# Architectures (img_size = 32 or 64)
# =========================================================
def get_channels(img_size):
    if img_size == 32:
        return dict(
            g_start=256,
            g_mid=[128, 64],
            d_mid=[64, 128, 256],
            feat_ch=256
        )
    elif img_size == 64:
        return dict(
            g_start=512,
            g_mid=[256, 128, 64],
            d_mid=[64, 128, 256, 512],
            feat_ch=512
        )
    else:
        raise ValueError("Only img_size=32 or 64 is supported in this sweep.")


# -------------------------
# ACGAN
# -------------------------
class ACGANGenerator(nn.Module):
    def __init__(self, img_size=64, nz=100, n_classes=2, emb_dim=32):
        super().__init__()
        cfg = get_channels(img_size)
        self.img_size = img_size
        self.nz = nz
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.fc = nn.Linear(nz + emb_dim, cfg["g_start"] * 4 * 4)

        layers = [
            nn.BatchNorm2d(cfg["g_start"]),
            nn.ReLU(True),
        ]

        in_ch = cfg["g_start"]
        for out_ch in cfg["g_mid"]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
            in_ch = out_ch

        layers += [
            nn.ConvTranspose2d(in_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z, y):
        y_emb = self.emb(y)
        x = torch.cat([z, y_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.net(x)


class ACGANDiscriminator(nn.Module):
    def __init__(self, img_size=64, n_classes=2):
        super().__init__()
        cfg = get_channels(img_size)

        layers = []
        in_ch = 3
        mids = cfg["d_mid"]
        for i, out_ch in enumerate(mids):
            if i == 0:
                layers += [
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            else:
                layers += [
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.adv_head = nn.Sequential(
            nn.Conv2d(cfg["feat_ch"], 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.cls_head = nn.Conv2d(cfg["feat_ch"], n_classes, 4, 1, 0, bias=False)

    def forward(self, x):
        feat = self.features(x)
        adv = self.adv_head(feat).view(-1)
        cls = self.cls_head(feat).view(x.size(0), -1)
        return adv, cls


# -------------------------
# CVAE
# -------------------------
class CVAE(nn.Module):
    def __init__(self, img_size=64, n_classes=2, latent_dim=64, emb_dim=16):
        super().__init__()
        cfg = get_channels(img_size)
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.feat_ch = cfg["feat_ch"]

        # encoder
        enc_layers = []
        in_ch = 3
        for i, out_ch in enumerate(cfg["d_mid"]):
            if i == 0:
                enc_layers += [
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                    nn.ReLU(True),
                ]
            else:
                enc_layers += [
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(True),
                ]
            in_ch = out_ch
        self.enc = nn.Sequential(*enc_layers)

        enc_dim = cfg["feat_ch"] * 4 * 4
        self.fc_mu = nn.Linear(enc_dim + emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_dim + emb_dim, latent_dim)

        # decoder
        self.fc_dec = nn.Linear(latent_dim + emb_dim, cfg["g_start"] * 4 * 4)

        dec_layers = [
            nn.BatchNorm2d(cfg["g_start"]),
            nn.ReLU(True),
        ]
        in_ch = cfg["g_start"]
        for out_ch in cfg["g_mid"]:
            dec_layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
            in_ch = out_ch

        dec_layers += [
            nn.ConvTranspose2d(in_ch, 3, 4, 2, 1),
            nn.Tanh()
        ]
        self.dec = nn.Sequential(*dec_layers)

    def encode(self, x, y):
        h = self.enc(x).view(x.size(0), -1)
        y_emb = self.emb(y)
        h = torch.cat([h, y_emb], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.emb(y)
        z = torch.cat([z, y_emb], dim=1)
        h = self.fc_dec(z).view(z.size(0), -1, 4, 4)
        return self.dec(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def cvae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


# =========================================================
# Training
# =========================================================
def train_acgan(args, loader, device, save_dir):
    G = ACGANGenerator(img_size=args.img_size, nz=args.nz, n_classes=2).to(device)
    D = ACGANDiscriminator(img_size=args.img_size, n_classes=2).to(device)

    adv_criterion = nn.BCELoss()
    cls_criterion = nn.CrossEntropyLoss()

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    history = []
    fixed_noise = torch.randn(32, args.nz, device=device)
    fixed_labels = torch.tensor([0] * 16 + [1] * 16, device=device)

    for epoch in range(1, args.epochs + 1):
        d_loss_avg = 0.0
        g_loss_avg = 0.0

        for real, labels in loader:
            real = real.to(device)
            labels = labels.to(device)
            bsz = real.size(0)

            real_targets = torch.ones(bsz, device=device)
            fake_targets = torch.zeros(bsz, device=device)

            # D
            optD.zero_grad()

            adv_real, cls_real = D(real)
            loss_real = adv_criterion(adv_real, real_targets) + cls_criterion(cls_real, labels)

            z = torch.randn(bsz, args.nz, device=device)
            fake_labels = torch.randint(0, 2, (bsz,), device=device)
            fake = G(z, fake_labels)

            adv_fake, cls_fake = D(fake.detach())
            loss_fake = adv_criterion(adv_fake, fake_targets) + cls_criterion(cls_fake, fake_labels)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            optD.step()

            # G
            optG.zero_grad()
            adv_fake2, cls_fake2 = D(fake)
            g_loss = adv_criterion(adv_fake2, real_targets) + cls_criterion(cls_fake2, fake_labels)
            g_loss.backward()
            optG.step()

            d_loss_avg += d_loss.item()
            g_loss_avg += g_loss.item()

        d_loss_avg /= len(loader)
        g_loss_avg /= len(loader)

        history.append({
            "epoch": epoch,
            "d_loss": d_loss_avg,
            "g_loss": g_loss_avg
        })
        print(f"[{args.run_name}] Epoch {epoch}/{args.epochs} D_loss={d_loss_avg:.4f} G_loss={g_loss_avg:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                fake = G(fixed_noise, fixed_labels).detach().cpu()
            utils.save_image(
                fake,
                os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png"),
                normalize=True,
                nrow=8
            )

    torch.save(G.state_dict(), os.path.join(save_dir, "G_last.pth"))
    torch.save(D.state_dict(), os.path.join(save_dir, "D_last.pth"))
    save_json(history, os.path.join(save_dir, "history.json"))
    return G


def train_cvae(args, loader, device, save_dir):
    model = CVAE(
        img_size=args.img_size,
        n_classes=2,
        latent_dim=args.latent_dim,
        emb_dim=16
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history = []

    fixed_z = torch.randn(32, args.latent_dim, device=device)
    fixed_labels = torch.tensor([0] * 16 + [1] * 16, device=device)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x, y)
            loss, recon_loss, kl = cvae_loss(recon, x, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()

        total_loss /= len(loader)
        total_recon /= len(loader)
        total_kl /= len(loader)

        history.append({
            "epoch": epoch,
            "loss": total_loss,
            "recon": total_recon,
            "kl": total_kl
        })
        print(f"[{args.run_name}] Epoch {epoch}/{args.epochs} loss={total_loss:.4f} recon={total_recon:.4f} kl={total_kl:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                fake = model.decode(fixed_z, fixed_labels).detach().cpu()
            utils.save_image(
                fake,
                os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png"),
                normalize=True,
                nrow=8
            )

    torch.save(model.state_dict(), os.path.join(save_dir, "cvae_last.pth"))
    save_json(history, os.path.join(save_dir, "history.json"))
    return model


# =========================================================
# Sampling + Eval
# =========================================================
def save_samples(batch, outdir, start_idx):
    ensure_dir(outdir)
    batch = (batch * 0.5 + 0.5).clamp(0, 1)
    for i in range(batch.size(0)):
        utils.save_image(batch[i], os.path.join(outdir, f"{start_idx + i:06d}.png"))


def sample_conditional(model, model_name, out_root, img_size, sample_n, batch_size, device,
                       nz=100, latent_dim=64):
    out_root = Path(out_root)
    ensure_dir(out_root / "ai")
    ensure_dir(out_root / "nature")

    label_map = {"ai": 0, "nature": 1}

    with torch.no_grad():
        for label_name, label_id in label_map.items():
            generated = 0
            while generated < sample_n:
                cur_bsz = min(batch_size, sample_n - generated)
                y = torch.full((cur_bsz,), label_id, device=device, dtype=torch.long)

                if model_name == "acgan":
                    z = torch.randn(cur_bsz, nz, device=device)
                    fake = model(z, y)
                elif model_name == "cvae":
                    z = torch.randn(cur_bsz, latent_dim, device=device)
                    fake = model.decode(z, y)
                else:
                    raise ValueError(model_name)

                save_samples(fake.cpu(), out_root / label_name, generated)
                generated += cur_bsz

            print(f"[sample] {label_name}: {generated}")


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
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = self.tf(img)
        return img, str(p)


def detector_consistency(img_dir, detector_ckpt, detector_img_size, batch_size, num_workers, device):
    ds = FlatImageFolder(img_dir, img_size=detector_img_size)
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
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())

    preds = np.array(preds)
    return preds


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["acgan", "cvae"])

    parser.add_argument("--img_size", type=int, default=64, choices=[32, 64])
    parser.add_argument("--data_frac", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument("--sample_n", type=int, default=4000)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--sample_every", type=int, default=5)

    parser.add_argument("--detector_ckpt", type=str, default="")
    parser.add_argument("--detector_img_size", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    run_dir = Path(os.path.expanduser(args.save_root)) / args.run_name
    ensure_dir(run_dir)

    # save config
    save_json(vars(args), run_dir / "config.json")

    # data
    train_ds = get_train_dataset(args.data_root, args.img_size, args.data_frac, args.seed)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print("train size:", len(train_ds))

    # train
    start = time.time()
    if args.model == "acgan":
        model = train_acgan(args, loader, device, run_dir)
    else:
        model = train_cvae(args, loader, device, run_dir)
    train_time = time.time() - start

    # sample
    sample_root = run_dir / "eval_samples"
    sample_conditional(
        model=model,
        model_name=args.model,
        out_root=sample_root,
        img_size=args.img_size,
        sample_n=args.sample_n,
        batch_size=args.sample_batch_size,
        device=device,
        nz=args.nz,
        latent_dim=args.latent_dim
    )

    # prepare FID refs
    ref_ai = prepare_fid_ref_dir(
        src_dir=os.path.join(os.path.expanduser(args.data_root), "val", "ai"),
        dst_dir=os.path.join(os.path.expanduser(args.save_root), "_fid_refs", f"ai_{args.img_size}"),
        img_size=args.img_size
    )
    ref_nature = prepare_fid_ref_dir(
        src_dir=os.path.join(os.path.expanduser(args.data_root), "val", "nature"),
        dst_dir=os.path.join(os.path.expanduser(args.save_root), "_fid_refs", f"nature_{args.img_size}"),
        img_size=args.img_size
    )

    # FID
    fid_ai = calculate_fid_given_paths(
        [ref_ai, str(sample_root / "ai")],
        batch_size=128,
        device=device,
        dims=2048,
        num_workers=max(1, args.num_workers)
    )
    fid_nature = calculate_fid_given_paths(
        [ref_nature, str(sample_root / "nature")],
        batch_size=128,
        device=device,
        dims=2048,
        num_workers=max(1, args.num_workers)
    )

    # detector consistency
    consistency_ai = None
    consistency_nature = None
    if args.detector_ckpt:
        preds_ai = detector_consistency(
            img_dir=sample_root / "ai",
            detector_ckpt=os.path.expanduser(args.detector_ckpt),
            detector_img_size=args.detector_img_size,
            batch_size=128,
            num_workers=args.num_workers,
            device=device
        )
        preds_nat = detector_consistency(
            img_dir=sample_root / "nature",
            detector_ckpt=os.path.expanduser(args.detector_ckpt),
            detector_img_size=args.detector_img_size,
            batch_size=128,
            num_workers=args.num_workers,
            device=device
        )
        consistency_ai = float((preds_ai == 0).mean())
        consistency_nature = float((preds_nat == 1).mean())

    row = {
        "run_name": args.run_name,
        "model": args.model,
        "img_size": args.img_size,
        "data_frac": args.data_frac,
        "epochs": args.epochs,
        "train_size": len(train_ds),
        "fid_ai": float(fid_ai),
        "fid_nature": float(fid_nature),
        "consistency_ai": "" if consistency_ai is None else float(consistency_ai),
        "consistency_nature": "" if consistency_nature is None else float(consistency_nature),
        "train_time_sec": round(train_time, 2),
        "seed": args.seed
    }
    append_csv(row, os.path.expanduser(args.results_csv))
    print("[done]", row)


if __name__ == "__main__":
    main()