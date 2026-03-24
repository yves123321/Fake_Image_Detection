import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import utils


# -------------------------
# Shared label mapping
# -------------------------
LABEL_TO_ID = {"ai": 0, "nature": 1}


# -------------------------
# DCGAN
# -------------------------
class DCGAN_Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


# -------------------------
# ACGAN
# -------------------------
class ACGAN_Generator(nn.Module):
    def __init__(self, nz=100, n_classes=2, emb_dim=50, ngf=64, nc=3):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, emb_dim)
        self.fc = nn.Linear(nz + emb_dim, ngf * 8 * 4 * 4)

        self.net = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, y):
        y_emb = self.label_emb(y)
        x = torch.cat([z, y_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.net(x)


# -------------------------
# CVAE
# -------------------------
class CVAE(nn.Module):
    def __init__(self, img_size=64, n_classes=2, latent_dim=128, emb_dim=16):
        super().__init__()
        self.img_size = img_size
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.emb = nn.Embedding(n_classes, emb_dim)

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        enc_out_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(enc_out_dim + emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim + emb_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + emb_dim, enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def decode(self, z, y):
        y_emb = self.emb(y)
        z = torch.cat([z, y_emb], dim=1)
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)


def save_batch(batch, outdir, start_idx):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    batch = (batch * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

    for i in range(batch.size(0)):
        utils.save_image(batch[i], outdir / f"{start_idx + i:06d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["dcgan", "acgan", "cvae"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=64)

    parser.add_argument("--nz", type=int, default=100)          # for dcgan/acgan
    parser.add_argument("--latent_dim", type=int, default=128)  # for cvae

    parser.add_argument("--label", type=str, default=None, choices=["ai", "nature"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.model == "dcgan":
        G = DCGAN_Generator(nz=args.nz).to(device)
        G.load_state_dict(torch.load(args.ckpt, map_location=device))
        G.eval()

    elif args.model == "acgan":
        if args.label is None:
            raise ValueError("ACGAN sampling requires --label ai or --label nature")
        G = ACGAN_Generator(nz=args.nz, n_classes=2).to(device)
        G.load_state_dict(torch.load(args.ckpt, map_location=device))
        G.eval()
        label_id = LABEL_TO_ID[args.label]

    elif args.model == "cvae":
        if args.label is None:
            raise ValueError("CVAE sampling requires --label ai or --label nature")
        G = CVAE(img_size=args.img_size, n_classes=2, latent_dim=args.latent_dim).to(device)
        G.load_state_dict(torch.load(args.ckpt, map_location=device))
        G.eval()
        label_id = LABEL_TO_ID[args.label]

    generated = 0
    with torch.no_grad():
        while generated < args.n:
            cur_bsz = min(args.batch_size, args.n - generated)

            if args.model == "dcgan":
                z = torch.randn(cur_bsz, args.nz, 1, 1, device=device)
                fake = G(z)

            elif args.model == "acgan":
                z = torch.randn(cur_bsz, args.nz, device=device)
                y = torch.full((cur_bsz,), label_id, device=device, dtype=torch.long)
                fake = G(z, y)

            elif args.model == "cvae":
                z = torch.randn(cur_bsz, args.latent_dim, device=device)
                y = torch.full((cur_bsz,), label_id, device=device, dtype=torch.long)
                fake = G.decode(z, y)

            save_batch(fake.cpu(), args.outdir, generated)
            generated += cur_bsz
            print(f"generated: {generated}/{args.n}")

    print("done.")


if __name__ == "__main__":
    main()