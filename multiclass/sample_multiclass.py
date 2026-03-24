import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import utils


class ACGAN_Generator(nn.Module):
    def __init__(self, nz=100, n_classes=10, emb_dim=50, ngf=64, nc=3):
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


class CVAE(nn.Module):
    def __init__(self, n_classes=10, latent_dim=128, emb_dim=16):
        super().__init__()
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

        enc_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(enc_dim + emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_dim + emb_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + emb_dim, enc_dim)
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
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def save_batch(batch, outdir, start_idx):
    outdir.mkdir(parents=True, exist_ok=True)
    batch = (batch * 0.5 + 0.5).clamp(0, 1)
    for i in range(batch.size(0)):
        utils.save_image(batch[i], outdir / "{:06d}.png".format(start_idx + i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["acgan", "cvae"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True,
                        help="multiclass dataset root, e.g. ~/data/vqdm_ai10")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--n_per_class", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    class_names = sorted([p.name for p in Path(os.path.expanduser(args.data_root), "train").iterdir() if p.is_dir()])
    n_classes = len(class_names)
    print("classes:", class_names)

    if args.model == "acgan":
        G = ACGAN_Generator(nz=args.nz, n_classes=n_classes).to(device)
        G.load_state_dict(torch.load(os.path.expanduser(args.ckpt), map_location=device))
        G.eval()
    else:
        G = CVAE(n_classes=n_classes, latent_dim=args.latent_dim).to(device)
        G.load_state_dict(torch.load(os.path.expanduser(args.ckpt), map_location=device))
        G.eval()

    out_root = Path(os.path.expanduser(args.out_root))

    with torch.no_grad():
        for cls_idx, cls_name in enumerate(class_names):
            generated = 0
            while generated < args.n_per_class:
                cur_bsz = min(args.batch_size, args.n_per_class - generated)
                y = torch.full((cur_bsz,), cls_idx, device=device, dtype=torch.long)

                if args.model == "acgan":
                    z = torch.randn(cur_bsz, args.nz, device=device)
                    fake = G(z, y)
                else:
                    z = torch.randn(cur_bsz, args.latent_dim, device=device)
                    fake = G.decode(z, y)

                save_batch(fake.cpu(), out_root / cls_name, generated)
                generated += cur_bsz

            print("generated class {}: {}".format(cls_name, generated))

    print("done.")


if __name__ == "__main__":
    main()