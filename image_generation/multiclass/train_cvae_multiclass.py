import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class CVAE(nn.Module):
    def __init__(self, n_classes=10, latent_dim=128, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # 64 -> 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),   # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        enc_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(enc_dim + emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_dim + emb_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + emb_dim, enc_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32 -> 64
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


def loss_fn(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./runs_cvae_ai10")
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--fixed_per_class", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = datasets.ImageFolder(
        root=os.path.join(os.path.expanduser(args.data_root), "train"),
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    print("class_to_idx:", train_set.class_to_idx)

    with open(os.path.join(args.save_dir, "class_to_idx.json"), "w") as f:
        json.dump(train_set.class_to_idx, f, indent=2)

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    n_classes = len(train_set.class_to_idx)
    model = CVAE(n_classes=n_classes, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fixed_labels = []
    for c in range(n_classes):
        fixed_labels += [c] * args.fixed_per_class
    fixed_labels = torch.tensor(fixed_labels, dtype=torch.long, device=device)
    fixed_z = torch.randn(len(fixed_labels), args.latent_dim, device=device)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x, y)
            loss, recon_loss, kl = loss_fn(recon, x, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()

        total_loss /= len(loader)
        total_recon /= len(loader)
        total_kl /= len(loader)

        print("Epoch [{}/{}] loss={:.4f} recon={:.4f} kl={:.4f}".format(
            epoch, args.epochs, total_loss, total_recon, total_kl
        ))

        if epoch % args.sample_every == 0:
            with torch.no_grad():
                fake = model.decode(fixed_z, fixed_labels).detach().cpu()
            utils.save_image(
                fake,
                os.path.join(args.save_dir, "samples_epoch_{:03d}.png".format(epoch)),
                normalize=True,
                nrow=args.fixed_per_class
            )

        torch.save(model.state_dict(), os.path.join(args.save_dir, "cvae_last.pth"))

    print("done.")


if __name__ == "__main__":
    main()