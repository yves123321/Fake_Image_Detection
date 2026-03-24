import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class CVAE(nn.Module):
    def __init__(self, img_size=64, n_classes=2, latent_dim=128, emb_dim=16):
        super().__init__()
        self.img_size = img_size
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.emb = nn.Embedding(n_classes, emb_dim)

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 64 -> 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        enc_out_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(enc_out_dim + emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim + emb_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + emb_dim, enc_out_dim)
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./runs_cvae")
    parser.add_argument("--sample_every", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = datasets.ImageFolder(
        root=os.path.join(args.data_root, "train"),
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    print("class_to_idx:", train_set.class_to_idx)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    n_classes = len(train_set.class_to_idx)
    model = CVAE(img_size=args.img_size, n_classes=n_classes, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fixed_z = torch.randn(16, args.latent_dim, device=device)
    fixed_y = torch.tensor([0] * 8 + [1] * 8, device=device)

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

        print(f"Epoch [{epoch}/{args.epochs}] loss={total_loss:.4f} recon={total_recon:.4f} kl={total_kl:.4f}")

        if epoch % args.sample_every == 0:
            with torch.no_grad():
                samples = model.decode(fixed_z, fixed_y).detach().cpu()
            utils.save_image(
                samples,
                os.path.join(args.save_dir, f"samples_epoch_{epoch:03d}.png"),
                normalize=True,
                nrow=4
            )

        torch.save(model.state_dict(), os.path.join(args.save_dir, "cvae_last.pth"))

    print("Done.")


if __name__ == "__main__":
    main()