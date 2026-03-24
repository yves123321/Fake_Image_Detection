import os
import argparse
from pathlib import Path
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in Path(folder).iterdir() if p.is_file() and p.suffix in exts])


class FolderDataset(Dataset):
    def __init__(self, folder, img_size=64):
        self.files = list_images(folder)
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
        return self.tf(img)


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
        self.head = nn.Sequential(nn.Conv2d(cfg["feat"], 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x)).view(-1)


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = FolderDataset(args.train_dir, img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    print("dataset size:", len(ds))

    G = Generator(img_size=args.img_size, nz=args.nz).to(device)
    D = Discriminator(img_size=args.img_size).to(device)

    criterion = nn.BCELoss()
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_z = torch.randn(32, args.nz, device=device)

    for epoch in range(1, args.epochs + 1):
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

            z = torch.randn(bsz, args.nz, device=device)
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
        print(f"Epoch [{epoch}/{args.epochs}] D_loss={d_avg:.4f} G_loss={g_avg:.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            with torch.no_grad():
                fake = G(fixed_z).cpu()
            utils.save_image(fake, os.path.join(args.save_dir, f"samples_epoch_{epoch:03d}.png"),
                             normalize=True, nrow=8)

    torch.save(G.state_dict(), os.path.join(args.save_dir, "G_last.pth"))
    print("done train")


def sample(args):
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    G = Generator(img_size=args.img_size, nz=args.nz).to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location=device))
    G.eval()

    generated = 0
    with torch.no_grad():
        while generated < args.n:
            cur = min(args.batch_size, args.n - generated)
            z = torch.randn(cur, args.nz, device=device)
            fake = (G(z).cpu() * 0.5 + 0.5).clamp(0, 1)
            for i in range(cur):
                utils.save_image(fake[i], os.path.join(args.outdir, f"{generated + i:06d}.png"))
            generated += cur
            print(generated, "/", args.n)

    print("done sample")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "sample"])
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=64, choices=[32, 64])
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()