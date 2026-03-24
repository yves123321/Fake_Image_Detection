import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, utils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SingleFolderDataset(Dataset):
    def __init__(self, root, img_size=64):
        self.ds = datasets.ImageFolder(
            root=str(root),
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x


def build_dataset(data_root, domain, img_size):
    train_root = Path(data_root) / "train"

    if domain in ["ai", "nature"]:
        # 为了复用 ImageFolder，这里临时把 train 当根目录，过滤出某一类
        full = datasets.ImageFolder(
            root=str(train_root),
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        target_idx = full.class_to_idx[domain]
        indices = [i for i, (_, y) in enumerate(full.samples) if y == target_idx]
        subset = torch.utils.data.Subset(full, indices)

        class OnlyX(Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                x, _ = self.ds[idx]
                return x

        return OnlyX(subset)

    elif domain == "all":
        full = datasets.ImageFolder(
            root=str(train_root),
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )

        class OnlyX(Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                x, _ = self.ds[idx]
                return x

        return OnlyX(full)
    else:
        raise ValueError("domain must be one of: ai, nature, all")


class Generator(nn.Module):
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


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--domain", type=str, default="nature", choices=["ai", "nature", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./runs_dcgan")
    parser.add_argument("--sample_every", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = build_dataset(args.data_root, args.domain, args.img_size)
    print("dataset size:", len(dataset))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    netG = Generator(nz=args.nz).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    for epoch in range(1, args.epochs + 1):
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        for real in loader:
            real = real.to(device)
            bsz = real.size(0)

            real_label = torch.ones(bsz, device=device)
            fake_label = torch.zeros(bsz, device=device)

            # D
            optD.zero_grad()
            out_real = netD(real)
            loss_real = criterion(out_real, real_label)

            noise = torch.randn(bsz, args.nz, 1, 1, device=device)
            fake = netG(noise)
            out_fake = netD(fake.detach())
            loss_fake = criterion(out_fake, fake_label)

            lossD = loss_real + loss_fake
            lossD.backward()
            optD.step()

            # G
            optG.zero_grad()
            out_fake2 = netD(fake)
            lossG = criterion(out_fake2, real_label)
            lossG.backward()
            optG.step()

            d_loss_avg += lossD.item()
            g_loss_avg += lossG.item()

        d_loss_avg /= len(loader)
        g_loss_avg /= len(loader)

        print(f"Epoch [{epoch}/{args.epochs}] D_loss={d_loss_avg:.4f} G_loss={g_loss_avg:.4f}")

        if epoch % args.sample_every == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            utils.save_image(
                fake,
                os.path.join(args.save_dir, f"samples_epoch_{epoch:03d}.png"),
                normalize=True,
                nrow=8
            )

        torch.save(netG.state_dict(), os.path.join(args.save_dir, "netG_last.pth"))
        torch.save(netD.state_dict(), os.path.join(args.save_dir, "netD_last.pth"))

    print("Done.")


if __name__ == "__main__":
    main()