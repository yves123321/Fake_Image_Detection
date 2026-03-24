import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
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


class Discriminator(nn.Module):
    def __init__(self, n_classes=2, ndf=64, nc=3):
        super().__init__()
        self.features = nn.Sequential(
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
        )
        self.adv_head = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.cls_head = nn.Conv2d(ndf * 8, n_classes, 4, 1, 0, bias=False)

    def forward(self, x):
        feat = self.features(x)
        adv = self.adv_head(feat).view(-1)
        cls = self.cls_head(feat).view(x.size(0), -1)
        return adv, cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./runs_acgan")
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

    G = Generator(nz=args.nz, n_classes=n_classes).to(device)
    D = Discriminator(n_classes=n_classes).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    adv_criterion = nn.BCELoss()
    cls_criterion = nn.CrossEntropyLoss()

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(16, args.nz, device=device)
    fixed_labels = torch.tensor([0] * 8 + [1] * 8, device=device)

    for epoch in range(1, args.epochs + 1):
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        for real, labels in loader:
            real = real.to(device)
            labels = labels.to(device)
            bsz = real.size(0)

            real_targets = torch.ones(bsz, device=device)
            fake_targets = torch.zeros(bsz, device=device)

            # ---- Train D ----
            optD.zero_grad()

            adv_real, cls_real = D(real)
            d_real_loss = adv_criterion(adv_real, real_targets) + cls_criterion(cls_real, labels)

            z = torch.randn(bsz, args.nz, device=device)
            fake_labels = torch.randint(0, n_classes, (bsz,), device=device)
            fake = G(z, fake_labels)

            adv_fake, cls_fake = D(fake.detach())
            d_fake_loss = adv_criterion(adv_fake, fake_targets) + cls_criterion(cls_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()

            # ---- Train G ----
            optG.zero_grad()
            adv_fake2, cls_fake2 = D(fake)
            g_loss = adv_criterion(adv_fake2, real_targets) + cls_criterion(cls_fake2, fake_labels)
            g_loss.backward()
            optG.step()

            d_loss_avg += d_loss.item()
            g_loss_avg += g_loss.item()

        d_loss_avg /= len(loader)
        g_loss_avg /= len(loader)

        print(f"Epoch [{epoch}/{args.epochs}] D_loss={d_loss_avg:.4f} G_loss={g_loss_avg:.4f}")

        if epoch % args.sample_every == 0:
            with torch.no_grad():
                fake = G(fixed_noise, fixed_labels).detach().cpu()
            utils.save_image(
                fake,
                os.path.join(args.save_dir, f"samples_epoch_{epoch:03d}.png"),
                normalize=True,
                nrow=4
            )

        torch.save(G.state_dict(), os.path.join(args.save_dir, "G_last.pth"))
        torch.save(D.state_dict(), os.path.join(args.save_dir, "D_last.pth"))

    print("Done.")


if __name__ == "__main__":
    main()