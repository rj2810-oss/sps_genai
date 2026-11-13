"""
Assignment 4: train the energy-based model on CIFAR-10 with contrastive divergence
and Langevin sampling from a replay buffer.
Run from the repo root:  uv run python train_ebm.py --epochs 5
Weights are saved after EVERY epoch, so you can stop early with Ctrl+C.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.ebm_model import EnergyModel, ReplayBuffer, langevin_sample

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--langevin_steps", type=int, default=60)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--alpha", type=float, default=0.1, help="energy regularization weight")
args = parser.parse_args()

torch.manual_seed(42)
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
])
dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

model = EnergyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.0, 0.999))
buffer = ReplayBuffer(device=device)
os.makedirs("samples", exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    for i, (real, _) in enumerate(loader):
        real = (real.to(device) + 0.005 * torch.randn_like(real.to(device))).clamp(-1, 1)

        # 1) Generate negatives: gradient descent on the IMAGES (weights frozen inside)
        init, idx = buffer.sample(real.size(0))
        fake = langevin_sample(model, init, steps=args.langevin_steps)
        buffer.update(fake, idx)

        # 2) Update the WEIGHTS: push real energy down, fake energy up
        e_real = model(real)
        e_fake = model(fake)
        cd_loss = e_real.mean() - e_fake.mean()
        reg_loss = args.alpha * (e_real ** 2 + e_fake ** 2).mean()
        loss = cd_loss + reg_loss

        optimizer.zero_grad()   # clear weight gradients from the previous step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} Batch {i}/{len(loader)} "
                  f"Loss: {loss.item():.4f} E_real: {e_real.mean().item():.4f} "
                  f"E_fake: {e_fake.mean().item():.4f}")

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "app/ebm.pth")
    save_image(fake[:16], f"samples/ebm_epoch_{epoch+1:02d}.png", nrow=8,
               normalize=True, value_range=(-1, 1))
    print(f"Saved app/ebm.pth and samples/ebm_epoch_{epoch+1:02d}.png")
