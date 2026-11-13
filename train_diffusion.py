"""
Assignment 4: train the diffusion model on CIFAR-10.
Run from the repo root:  uv run python train_diffusion.py --epochs 15
Weights are saved after EVERY epoch, so you can stop early with Ctrl+C.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.diffusion_model import UNet, Diffusion

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
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

model = UNet().to(device)
diffusion = Diffusion(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
os.makedirs("samples", exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    running = 0.0
    for i, (x0, _) in enumerate(loader):
        x0 = x0.to(device)
        loss = diffusion.loss(model, x0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} Batch {i}/{len(loader)} Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1} average loss: {running/len(loader):.4f}")

    # Save weights every epoch (on CPU so they load in Docker)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "app/diffusion.pth")
    model.eval()
    samples = diffusion.sample(model, 16, device=device)
    save_image(samples, f"samples/diffusion_epoch_{epoch+1:02d}.png", nrow=8,
               normalize=True, value_range=(-1, 1))
    print(f"Saved app/diffusion.pth and samples/diffusion_epoch_{epoch+1:02d}.png")
