"""
Assignment 3, Part 2: train the GAN on MNIST and save the generator weights.
Run from the repo root:  uv run python train_gan.py
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import Generator, Discriminator, weights_init, LATENT_DIM

torch.manual_seed(42)

EPOCHS = 20
BATCH_SIZE = 128
LR = 2e-4
BETAS = (0.5, 0.999)
WEIGHTS_PATH = "app/generator.pth"   # inside app/ so Docker copies it
SAMPLES_DIR = "samples"

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# MNIST scaled to [-1, 1] to match the generator's Tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

G = Generator().to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

fixed_noise = torch.randn(64, LATENT_DIM, device=device)
os.makedirs(SAMPLES_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    for i, (real, _) in enumerate(loader):
        real = real.to(device)
        b = real.size(0)
        real_labels = torch.ones(b, 1, device=device)
        fake_labels = torch.zeros(b, 1, device=device)

        # Train discriminator: maximize log D(x) + log(1 - D(G(z)))
        z = torch.randn(b, LATENT_DIM, device=device)
        fake = G(z)
        loss_D = criterion(D(real), real_labels) + criterion(D(fake.detach()), fake_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train generator: maximize log D(G(z)) (non-saturating loss)
        loss_G = criterion(D(fake), real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if i % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} Batch {i}/{len(loader)} "
                  f"Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")

    # Save a sample grid each epoch so you can see progress
    G.eval()
    with torch.no_grad():
        samples = G(fixed_noise)
    save_image(samples, f"{SAMPLES_DIR}/epoch_{epoch+1:02d}.png", nrow=8,
               normalize=True, value_range=(-1, 1))
    G.train()

# Save on CPU so the weights load anywhere (including the Docker container)
torch.save(G.cpu().state_dict(), WEIGHTS_PATH)
print(f"Saved generator weights to {WEIGHTS_PATH}")
