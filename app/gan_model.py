"""
Assignment 3, Part 1: GAN architecture for MNIST (28x28 grayscale).
"""
import torch
import torch.nn as nn

LATENT_DIM = 100


class Generator(nn.Module):
    """Noise (B, 100) -> image (B, 1, 28, 28) in [-1, 1]."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Fully connected layer to 7 x 7 x 128
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)

        self.deconv = nn.Sequential(
            # 128 x 7 x 7 -> 64 x 14 x 14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 14 x 14 -> 1 x 28 x 28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # reshape
        return self.deconv(x)


class Discriminator(nn.Module):
    """Image (B, 1, 28, 28) -> probability (B, 1) that the image is real."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 1 x 28 x 28 -> 64 x 14 x 14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14 -> 128 x 7 x 7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # real/fake probability, pair with nn.BCELoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(x))


def weights_init(m: nn.Module) -> None:
    """DCGAN-style init: N(0, 0.02) for conv layers, N(1, 0.02) for BatchNorm."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Shape sanity check: run `python gan_model.py`
    G, D = Generator(), Discriminator()
    G.apply(weights_init)
    D.apply(weights_init)

    z = torch.randn(16, LATENT_DIM)
    fake = G(z)
    out = D(fake)

    print("Generator output:", tuple(fake.shape))   # expect (16, 1, 28, 28)
    print("Discriminator output:", tuple(out.shape))  # expect (16, 1)
    assert fake.shape == (16, 1, 28, 28)
    assert out.shape == (16, 1)
    assert fake.min() >= -1 and fake.max() <= 1
    print("All shape checks passed.")
