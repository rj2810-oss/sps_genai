"""
Assignment 4: Energy-Based Model for CIFAR-10 (3 x 32 x 32).

The network maps an image to a single scalar energy E(x).
Real images should get LOW energy, generated images HIGH energy.

Sampling uses Langevin dynamics: gradient descent on the INPUT IMAGE
(not on the weights) to move random noise toward low-energy, realistic images.
"""
import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(),              # 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),   # 16x16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),  # 8x8
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(), # 4x4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) energies


def langevin_sample(model, x, steps=60, step_size=10.0, noise_std=0.005, grad_clip=0.03):
    """
    Langevin dynamics: x <- x - step_size * dE/dx + noise.

    Gradients are taken with respect to the input x ONLY:
    - model parameters are frozen (requires_grad=False) so no weight gradients
      are computed or accumulated during sampling
    - x is a leaf tensor with requires_grad=True at each step
    - torch.autograd.grad returns dE/dx directly instead of storing it in .grad,
      so nothing accumulates across steps

    Must NOT be called inside torch.no_grad(), since it needs gradients w.r.t. x.
    """
    param_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x = x.detach().clone()
    for _ in range(steps):
        x = (x + noise_std * torch.randn_like(x)).clamp(-1, 1)
        x.requires_grad_(True)                      # track gradients for the IMAGE
        energy = model(x).sum()
        grad_x, = torch.autograd.grad(energy, x)    # dE/dx, shape (B, 3, 32, 32)
        x = (x - step_size * grad_x.clamp(-grad_clip, grad_clip)).detach().clamp(-1, 1)

    for p, flag in zip(model.parameters(), param_flags):
        p.requires_grad_(flag)                      # restore for training
    return x


class ReplayBuffer:
    """Keeps past Langevin samples so chains can continue across training steps."""

    def __init__(self, size=8192, device="cpu"):
        self.buffer = torch.rand(size, 3, 32, 32, device=device) * 2 - 1

    def sample(self, n, fresh_ratio=0.05):
        idx = torch.randint(0, len(self.buffer), (n,), device=self.buffer.device)
        x = self.buffer[idx].clone()
        n_fresh = int(n * fresh_ratio)
        x[:n_fresh] = torch.rand_like(x[:n_fresh]) * 2 - 1  # restart a few chains from noise
        return x, idx

    def update(self, x, idx):
        self.buffer[idx] = x.detach()


if __name__ == "__main__":
    # Sanity check: run `uv run python app/ebm_model.py`
    model = EnergyModel()
    x = torch.rand(4, 3, 32, 32) * 2 - 1
    print("Energy output:", tuple(model(x).shape))  # expect (4,)
    samples = langevin_sample(model, x, steps=5)
    print("Langevin output:", tuple(samples.shape))  # expect (4, 3, 32, 32)
    assert samples.shape == x.shape
    assert all(p.grad is None for p in model.parameters()), "weights should get no gradients"
    assert all(p.requires_grad for p in model.parameters()), "requires_grad should be restored"
    print("EBM checks passed: sampling changed the image, not the weights.")
