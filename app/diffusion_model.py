"""
Assignment 4: Diffusion model (DDPM) for CIFAR-10 (3 x 32 x 32).

Contains:
- sinusoidal_embedding: timestep -> vector (see theory Q1/Q2)
- UNet: predicts the noise that was added to x_t, conditioned on t
- Diffusion: forward noising q(x_t | x_0) and reverse sampling loop
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DIFFUSION_STEPS = 500  # T, shared by training and inference


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    emb[i]        = sin(t / max_period^(i / half))   for i in [0, half)
    emb[half + i] = cos(t / max_period^(i / half))
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + SiLU; the time embedding is added in between."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]  # inject timestep
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """
    32x32 -> 16x16 -> 8x8 (bottleneck) -> 16x16 -> 32x32, with skip connections.
    Input: noisy image x_t (B, 3, 32, 32) and timestep t (B,)
    Output: predicted noise epsilon (B, 3, 32, 32)
    """

    def __init__(self, base: int = 64, emb_dim: int = 128, t_dim: int = 256):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))

        self.inc = nn.Conv2d(3, base, 3, padding=1)
        # Encoder
        self.enc1 = ResBlock(base, base, t_dim)                              # 32x32
        self.down1 = nn.Conv2d(base, base, 4, stride=2, padding=1)           # -> 16x16
        self.enc2 = ResBlock(base, base * 2, t_dim)                          # 16x16
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)   # -> 8x8
        # Bottleneck
        self.mid1 = ResBlock(base * 2, base * 4, t_dim)                      # 8x8
        self.mid2 = ResBlock(base * 4, base * 4, t_dim)
        # Decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)  # -> 16x16
        self.dec2 = ResBlock(base * 4, base * 2, t_dim)                      # cat with enc2
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)      # -> 32x32
        self.dec1 = ResBlock(base * 2, base, t_dim)                          # cat with enc1

        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.emb_dim))

        h = self.inc(x)
        h1 = self.enc1(h, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h = self.mid2(self.mid1(self.down2(h2), t_emb), t_emb)
        h = self.dec2(torch.cat([self.up2(h), h2], dim=1), t_emb)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), t_emb)
        return self.out(h)


class Diffusion:
    """Linear beta schedule DDPM (Ho et al., 2020)."""

    def __init__(self, steps: int = DIFFUSION_STEPS, device="cpu"):
        self.T = steps
        self.betas = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise."""
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0):
        """Pick random t, add noise, ask the UNet to predict that noise, MSE loss."""
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        pred_noise = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, model, n: int, device="cpu"):
        """Reverse process: start from pure noise and denoise for T steps."""
        x = torch.randn(n, 3, 32, 32, device=device)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, tt)
            alpha, ab, beta = self.alphas[t], self.alpha_bars[t], self.betas[t]
            x = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if t > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        return x.clamp(-1, 1)


if __name__ == "__main__":
    # Sanity check: run `uv run python app/diffusion_model.py`
    model = UNet()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, DIFFUSION_STEPS, (4,))
    out = model(x, t)
    print("UNet output:", tuple(out.shape))  # expect (4, 3, 32, 32)
    assert out.shape == x.shape
    emb = sinusoidal_embedding(torch.tensor([1]), 8)
    print("Embedding for t=1, dim=8:", [round(v, 4) for v in emb[0].tolist()])
    print("Loss:", round(Diffusion().loss(model, x).item(), 4))
    print("Params:", sum(p.numel() for p in model.parameters()))
    print("Diffusion model checks passed.")
