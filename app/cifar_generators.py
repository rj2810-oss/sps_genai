"""
Inference wrappers for the CIFAR-10 diffusion model and energy-based model, used by the API.
"""
import io
from pathlib import Path

import torch
from torchvision.utils import save_image

from app.diffusion_model import UNet, Diffusion
from app.ebm_model import EnergyModel, langevin_sample

APP_DIR = Path(__file__).parent


def _to_png(images: torch.Tensor, nrow: int) -> bytes:
    buf = io.BytesIO()
    save_image(images, buf, format="png", nrow=nrow, normalize=True, value_range=(-1, 1))
    return buf.getvalue()


class DiffusionImageGenerator:
    def __init__(self, weights_path: Path = APP_DIR / "diffusion.pth"):
        self.model = UNet()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        self.diffusion = Diffusion(device="cpu")

    def generate_png(self, num_images: int = 8, seed: int | None = None) -> bytes:
        if seed is not None:
            torch.manual_seed(seed)
        images = self.diffusion.sample(self.model, num_images, device="cpu")
        return _to_png(images, nrow=min(num_images, 8))


class EnergyImageGenerator:
    def __init__(self, weights_path: Path = APP_DIR / "ebm.pth"):
        self.model = EnergyModel()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def generate_png(self, num_images: int = 8, steps: int = 256, seed: int | None = None) -> bytes:
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.rand(num_images, 3, 32, 32) * 2 - 1  # start from uniform noise
        # No torch.no_grad() here: Langevin sampling needs gradients w.r.t. the image
        images = langevin_sample(self.model, x, steps=steps)
        return _to_png(images, nrow=min(num_images, 8))
