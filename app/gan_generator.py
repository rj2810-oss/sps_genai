"""
Inference wrapper for the trained MNIST GAN generator, used by the FastAPI app.
"""
import io
from pathlib import Path

import torch
from torchvision.utils import save_image

from app.gan_model import Generator, LATENT_DIM

WEIGHTS_PATH = Path(__file__).parent / "generator.pth"


class DigitGenerator:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.model = Generator()
        # map_location="cpu": weights trained on MPS must load on a CPU-only container
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def generate_png(self, num_images: int = 16, seed: int | None = None) -> bytes:
        """Generate num_images digits and return them as one PNG grid."""
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(num_images, LATENT_DIM)
            images = self.model(z)
        buf = io.BytesIO()
        save_image(images, buf, format="png", nrow=min(num_images, 8),
                   normalize=True, value_range=(-1, 1))
        return buf.getvalue()
