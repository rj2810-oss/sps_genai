import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.cnn_model import CIFAR10_CLASSES, Assignment2CNN

WEIGHTS_PATH = Path(__file__).parent / "models" / "cnn_cifar10.pt"

# Must match the test-time transform used in train_cnn.py
PREPROCESS = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


class ImageClassifier:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.model = Assignment2CNN(num_classes=len(CIFAR10_CLASSES))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()  # inference mode for the whole lifetime of the API

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = PREPROCESS(image).unsqueeze(0)  # (1, 3, 64, 64)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        top_p, top_i = probs.topk(top_k)
        return {
            "predicted_class": CIFAR10_CLASSES[top_i[0].item()],
            "confidence": round(top_p[0].item(), 4),
            "top_k": [
                {"class": CIFAR10_CLASSES[i.item()], "probability": round(p.item(), 4)}
                for p, i in zip(top_p, top_i)
            ],
        }
