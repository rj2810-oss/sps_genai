import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class Assignment2CNN(nn.Module):
    """CNN matching the Assignment 2 spec. Expects input of shape (N, 3, 64, 64)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 64x64x3 -> conv -> 64x64x16 -> pool -> 32x32x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 32x32x16 -> conv -> 32x32x32 -> pool -> 16x16x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # flatten: 32 * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits; CrossEntropyLoss applies softmax
