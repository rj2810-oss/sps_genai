"""Train the Assignment 2 CNN on CIFAR-10 and save weights for the API.

Run from the repo root:  uv run python train_cnn.py --epochs 10
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from app.cnn_model import Assignment2CNN

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
OUT_PATH = Path("app/models/cnn_cifar10.pt")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(batch_size: int):
    # CIFAR-10 images are 32x32, but the spec says 64x64 input, so upsample.
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = get_device()
    print(f"Training on {device}")
    train_loader, test_loader = get_loaders(args.batch_size)

    model = Assignment2CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()  # evaluate() switches to eval mode, so switch back every epoch
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs}  train_loss={running_loss / len(train_loader):.4f}  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            # save on CPU so the Docker container (no GPU) can load it
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT_PATH)

    print(f"Best test accuracy: {best_acc:.2f}%. Weights saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
