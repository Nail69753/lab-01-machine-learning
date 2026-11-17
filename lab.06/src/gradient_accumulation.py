
# Lab 6 - Gradient Accumulation Comparison

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()

    return 100 * correct / len(loader.dataset)


def train_model(model, train_loader, test_loader, device,
                learning_rate, accumulation_steps, epochs=5):

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        pending_batches = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(images)

            # Scale gradients to emulate a larger batch.
            # The last incomplete group is handled separately.
            pending_batches += 1
            is_last_batch = batch_index == len(train_loader) - 1
            group_size = (
                pending_batches if is_last_batch
                else accumulation_steps
            )

            (loss / group_size).backward()

            if pending_batches == accumulation_steps or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
                pending_batches = 0

        accuracy = evaluate(model, test_loader, device)
        avg_loss = total_loss / len(train_loader.dataset)

        history.append([epoch + 1, avg_loss, accuracy])
        print(
            f"  Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%"
        )

    return history


def main():
    lab_dir = Path(__file__).resolve().parent.parent

    with open(lab_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    learning_rate = config["training"]["learning_rate"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_data = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(
        test_data, batch_size=256, shuffle=False, num_workers=0
    )

    results = []

    experiments = [
        ("Batch 256", 256, 1),
        ("Batch 64 + Accumulation 4", 64, 4),
    ]

    for name, batch_size, accumulation_steps in experiments:
        print(f"\nRunning: {name}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = CNN().to(device)

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed)
        )

        history = train_model(
            model, train_loader, test_loader, device,
            learning_rate, accumulation_steps, epochs=5
        )

        for epoch, loss, accuracy in history:
            results.append([
                name, batch_size, accumulation_steps,
                epoch, round(loss, 4), round(accuracy, 2)
            ])

    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "gradient_accumulation.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "batch_size", "accumulation_steps",
            "epoch", "train_loss", "test_accuracy_percent"
        ])
        writer.writerows(results)

    print("\nGradient accumulation study completed successfully!")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
