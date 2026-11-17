
# Lab 7 - Train Oversized Baseline CNN

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baseline import BaselineCNN


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lab_dir = Path(__file__).resolve().parent.parent
    repo_dir = lab_dir.parent

    results_dir = lab_dir / "results"
    checkpoints_dir = lab_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_data = datasets.FashionMNIST(
        root=str(repo_dir / "data"),
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=str(repo_dir / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_data, batch_size=128, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=256, shuffle=False, num_workers=0
    )

    model = BaselineCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)

        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / len(test_data)
        avg_loss = total_loss / len(train_data)

        history.append([epoch + 1, avg_loss, accuracy])

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Loss: {avg_loss:.4f} | "
            f"Test accuracy: {accuracy:.2f}%"
        )

    checkpoint_path = checkpoints_dir / "baseline_cnn.pt"
    torch.save(model.state_dict(), checkpoint_path)

    parameters = sum(p.numel() for p in model.parameters())
    model_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)

    # Measure single-image inference latency
    model.eval()
    sample = torch.randn(1, 1, 28, 28, device=device)

    with torch.inference_mode():
        for _ in range(20):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(100):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            timings.append((time.perf_counter() - start) * 1000)

    mean_latency_ms = sum(timings) / len(timings)

    with open(
        results_dir / "baseline_history.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_accuracy_percent"])
        writer.writerows(history)

    with open(
        results_dir / "baseline_metrics.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "parameters", "model_size_mb",
            "inference_latency_ms", "test_accuracy_percent",
            "device"
        ])
        writer.writerow([
            parameters, round(model_size_mb, 3),
            round(mean_latency_ms, 4), round(accuracy, 2),
            str(device)
        ])

    print("\nBaseline training completed successfully!")
    print(f"Parameters: {parameters:,}")
    print(f"Model size: {model_size_mb:.3f} MB")
    print(f"Latency: {mean_latency_ms:.4f} ms/image")
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
