
# Lab 6 - Batch Size Study

import csv
import random
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Same CNN architecture as Lab 3
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)


def main():
    lab_dir = Path(__file__).resolve().parent.parent

    with open(lab_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    results = []

    for batch_size in [32, 64, 128, 256, 512]:
        print(f"\nTraining with batch size: {batch_size}")

        torch.manual_seed(seed)
        model = CNN().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"]
        )

        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            tracemalloc.start()

        start = time.perf_counter()

        for epoch in range(5):
            model.train()

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            print(f"  Epoch {epoch + 1}/5 completed")

        elapsed = time.perf_counter() - start

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
            )
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_memory_mb = peak_bytes / 1024**2
            tracemalloc.stop()

        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                predictions = model(images.to(device)).argmax(1)
                correct += (
                    predictions == labels.to(device)
                ).sum().item()

        accuracy = 100 * correct / len(test_dataset)
        throughput = (len(dataset) * 5) / elapsed

        results.append([
            batch_size,
            round(throughput, 2),
            round(peak_memory_mb, 2),
            round(accuracy, 2)
        ])

        print(f"  Throughput: {throughput:.2f} samples/s")
        print(f"  Peak memory: {peak_memory_mb:.2f} MB")
        print(f"  Test accuracy: {accuracy:.2f}%")

    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = results_dir / "batch_study.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_size",
            "throughput_samples_per_second",
            "peak_memory_mb",
            "test_accuracy_percent"
        ])
        writer.writerows(results)

    print("\nBatch size study completed successfully!")
    print("Results saved to:", output)
    print("Note: CPU memory uses Python tracemalloc, not total PyTorch memory.")


if __name__ == "__main__":
    main()
