
# Lab 6 - Optimizer and Scheduler Comparison

import csv
import random
import time
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


def main():
    lab_dir = Path(__file__).resolve().parent.parent

    with open(lab_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    lr = config["training"]["learning_rate"]
    batch_size = config["training"]["batch_size"]

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

    train_loader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    experiments = [
        ("Adam", "StepLR"),
        ("Adam", "CosineAnnealingLR"),
        ("Adam", "ReduceLROnPlateau"),
        ("SGD+Momentum", "StepLR"),
        ("SGD+Momentum", "CosineAnnealingLR"),
        ("SGD+Momentum", "ReduceLROnPlateau"),
    ]

    results = []
    history = []

    for optimizer_name, scheduler_name in experiments:
        print(f"\nRunning: {optimizer_name} + {scheduler_name}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = CNN().to(device)

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9
            )

        if scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=0.5
            )
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

        criterion = nn.CrossEntropyLoss()
        start = time.perf_counter()

        for epoch in range(10):
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

            train_loss = total_loss / len(train_data)

            model.eval()
            correct = 0
            test_loss = 0.0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)

                    test_loss += (
                        criterion(outputs, labels).item() * len(images)
                    )
                    correct += (
                        outputs.argmax(1) == labels
                    ).sum().item()

            test_loss /= len(test_data)
            accuracy = 100 * correct / len(test_data)

            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(test_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            history.append([
                optimizer_name, scheduler_name,
                epoch + 1, round(train_loss, 4),
                round(test_loss, 4), round(accuracy, 2),
                current_lr
            ])

            print(
                f"  Epoch {epoch + 1}/10 | "
                f"Accuracy: {accuracy:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        elapsed = time.perf_counter() - start
        results.append([
            optimizer_name, scheduler_name,
            round(accuracy, 2), round(elapsed, 2)
        ])

    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(
        results_dir / "optimizer_results.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "optimizer", "scheduler",
            "final_accuracy_percent", "time_seconds"
        ])
        writer.writerows(results)

    with open(
        results_dir / "optimizer_history.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "optimizer", "scheduler", "epoch",
            "train_loss", "test_loss",
            "test_accuracy_percent", "learning_rate"
        ])
        writer.writerows(history)

    print("\nOptimizer study completed successfully!")
    print("Results saved to:", results_dir)


if __name__ == "__main__":
    main()
