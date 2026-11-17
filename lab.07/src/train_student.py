
# Lab 7 - Student Training and Knowledge Distillation

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baseline import BaselineCNN
from student import StudentCNN


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


def measure_latency(model, device):
    model.eval()
    example = torch.randn(1, 1, 28, 28, device=device)

    with torch.no_grad():
        for _ in range(20):
            model(example)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(100):
            model(example)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return elapsed * 1000 / 100


def train_student(model, teacher, loader, device, distilled):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    temperature = 4.0
    alpha = 0.5

    if teacher is not None:
        teacher.eval()

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            student_logits = model(images)
            ce_loss = criterion(student_logits, labels)

            if distilled:
                with torch.no_grad():
                    teacher_logits = teacher(images)

                kd_loss = F.kl_div(
                    F.log_softmax(
                        student_logits / temperature, dim=1
                    ),
                    F.softmax(
                        teacher_logits / temperature, dim=1
                    ),
                    reduction="batchmean"
                )

                loss = (
                    alpha * ce_loss
                    + (1 - alpha) * temperature**2 * kd_loss
                )
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        average_loss = total_loss / len(loader.dataset)

        print(
            f"  Epoch {epoch + 1}/5 | "
            f"Loss: {average_loss:.4f}"
        )


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lab_dir = Path(__file__).resolve().parent.parent
    repo_dir = lab_dir.parent

    checkpoint_dir = lab_dir / "checkpoints"
    results_dir = lab_dir / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

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
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    teacher = BaselineCNN().to(device)
    teacher.load_state_dict(
        torch.load(
            checkpoint_dir / "baseline_cnn.pt",
            map_location=device,
            weights_only=True
        )
    )
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    results = []

    for method in ["normal", "distillation"]:
        print(f"\nTraining student: {method}")

        # Same initialization for a fair comparison
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        student = StudentCNN().to(device)

        train_student(
            student,
            teacher,
            train_loader,
            device,
            distilled=(method == "distillation")
        )

        accuracy = evaluate(student, test_loader, device)

        parameters = sum(
            p.numel() for p in student.parameters()
        )

        checkpoint = checkpoint_dir / f"student_{method}.pt"
        torch.save(student.state_dict(), checkpoint)

        size_mb = checkpoint.stat().st_size / (1024**2)
        latency_ms = measure_latency(student, device)

        results.append([
            method,
            round(accuracy, 2),
            parameters,
            round(size_mb, 3),
            round(latency_ms, 3)
        ])

        print(
            f"  Accuracy: {accuracy:.2f}% | "
            f"Size: {size_mb:.3f} MB | "
            f"Latency: {latency_ms:.3f} ms"
        )

    output = results_dir / "student_comparison.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "test_accuracy",
            "parameters",
            "model_size_mb",
            "latency_ms"
        ])
        writer.writerows(results)

    print("\nStudent training completed successfully!")
    print("Results saved to:", output)


if __name__ == "__main__":
    main()
