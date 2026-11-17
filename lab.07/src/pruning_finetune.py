
# Lab 7 - Global Pruning with 3-Epoch Fine-Tuning

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baseline import BaselineCNN


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


def main():
    seed = 42
    lab_dir = Path(__file__).resolve().parent.parent
    repo_dir = lab_dir.parent

    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

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

    test_loader = DataLoader(
        test_data, batch_size=256, shuffle=False, num_workers=0
    )

    checkpoint_path = lab_dir / "checkpoints" / "baseline_cnn.pt"
    sparsity_levels = [0, 0.25, 0.5, 0.75, 0.9, 0.95]

    results = []

    for sparsity in sparsity_levels:
        print(f"\nFine-tuning sparsity: {sparsity:.0%}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = BaselineCNN().to(device)
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True
            )
        )

        layers = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]

        if sparsity > 0:
            prune.global_unstructured(
                layers,
                pruning_method=prune.L1Unstructured,
                amount=sparsity
            )

        train_loader = DataLoader(
            train_data,
            batch_size=128,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed)
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0001
        )
        criterion = nn.CrossEntropyLoss()

        before_accuracy = evaluate(model, test_loader, device)

        for epoch in range(3):
            model.train()

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            accuracy = evaluate(model, test_loader, device)

            print(
                f"  Epoch {epoch + 1}/3 | "
                f"Accuracy: {accuracy:.2f}%"
            )

        total_weights = sum(
            module.weight.numel() for module, _ in layers
        )
        zero_weights = sum(
            (module.weight == 0).sum().item()
            for module, _ in layers
        )
        actual_sparsity = zero_weights / total_weights

        results.append([
            sparsity,
            round(actual_sparsity, 4),
            round(before_accuracy, 2),
            round(accuracy, 2)
        ])

        print(
            f"  Final sparsity: {actual_sparsity:.2%} | "
            f"Final accuracy: {accuracy:.2f}%"
        )

    output = results_dir / "global_pruning_finetuned.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target_sparsity",
            "actual_sparsity",
            "accuracy_before_finetuning",
            "accuracy_after_finetuning"
        ])
        writer.writerows(results)

    print("\nGlobal pruning fine-tuning completed successfully!")
    print("Results saved to:", output)


if __name__ == "__main__":
    main()
