
# Lab 7 - Structured Filter Pruning

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baseline import BaselineCNN


class SmallerCNN(nn.Module):
    def __init__(self, channels1, channels2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, channels1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels1, channels2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels2 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def prune_filters(original, fraction):
    conv1 = original.features[0]
    conv2 = original.features[3]
    fc1 = original.classifier[1]

    keep1 = max(1, round(conv1.out_channels * (1 - fraction)))
    keep2 = max(1, round(conv2.out_channels * (1 - fraction)))

    norm1 = torch.linalg.vector_norm(
        conv1.weight.detach().flatten(1), dim=1
    )
    norm2 = torch.linalg.vector_norm(
        conv2.weight.detach().flatten(1), dim=1
    )

    idx1 = torch.topk(norm1, keep1).indices.sort().values
    idx2 = torch.topk(norm2, keep2).indices.sort().values

    smaller = SmallerCNN(keep1, keep2).to(
        next(original.parameters()).device
    )

    with torch.no_grad():
        smaller.features[0].weight.copy_(conv1.weight[idx1])
        smaller.features[0].bias.copy_(conv1.bias[idx1])

        smaller.features[3].weight.copy_(
            conv2.weight[idx2][:, idx1]
        )
        smaller.features[3].bias.copy_(conv2.bias[idx2])

        # Each conv2 channel contributes 7x7 features.
        feature_indices = (
            idx2[:, None] * 49
            + torch.arange(49, device=idx2.device)[None, :]
        ).reshape(-1)

        smaller.classifier[1].weight.copy_(
            fc1.weight[:, feature_indices]
        )
        smaller.classifier[1].bias.copy_(fc1.bias)

        smaller.classifier[3].weight.copy_(
            original.classifier[3].weight
        )
        smaller.classifier[3].bias.copy_(
            original.classifier[3].bias
        )

    return smaller


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
    lab_dir = Path(__file__).resolve().parent.parent
    repo_dir = lab_dir.parent

    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = lab_dir / "checkpoints"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_data = datasets.FashionMNIST(
        root=str(repo_dir / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(
        test_data,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    baseline = BaselineCNN().to(device)
    baseline.load_state_dict(
        torch.load(
            checkpoint_dir / "baseline_cnn.pt",
            map_location=device,
            weights_only=True
        )
    )

    original_params = sum(
        p.numel() for p in baseline.parameters()
    )

    results = []

    for fraction in [0.10, 0.25, 0.50]:
        model = prune_filters(baseline, fraction)

        accuracy = evaluate(model, test_loader, device)
        parameters = sum(p.numel() for p in model.parameters())

        checkpoint = checkpoint_dir / (
            f"structured_{int(fraction * 100)}.pt"
        )
        torch.save(model.state_dict(), checkpoint)

        results.append([
            fraction,
            parameters,
            round(100 * (1 - parameters / original_params), 2),
            round(accuracy, 2)
        ])

        print(
            f"Filter pruning: {fraction:.0%} | "
            f"Parameters: {parameters:,} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    output = results_dir / "structured_pruning.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filter_pruning_fraction",
            "parameters",
            "parameter_reduction_percent",
            "test_accuracy"
        ])
        writer.writerows(results)

    print("\nStructured pruning completed successfully!")
    print("Results saved to:", output)


if __name__ == "__main__":
    main()
