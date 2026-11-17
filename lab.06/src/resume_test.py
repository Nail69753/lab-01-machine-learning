
# Lab 6 - Checkpoint and Resume Test

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(1)
            correct += (predictions == labels).sum().item()

    return 100 * correct / len(loader.dataset)


def train_epochs(model, optimizer, train_loader, test_loader,
                 device, start_epoch, end_epoch, history):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = evaluate(model, test_loader, device)

        history.append([epoch + 1, avg_loss, accuracy])

        print(
            f"  Epoch {epoch + 1}/5 | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )


def main():
    lab_dir = Path(__file__).resolve().parent.parent

    with open(lab_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    learning_rate = config["training"]["learning_rate"]
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

    test_loader = DataLoader(
        test_data, batch_size=256, shuffle=False, num_workers=0
    )

    def make_train_loader():
        return DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

    # Experiment 1: Uninterrupted 5-epoch training
    print("\nExperiment 1: Continuous training")
    set_seed(seed)
    continuous_model = CNN().to(device)
    continuous_optimizer = torch.optim.Adam(
        continuous_model.parameters(), lr=learning_rate
    )

    continuous_history = []
    train_epochs(
        continuous_model, continuous_optimizer,
        make_train_loader(), test_loader, device,
        0, 5, continuous_history
    )

    # Experiment 2: Train 3 epochs and save checkpoint
    print("\nExperiment 2: Train 3 epochs, save, resume")
    set_seed(seed)
    resumed_model = CNN().to(device)
    resumed_optimizer = torch.optim.Adam(
        resumed_model.parameters(), lr=learning_rate
    )

    resumed_history = []
    train_loader = make_train_loader()

    train_epochs(
        resumed_model, resumed_optimizer,
        train_loader, test_loader, device,
        0, 3, resumed_history
    )

    checkpoint_dir = lab_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "resume_epoch3.pt"

    torch.save({
        "epoch": 3,
        "model_state_dict": resumed_model.state_dict(),
        "optimizer_state_dict": resumed_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "history": resumed_history,
    }, checkpoint_path)

    print("\nCheckpoint saved at epoch 3!")

    # Simulate restarting the model and optimizer
    restored_model = CNN().to(device)
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(), lr=learning_rate
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    restored_history = checkpoint["history"]

    print("Resuming from epoch 4...")

    train_epochs(
        restored_model, restored_optimizer,
        train_loader, test_loader, device,
        checkpoint["epoch"], 5, restored_history
    )

    # Save comparison results
    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = results_dir / "resume_comparison.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "epoch", "train_loss",
            "test_accuracy_percent"
        ])

        for epoch, loss, accuracy in continuous_history:
            writer.writerow([
                "continuous", epoch,
                round(loss, 4), round(accuracy, 2)
            ])

        for epoch, loss, accuracy in restored_history:
            writer.writerow([
                "resumed", epoch,
                round(loss, 4), round(accuracy, 2)
            ])

    print("\nResume test completed successfully!")
    print("Results saved to:", output)


if __name__ == "__main__":
    main()
