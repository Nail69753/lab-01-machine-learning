# STEP 10: PLOT WORKER BENCHMARK

def plot_worker_results(worker_results):
    import matplotlib.pyplot as plt

    workers = [row[0] for row in worker_results]
    speeds = [row[2] for row in worker_results]

    plt.figure(figsize=(8, 5))
    plt.plot(workers, speeds, marker="o")
    plt.xlabel("Number of Workers")
    plt.ylabel("Throughput (samples/second)")
    plt.title("FashionMNIST DataLoader Worker Benchmark")
    plt.grid(True)
    plt.savefig("lab.06/results/workers_plot.png")
    plt.close()

    print("Worker plot saved successfully!")

# Lab 6 - Training Pipeline

import torch
import torch.nn as nn


# CNN model from Lab 3
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)



def main():
    # Test model
    model = CNN()
    print("Lab 6 - CNN Model")
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("CNN model created successfully!")

    # STEP 2: Load YAML configuration

    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("\nLab 6 - YAML Configuration")
    print("Experiment:", config["experiment"]["name"])
    print("Seed:", config["training"]["seed"])
    print("Epochs:", config["training"]["epochs"])
    print("Batch size:", config["training"]["batch_size"])
    print("Learning rate:", config["training"]["learning_rate"])
    print("YAML configuration loaded successfully!")

    # =====================================
    # STEP 3: LOAD FASHIONMNIST DATASET
    # =====================================

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("\nLab 6 - Loading FashionMNIST")

    # Reproducibility
    torch.manual_seed(config["training"]["seed"])

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=config["data"]["root"],
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Training batches:", len(train_loader))
    print("FashionMNIST dataset loaded successfully!")

    # =====================================
    # STEP 4: TRAIN CNN USING YAML CONFIG
    # =====================================

    import torch.optim as optim

    print("\nLab 6 - CNN Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        average_loss = total_loss / total
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Train Accuracy: {accuracy:.2f}%"
        )

    print("CNN training completed successfully!")

    # =====================================
    # STEP 5: SAVE MODEL CHECKPOINT
    # =====================================

    print("\nLab 6 - Saving Checkpoint")

    checkpoint_folder = Path(
        config["checkpoint"]["directory"]
    )

    checkpoint_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = checkpoint_folder / "cnn_checkpoint.pth"

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
            "accuracy": accuracy,
        },
        checkpoint_path
    )

    print("Checkpoint saved to:", checkpoint_path)
    print("Checkpoint saved successfully!")

    # =====================================
    # STEP 6: MLFLOW EXPERIMENT LOGGING
    # =====================================

    import mlflow

    print("\nLab 6 - MLflow Experiment Logging")

    # Save MLflow experiments inside Lab 6
    mlflow_folder = Path(__file__).resolve().parent.parent / "mlruns"
    mlflow.set_tracking_uri(mlflow_folder.as_uri())

    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name="cnn_training"):
        # Log training parameters
        mlflow.log_params(config["training"])

        # Log final training metrics
        mlflow.log_metric("train_loss", average_loss)
        mlflow.log_metric("train_accuracy", accuracy)

        # Save checkpoint as an artifact
        mlflow.log_artifact(str(checkpoint_path))

    print("MLflow experiment logged successfully!")

    # =====================================
    # STEP 7: PROFILE ONE TRAINING EPOCH
    # =====================================

    from torch.profiler import profile, ProfilerActivity, record_function

    print("\nLab 6 - Training Profiler")

    profile_model = CNN().to(device)
    profile_optimizer = optim.Adam(
        profile_model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    profile_model.train()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:

        data_iterator = iter(train_loader)

        for _ in range(len(train_loader)):

            with record_function("data_loading"):
                images, labels = next(data_iterator)
                images = images.to(device)
                labels = labels.to(device)

            with record_function("forward_pass"):
                outputs = profile_model(images)
                loss = criterion(outputs, labels)

            with record_function("backward_pass"):
                profile_optimizer.zero_grad()
                loss.backward()

            with record_function("optimizer_step"):
                profile_optimizer.step()

    print("\nProfiler results:")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=15
        )
    )

    profiler_path = Path(config["results"]["directory"]) / "profiler.txt"
    profiler_path.parent.mkdir(parents=True, exist_ok=True)

    with open(profiler_path, "w", encoding="utf-8") as f:
        f.write(
            prof.key_averages().table(
                sort_by="self_cpu_time_total",
                row_limit=30
            )
        )

    print("Profiler results saved to:", profiler_path)
    print("Training profiler completed successfully!")

    # =====================================
    # STEP 8: PROFILER TIME BREAKDOWN
    # =====================================

    print("\nLab 6 - Profiler Time Breakdown")

    stages = [
        "data_loading",
        "forward_pass",
        "backward_pass",
        "optimizer_step"
    ]

    # Profiler time is reported in microseconds
    stage_times = {stage: 0.0 for stage in stages}

    for event in prof.key_averages():
        if event.key in stage_times:
            stage_times[event.key] = event.cpu_time_total

    total_stage_time = sum(stage_times.values())

    breakdown_path = (
        Path(config["results"]["directory"]) / "profile_breakdown.txt"
    )

    with open(breakdown_path, "w", encoding="utf-8") as f:
        f.write("LAB 6 - PROFILER TIME BREAKDOWN\n")
        f.write("================================\n\n")

        for stage in stages:
            percentage = (
                100.0 * stage_times[stage] / total_stage_time
                if total_stage_time > 0 else 0.0
            )

            line = (
                f"{stage}: {stage_times[stage] / 1000:.2f} ms "
                f"({percentage:.2f}%)"
            )

            print(line)
            f.write(line + "\n")

    print("Profile breakdown saved to:", breakdown_path)
    print("Profiler breakdown completed successfully!")

    # =====================================
    # STEP 9: NUM_WORKERS BENCHMARK
    # =====================================

    import time

    print("\nLab 6 - DataLoader Workers Benchmark")

    worker_counts = [0, 1, 2, 4, 8]
    worker_results = []

    # Benchmark one full pass through the dataset
    for workers in worker_counts:
        loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=workers,
            persistent_workers=(workers > 0)
        )

        start = time.perf_counter()
        total_samples = 0

        for images, labels in loader:
            total_samples += images.size(0)

        elapsed = time.perf_counter() - start
        throughput = total_samples / elapsed

        worker_results.append((workers, elapsed, throughput))

        print(
            f"Workers: {workers} | "
            f"Time: {elapsed:.2f}s | "
            f"Throughput: {throughput:.2f} samples/s"
        )

        del loader

    results_path = Path(config["results"]["directory"]) / "workers_results.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Workers | Time (s) | Throughput (samples/s)\n")

        for workers, elapsed, throughput in worker_results:
            f.write(f"{workers} | {elapsed:.4f} | {throughput:.2f}\n")

        print("Worker benchmark completed successfully!")
    plot_worker_results(worker_results)





if __name__ == "__main__":
    main()
