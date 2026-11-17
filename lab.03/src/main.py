
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Reproducibility
torch.manual_seed(42)

# Step 1: Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Download Fashion-MNIST
train_full = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Split: 55,000 training + 5,000 validation
train_dataset, val_dataset = random_split(
    train_full,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)

print("Step 1: Fashion-MNIST Dataset")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

# Step 2: MLP Architecture
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


mlp_model = MLP()

print("\nStep 2: MLP Architecture")
print(mlp_model)

mlp_parameters = sum(
    p.numel() for p in mlp_model.parameters()
)
print("MLP parameters:", mlp_parameters)

# Test forward pass
with torch.no_grad():
    mlp_output = mlp_model(images)

print("MLP output shape:", mlp_output.shape)

# Step 3: CNN Architecture
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


cnn_model = CNN()

print("\nStep 3: CNN Architecture")
print(cnn_model)

cnn_parameters = sum(
    p.numel() for p in cnn_model.parameters()
)
print("CNN parameters:", cnn_parameters)

# Test forward pass
with torch.no_grad():
    cnn_output = cnn_model(images)

print("CNN output shape:", cnn_output.shape)

# Step 4: Attention Architecture
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_projection = nn.Linear(7 * 7, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.0
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.shape[0]

        # 28x28 -> 16 patches of 7x7
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.contiguous().view(
            batch_size, 16, 49
        )

        # Project each patch to 64 features
        x = self.patch_projection(patches)

        # Transformer encoder
        x = self.transformer(x)

        # Mean pooling over 16 patches
        x = x.mean(dim=1)

        return self.classifier(x)


attention_model = AttentionModel()

print("\nStep 4: Attention Architecture")
print(attention_model)

attention_parameters = sum(
    p.numel() for p in attention_model.parameters()
)
print("Attention parameters:", attention_parameters)

# Test forward pass
with torch.no_grad():
    attention_output = attention_model(images)

print("Attention output shape:", attention_output.shape)

# Step 5: Parameter Counting with torchinfo
from torchinfo import summary

print("\nStep 5: Model Parameter Summary")

models = {
    "MLP": mlp_model,
    "CNN": cnn_model,
    "Attention": attention_model
}

for name, model in models.items():
    print(f"\n{name} Model Summary")

    model_summary = summary(
        model,
        input_size=(1, 1, 28, 28),
        verbose=0
    )

    print("Total parameters:", model_summary.total_params)

# Step 6: FLOPs Profiling with THOP
from thop import profile

print("\nStep 6: FLOPs Profiling")

dummy_input = torch.randn(1, 1, 28, 28)

for name, model in models.items():
    model.eval()

    try:
        with torch.no_grad():
            macs, params = profile(
                model,
                inputs=(dummy_input,),
                verbose=False
            )

        print(f"\n{name}:")
        print(f"THOP MACs: {macs:,.0f}")
        print(f"Approximate FLOPs (2 x MACs): {2 * macs:,.0f}")
        print(f"THOP parameters: {params:,.0f}")

    except Exception as error:
        print(f"\n{name}: THOP profiling failed")
        print("Reason:", error)

# Step 7: Hand-Calculated FLOPs
print("\nStep 7: Hand-Calculated FLOPs")

# MLP: first Linear layer (784 -> 256)
mlp_flops = 2 * 784 * 256

# CNN: first Conv2d layer
# Kernel 3x3, input channels 1, output channels 16
# Output image size 28x28
cnn_flops = 2 * 3 * 3 * 1 * 16 * 28 * 28

# Attention: QKV projections in one encoder layer
# Sequence length 16, embedding dimension 64
attention_qkv_flops = 2 * 16 * 64 * 3 * 64

# Attention scores Q x K^T in one encoder layer
attention_scores_flops = 2 * 16 * 16 * 64

print("MLP first Linear FLOPs:", f"{mlp_flops:,}")
print("CNN first Conv2d FLOPs:", f"{cnn_flops:,}")
print("Attention QKV FLOPs:", f"{attention_qkv_flops:,}")
print("Attention scores FLOPs:", f"{attention_scores_flops:,}")

# Step 8: Inference Latency
import time
import statistics
import csv
from pathlib import Path

print("\nStep 8: Inference Latency")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

latency_results = []

def measure_latency(model, batch_size, runs):
    model = model.to(device)
    model.eval()

    sample = torch.randn(
        batch_size, 1, 28, 28, device=device
    )

    # Warm-up runs
    with torch.inference_mode():
        for _ in range(10):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []

        for _ in range(runs):
            start = time.perf_counter()
            model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            times.append(elapsed_ms)

    return statistics.median(times)


for name, model in models.items():
    for batch_size, runs in [(1, 100), (128, 20)]:
        median_ms = measure_latency(
            model, batch_size, runs
        )

        print(
            f"{name} | Batch {batch_size} | "
            f"Median: {median_ms:.4f} ms"
        )

        latency_results.append({
            "model": name,
            "batch_size": batch_size,
            "median_ms": median_ms
        })

# Save results
results_dir = Path("lab.03/results")
results_dir.mkdir(parents=True, exist_ok=True)

with open(
    results_dir / "latency_results.csv",
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "model", "batch_size", "median_ms"
        ]
    )
    writer.writeheader()
    writer.writerows(latency_results)

print("Latency results saved!")

# Step 9: Train and Evaluate Models
import torch.optim as optim

print("\nStep 9: Training Models")

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images_batch, labels_batch in loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images_batch)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels_batch
            ).sum().item()

            total += labels_batch.size(0)

    return correct / total


def train_model(model, name, epochs=5):
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images_batch, labels_batch in train_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * labels_batch.size(0)
            )

        average_loss = total_loss / len(train_dataset)
        val_accuracy = evaluate_accuracy(
            model, val_loader
        )

        print(
            f"{name} | Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}",
            flush=True
        )

    test_accuracy = evaluate_accuracy(
        model, test_loader
    )

    print(
        f"{name} | Test Accuracy: "
        f"{test_accuracy:.4f}",
        flush=True
    )

    return test_accuracy


training_results = []

for name, model in models.items():
    accuracy = train_model(model, name)

    training_results.append({
        "model": name,
        "test_accuracy": accuracy
    })

with open(
    results_dir / "training_results.csv",
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model", "test_accuracy"]
    )
    writer.writeheader()
    writer.writerows(training_results)

print("\nTraining results saved!")

# Step 10: Accuracy vs Cost Plots
import matplotlib.pyplot as plt

print("\nStep 10: Accuracy vs Cost Plots")

parameter_counts = {
    name: sum(p.numel() for p in model.parameters())
    for name, model in models.items()
}

accuracy_by_model = {
    row["model"]: row["test_accuracy"] * 100
    for row in training_results
}

latency_by_model = {
    row["model"]: row["median_ms"]
    for row in latency_results
    if row["batch_size"] == 1
}

model_names = list(models.keys())

# Plot 1: Accuracy vs Parameters
plt.figure(figsize=(8, 5))

for name in model_names:
    plt.scatter(
        parameter_counts[name],
        accuracy_by_model[name],
        s=100,
        label=name
    )

    plt.annotate(
        name,
        (
            parameter_counts[name],
            accuracy_by_model[name]
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Parameters")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "accuracy_vs_parameters.png",
    dpi=150
)
plt.close()

# Plot 2: Accuracy vs Latency (batch size 1)
plt.figure(figsize=(8, 5))

for name in model_names:
    plt.scatter(
        latency_by_model[name],
        accuracy_by_model[name],
        s=100,
        label=name
    )

    plt.annotate(
        name,
        (
            latency_by_model[name],
            accuracy_by_model[name]
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Median Inference Latency (ms, batch=1)")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "accuracy_vs_latency.png",
    dpi=150
)
plt.close()

print("Accuracy vs Parameters plot saved!")
print("Accuracy vs Latency plot saved!")
