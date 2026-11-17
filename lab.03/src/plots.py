
import csv
from pathlib import Path
import matplotlib.pyplot as plt

results_dir = Path("lab.03/results")

# Read training results
with open(results_dir / "training_results.csv") as file:
    training = list(csv.DictReader(file))

# Read latency results
with open(results_dir / "latency_results.csv") as file:
    latency = list(csv.DictReader(file))

# Parameter counts from the three architectures
parameters = {
    "MLP": 235146,
    "CNN": 20458,
    "Attention": 70794
}

accuracy = {
    row["model"]: float(row["test_accuracy"]) * 100
    for row in training
}

batch1_latency = {
    row["model"]: float(row["median_ms"])
    for row in latency
    if int(row["batch_size"]) == 1
}

# Accuracy vs Parameters
plt.figure(figsize=(8, 5))

for name in accuracy:
    plt.scatter(parameters[name], accuracy[name], s=100)
    plt.annotate(
        name,
        (parameters[name], accuracy[name]),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Parameters")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / "accuracy_vs_parameters.png")
plt.close()

# Accuracy vs Latency
plt.figure(figsize=(8, 5))

for name in accuracy:
    plt.scatter(batch1_latency[name], accuracy[name], s=100)
    plt.annotate(
        name,
        (batch1_latency[name], accuracy[name]),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Median Inference Latency (ms, batch=1)")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Latency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / "accuracy_vs_latency.png")
plt.close()

print("Both plots saved successfully!")
