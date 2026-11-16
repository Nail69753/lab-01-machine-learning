import os
import csv
import time
import random
import joblib
import psutil
import numpy as np
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Results folder
RESULTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)
os.makedirs(RESULTS, exist_ok=True)

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split dataset: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Define models
models = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

process = psutil.Process(os.getpid())

accuracy_results = []
measurement_results = []

# Train and evaluate models
for name, create_model in models.items():

    print(f"\nModel: {name}")

    # Training warm-up
    warmup_model = create_model()
    warmup_model.fit(X_train, y_train)

    # Measure training time
    training_times = []
    peak_memory = process.memory_info().rss

    for i in range(5):
        model = create_model()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        end = time.perf_counter()

        training_times.append((end - start) * 1000)

        peak_memory = max(
            peak_memory,
            process.memory_info().rss
        )

    training_time = np.median(training_times)

    # Calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Inference warm-up
    sample = X_test[:1]
    model.predict(sample)

    # Measure inference latency
    inference_times = []

    for i in range(100):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        inference_times.append((end - start) * 1000)

        peak_memory = max(
            peak_memory,
            process.memory_info().rss
        )

    inference_time = np.median(inference_times)

    # Save model
    filename = name.lower().replace(" ", "_")
    model_path = os.path.join(
        RESULTS, filename + ".joblib"
    )

    joblib.dump(model, model_path)

    model_bytes = os.path.getsize(model_path)
    model_kb = model_bytes / 1024

    memory_mb = peak_memory / (1024 ** 2)

    # Save measurements
    accuracy_results.append({
        "model": name,
        "accuracy": f"{accuracy:.4f}"
    })

    measurement_results.append({
        "model": name,
        "training_ms": round(float(training_time), 4),
        "inference_ms": round(float(inference_time), 4),
        "model_bytes": model_bytes,
        "model_kb": round(model_kb, 4),
        "peak_memory_mb": round(memory_mb, 4)
    })

    # Display results
    print("Accuracy:", f"{accuracy:.4f}")
    print("Training time:", round(training_time, 4), "ms")
    print("Inference time:", round(inference_time, 4), "ms")
    print("Model size:", round(model_kb, 4), "KB")
    print("Memory:", round(memory_mb, 4), "MB")


# Save results to CSV files
def save_csv(filename, rows):
    path = os.path.join(RESULTS, filename)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


save_csv("baseline_accuracy.csv", accuracy_results)
save_csv("measurements.csv", measurement_results)

print("\nLab 1 completed!")
print("Results saved in lab01/results/")