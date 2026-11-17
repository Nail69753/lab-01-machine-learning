
# Lab 7 - Compression Benchmark

import csv
import io
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from baseline import BaselineCNN
from student import StudentCNN


def measure_latency(model, repeats=200):
    model = model.cpu().eval()
    sample = torch.randn(1, 1, 28, 28)

    with torch.inference_mode():
        for _ in range(30):
            model(sample)

        start = time.perf_counter()

        for _ in range(repeats):
            model(sample)

        elapsed = time.perf_counter() - start

    return elapsed * 1000 / repeats


def serialized_size_mb(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return len(buffer.getvalue()) / (1024 ** 2)


def load_model(model_class, checkpoint):
    model = model_class()
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )
    )
    return model.eval()


def main():
    lab_dir = Path(__file__).resolve().parent.parent
    checkpoint_dir = lab_dir / "checkpoints"
    results_dir = lab_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_model(
        BaselineCNN,
        checkpoint_dir / "baseline_cnn.pt"
    )

    normal_student = load_model(
        StudentCNN,
        checkpoint_dir / "student_normal.pt"
    )

    distilled_student = load_model(
        StudentCNN,
        checkpoint_dir / "student_distillation.pt"
    )

    models = {
        "baseline": baseline,
        "student_normal": normal_student,
        "student_distillation": distilled_student
    }

    rows = []

    for name, model in models.items():
        parameters = sum(
            p.numel() for p in model.parameters()
        )

        size_mb = serialized_size_mb(model.state_dict())
        latency_ms = measure_latency(model)

        rows.append([
            name,
            parameters,
            round(size_mb, 4),
            round(latency_ms, 4)
        ])

        print(
            f"{name}: "
            f"{parameters:,} parameters | "
            f"{size_mb:.4f} MB | "
            f"{latency_ms:.4f} ms"
        )

    output = results_dir / "compression_benchmark.csv"

    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "parameters",
            "size_mb",
            "cpu_latency_ms"
        ])
        writer.writerows(rows)

    # Dense versus sparse storage comparison
    sparsity_rows = []

    for sparsity in [0, 0.25, 0.5, 0.75, 0.9, 0.95]:
        model = load_model(
            BaselineCNN,
            checkpoint_dir / "baseline_cnn.pt"
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

            # Make pruning permanent before serialization
            for module, name in layers:
                prune.remove(module, name)

        # Dense state_dict keeps all zeros in full tensors
        dense_size = serialized_size_mb(model.state_dict())

        # Sparse COO storage for pruned weight tensors
        sparse_state = {}

        for name, tensor in model.state_dict().items():
            if name.endswith("weight") and tensor.ndim >= 2:
                sparse_state[name] = tensor.to_sparse_coo().coalesce()
            else:
                sparse_state[name] = tensor

        sparse_size = serialized_size_mb(sparse_state)

        latency = measure_latency(model)

        sparsity_rows.append([
            sparsity,
            round(dense_size, 4),
            round(sparse_size, 4),
            round(latency, 4)
        ])

        print(
            f"Sparsity {sparsity:.0%}: "
            f"Dense={dense_size:.4f} MB | "
            f"Sparse={sparse_size:.4f} MB | "
            f"CPU={latency:.4f} ms"
        )

    sparse_output = results_dir / "dense_vs_sparse.csv"

    with open(
        sparse_output,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "sparsity",
            "dense_size_mb",
            "sparse_size_mb",
            "cpu_latency_ms"
        ])
        writer.writerows(sparsity_rows)

    print("\nCompression benchmark completed successfully!")
    print("Results saved to:", results_dir)


if __name__ == "__main__":
    main()
