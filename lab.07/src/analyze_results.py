
# Lab 7 - Results Analysis and Visualization

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    lab_dir = Path(__file__).resolve().parent.parent
    results_dir = lab_dir / "results"
    figures_dir = lab_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    global_results = read_csv(
        results_dir / "global_pruning_finetuned.csv"
    )
    layerwise_results = read_csv(
        results_dir / "layerwise_pruning.csv"
    )
    student_results = read_csv(
        results_dir / "student_comparison.csv"
    )

    # Figure 1: Accuracy vs sparsity
    global_sparsity = [
        float(row["actual_sparsity"]) * 100
        for row in global_results
    ]
    global_accuracy = [
        float(row["accuracy_after_finetuning"])
        for row in global_results
    ]

    layer_sparsity = [
        float(row["actual_sparsity"]) * 100
        for row in layerwise_results
    ]
    layer_accuracy = [
        float(row["accuracy_after_finetuning"])
        for row in layerwise_results
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(
        global_sparsity,
        global_accuracy,
        marker="o",
        label="Global pruning"
    )
    plt.plot(
        layer_sparsity,
        layer_accuracy,
        marker="s",
        label="Layer-wise pruning"
    )

    plt.xlabel("Actual sparsity (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Fashion-MNIST: Accuracy vs Sparsity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "accuracy_vs_sparsity.png",
        dpi=200
    )
    plt.close()

    # Figure 2: Student model comparison
    methods = [row["method"] for row in student_results]
    accuracies = [
        float(row["test_accuracy"])
        for row in student_results
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(methods, accuracies)
    plt.xlabel("Training method")
    plt.ylabel("Test accuracy (%)")
    plt.title("Student CNN: Normal vs Distillation")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "student_accuracy.png",
        dpi=200
    )
    plt.close()

    # Figure 3: Student latency comparison
    latencies = [
        float(row["latency_ms"])
        for row in student_results
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(methods, latencies)
    plt.xlabel("Training method")
    plt.ylabel("Inference latency (ms/image)")
    plt.title("Student CNN: Inference Latency")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "student_latency.png",
        dpi=200
    )
    plt.close()

    # Print summary
    print("\nStudent model comparison:")
    print(
        f"{'Method':<15}"
        f"{'Accuracy (%)':<16}"
        f"{'Size (MB)':<14}"
        f"{'Latency (ms)':<14}"
    )

    for row in student_results:
        print(
            f"{row['method']:<15}"
            f"{row['test_accuracy']:<16}"
            f"{row['model_size_mb']:<14}"
            f"{row['latency_ms']:<14}"
        )

    print("\nAnalysis completed successfully!")
    print("Figures saved to:", figures_dir)


if __name__ == "__main__":
    main()
