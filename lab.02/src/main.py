import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)

digits = load_digits()

X = digits.data.astype(np.float64) / 16.0
y = digits.target

Y = np.eye(10)[y]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("Dataset loaded successfully!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("One-hot labels shape:", Y_train.shape)

def initialize_network(width):
    weights = {
        "W1": np.random.normal(0, 0.01, (64, width)),
        "b1": np.zeros(width),
        "W2": np.random.normal(0, 0.01, (width, width)),
        "b2": np.zeros(width),
        "W3": np.random.normal(0, 0.01, (width, 10)),
        "b3": np.zeros(10)
    }
    return weights

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    z3 = a2 @ params["W3"] + params["b3"]
    output = softmax(z3)

    return output


for width in [16, 64]:
    params = initialize_network(width)
    predictions = forward(X_train[:32], params)

    assert predictions.shape == (32, 10)
    assert np.allclose(predictions.sum(axis=1), 1, atol=1e-6)

    print(f"Width {width}: forward pass successful!")
    print("Output shape:", predictions.shape)
    print("Row probability sum:", predictions[0].sum())



# Step 6: Cross-Entropy Loss and Backpropagation


def cross_entropy(predictions, labels):
    return -np.mean(
        np.sum(labels * np.log(predictions), axis=1)
    )


# Forward pass with saved intermediate values
def forward_with_cache(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    z3 = a2 @ params["W3"] + params["b3"]
    predictions = softmax(z3)

    cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "predictions": predictions
    }

    return predictions, cache

# Backpropagation using saved forward values
def backward_from_cache(X, Y, params, cache):
    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    a2 = cache["a2"]
    predictions = cache["predictions"]

    batch_size = X.shape[0]

    dz3 = (predictions - Y) / batch_size
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3
    }

def backward(X, Y, params):
   
    predictions, cache = forward_with_cache(X, params)

    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    a2 = cache["a2"]


    # Backpropagation
    batch_size = X.shape[0]

    dz3 = (predictions - Y) / batch_size
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3
    }


# Test loss and gradients
params = initialize_network(16)

predictions = forward(X_train[:32], params)
loss = cross_entropy(predictions, Y_train[:32])

gradients = backward(
    X_train[:32],
    Y_train[:32],
    params
)

print("\nStep 6: Loss and Backpropagation")
print("Cross-Entropy Loss:", round(loss, 4))
print("Backpropagation successful!")

for name, gradient in gradients.items():
    print(name, "gradient shape:", gradient.shape)

# Step 7: Numerical Gradient Checking

def gradient_check(X, Y, params, num_checks=10, epsilon=1e-5):
    analytical_grads = backward(X, Y, params)

    rng = np.random.default_rng(42)
    passed = 0

    print("\nStep 7: Gradient Checking")

    # Check 10 randomly selected weights
    weight_names = ["W1", "W2", "W3"]

    for i in range(num_checks):
        name = rng.choice(weight_names)
        weights = params[name]

        index = tuple(
            rng.integers(0, size)
            for size in weights.shape
        )

        original_value = weights[index]

        # Loss with positive perturbation
        weights[index] = original_value + epsilon
        loss_plus = cross_entropy(
            forward(X, params), Y
        )

        # Loss with negative perturbation
        weights[index] = original_value - epsilon
        loss_minus = cross_entropy(
            forward(X, params), Y
        )

        # Restore original weight
        weights[index] = original_value

        numerical_grad = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        analytical_grad = analytical_grads[name][index]

        relative_error = abs(
            numerical_grad - analytical_grad
        ) / max(
               1e-8,
            abs(numerical_grad) + abs(analytical_grad)
        )

        print(
            f"Numerical: {numerical_grad:.12e}, "
            f"Analytical: {analytical_grad:.12e}"
        )

        success = relative_error < 1e-6

        if success:
            passed += 1

        print(
            f"Check {i+1}: {name}{index} | "
            f"Relative error: {relative_error:.2e} | "
            f"{'PASS' if success else 'FAIL'}"
        )

    print(f"\nGradient checks passed: {passed}/{num_checks}")

    return passed


# Run gradient checking
check_params = initialize_network(16)

gradient_check(
    X_train[:32],
    Y_train[:32],
    check_params
)

# Step 8: Neural Network Training

def train_network(width, learning_rate, epochs=50, batch_size=32):
    params = initialize_network(width)
    losses = []

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        # Mini-batch training
        for start in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            Y_batch = Y_shuffled[start:start + batch_size]

            gradients = backward(X_batch, Y_batch, params)

            # Update weights and biases
            for name in params:
                params[name] -= learning_rate * gradients[name]

        # Record training loss
        predictions = forward(X_train, params)
        loss = cross_entropy(predictions, Y_train)
        losses.append(loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {loss:.4f}"
        )

    # Test accuracy
    test_predictions = forward(X_test, params)
    predicted_labels = np.argmax(test_predictions, axis=1)
    true_labels = np.argmax(Y_test, axis=1)

    accuracy = np.mean(predicted_labels == true_labels)

    print(f"Final Test Accuracy: {accuracy:.4f}")

    return params, losses, accuracy



# Step 9: Compare all learning rates and widths

import matplotlib.pyplot as plt
import os
import csv

print("\nStep 9: Running All Experiments")

results_dir = os.path.join("lab.02", "results")
os.makedirs(results_dir, exist_ok=True)

experiment_results = []

plt.figure(figsize=(10, 6))

for width in [16, 64]:
    for learning_rate in [0.001, 0.01, 0.1]:

        print(
            f"\nTraining: width={width}, "
            f"learning_rate={learning_rate}"
        )

        params, losses, accuracy = train_network(
            width=width,
            learning_rate=learning_rate,
            epochs=50,
            batch_size=32
        )

        plt.plot(
            range(1, 51),
            losses,
            label=f"W={width}, LR={learning_rate}"
        )

        experiment_results.append({
            "width": width,
            "learning_rate": learning_rate,
            "test_accuracy": accuracy,
            "final_loss": losses[-1]
        })

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Lab 2: Neural Network Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(results_dir, "loss_curves.png"),
    dpi=150
)
plt.close()

with open(
    os.path.join(results_dir, "experiment_results.csv"),
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "width",
            "learning_rate",
            "test_accuracy",
            "final_loss"
        ]
    )
    writer.writeheader()
    writer.writerows(experiment_results)

print("\nAll experiments completed!")
print("Loss curves saved to lab.02/results/loss_curves.png")
print("Results saved to lab.02/results/experiment_results.csv")

# Step 10: Forward and Backward Timing


# Step 10: Separate Forward and Backward Timing

import time

print("\nStep 10: Separate Forward and Backward Timing")

timing_results = []

for width in [16, 64]:
    params = initialize_network(width)

    X_batch = X_train[:32]
    Y_batch = Y_train[:32]

    forward_times = []
    backward_times = []

    # Warm-up
    for _ in range(10):
        predictions, cache = forward_with_cache(X_batch, params)
        backward_from_cache(X_batch, Y_batch, params, cache)

    # Measure 100 runs
    for _ in range(100):
        start = time.perf_counter()
        predictions, cache = forward_with_cache(X_batch, params)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        gradients = backward_from_cache(
            X_batch, Y_batch, params, cache
        )
        backward_times.append(time.perf_counter() - start)

    forward_ms = np.median(forward_times) * 1000
    backward_ms = np.median(backward_times) * 1000

    print(f"\nWidth: {width}")
    print(f"Forward median: {forward_ms:.4f} ms")
    print(f"Backward median: {backward_ms:.4f} ms")

    timing_results.append({
        "width": width,
        "forward_median_ms": forward_ms,
        "backward_median_ms": backward_ms
    })

with open(
    os.path.join(results_dir, "timing_results.csv"),
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "width",
            "forward_median_ms",
            "backward_median_ms"
        ]
    )
    writer.writeheader()
    writer.writerows(timing_results)

print("\nTiming completed!")
print("Results saved to lab.02/results/timing_results.csv")
