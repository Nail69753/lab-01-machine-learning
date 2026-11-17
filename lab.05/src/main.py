
# Lab 5: Frameworks and Automatic Differentiation

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # Power
    def __pow__(self, power):
        out = Value(
            self.data ** power,
            (self,),
            f"**{power}"
        )

        def _backward():
            self.grad += (
                power
                * (self.data ** (power - 1))
                * out.grad
            )

        out._backward = _backward
        return out

    # ReLU activation
    def relu(self):
        out = Value(
            max(0.0, self.data),
            (self,),
            "ReLU"
        )

        def _backward():
            self.grad += (
                (1.0 if self.data > 0 else 0.0)
                * out.grad
            )

        out._backward = _backward
        return out

    # Backpropagation
    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


# Test addition and multiplication
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)

d = a * b + c

print("Lab 5 - Step 2: Operations")
print("a =", a)
print("b =", b)
print("c =", c)
print("d = a * b + c")
print("d =", d)
print("Parent nodes:", len(d._prev))
print("Operations created successfully!")

# Test Power and ReLU
x = Value(3.0)
y = (x ** 2).relu()

print("\nLab 5 - Power and ReLU Test")
print("x =", x)
print("y =", y)
print("Power and ReLU created successfully!")

# Test backward propagation
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)

result = ((a * b + c) ** 2).relu()

result.backward()

print("\nLab 5 - Backward Test")
print("Result:", result.data)
print("Gradient a:", a.grad)
print("Gradient b:", b.grad)
print("Gradient c:", c.grad)
print("Backward completed successfully!")

# =====================================
# STEP 5: COMPARE WITH PYTORCH
# =====================================

import torch

print("\nLab 5 - PyTorch Gradient Comparison")

# PyTorch variables
a_t = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
b_t = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
c_t = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)

# Same calculation as our Value engine
result_t = ((a_t * b_t + c_t) ** 2).relu().sum()

result_t.backward()

print("PyTorch result:", result_t.item())
print("PyTorch gradient a:", a_t.grad.item())
print("PyTorch gradient b:", b_t.grad.item())
print("PyTorch gradient c:", c_t.grad.item())

# Compare gradients
differences = [
    abs(a.grad - a_t.grad.item()),
    abs(b.grad - b_t.grad.item()),
    abs(c.grad - c_t.grad.item())
]

max_difference = max(differences)

print("Maximum gradient difference:", max_difference)

assert max_difference < 1e-6, "Gradient comparison failed!"

print("PyTorch comparison passed!")

# =====================================
# STEP 6: NUMERICAL GRADIENT CHECK
# =====================================

print("\nLab 5 - Numerical Gradient Check")

def calculate_output(a, b, c):
    return max(0.0, (a * b + c) ** 2)

def numerical_gradient(a, b, c, variable):
    h = 1e-5

    if variable == "a":
        return (
            calculate_output(a + h, b, c)
            - calculate_output(a - h, b, c)
        ) / (2 * h)

    if variable == "b":
        return (
            calculate_output(a, b + h, c)
            - calculate_output(a, b - h, c)
        ) / (2 * h)

    if variable == "c":
        return (
            calculate_output(a, b, c + h)
            - calculate_output(a, b, c - h)
        ) / (2 * h)


numerical_a = numerical_gradient(2.0, 3.0, 4.0, "a")
numerical_b = numerical_gradient(2.0, 3.0, 4.0, "b")
numerical_c = numerical_gradient(2.0, 3.0, 4.0, "c")

print("Numerical gradient a:", numerical_a)
print("Numerical gradient b:", numerical_b)
print("Numerical gradient c:", numerical_c)

numerical_differences = [
    abs(a.grad - numerical_a),
    abs(b.grad - numerical_b),
    abs(c.grad - numerical_c)
]

max_numerical_difference = max(numerical_differences)

print("Maximum numerical difference:", max_numerical_difference)

assert max_numerical_difference < 1e-6, (
    "Numerical gradient check failed!"
)

print("Numerical gradient check passed!")

# =====================================
# STEP 7: MAKE_MOONS DATASET
# =====================================

from sklearn.datasets import make_moons

print("\nLab 5 - Make Moons Dataset")

X_moons, y_moons = make_moons(
    n_samples=200,
    noise=0.15,
    random_state=42
)

print("Input shape:", X_moons.shape)
print("Target shape:", y_moons.shape)
print("First sample:", X_moons[0])
print("First label:", y_moons[0])
print("Make moons dataset created successfully!")

# =====================================
# STEP 8: BUILD TINY NEURAL NETWORK
# =====================================

import random
import math

print("\nLab 5 - Building Neural Network")

random.seed(42)

# Sigmoid activation using existing Value operations
def sigmoid_value(x):
    # Sigmoid will be implemented as a Value operation
    out = Value(1.0 / (1.0 + math.exp(-x.data)), (x,), "sigmoid")

    def _backward():
        x.grad += out.data * (1.0 - out.data) * out.grad

    out._backward = _backward
    return out


class TinyNetwork:
    def __init__(self):
        # 2 inputs -> 8 hidden neurons
        self.w1 = [
            [Value(random.uniform(-0.5, 0.5)) for _ in range(2)]
            for _ in range(8)
        ]
        self.b1 = [Value(0.0) for _ in range(8)]

        # 8 hidden neurons -> 1 output
        self.w2 = [
            Value(random.uniform(-0.5, 0.5))
            for _ in range(8)
        ]
        self.b2 = Value(0.0)

    def __call__(self, x):
        hidden = []

        for i in range(8):
            z = (
                self.w1[i][0] * x[0]
                + self.w1[i][1] * x[1]
                + self.b1[i]
            )
            hidden.append(z.relu())

        output = self.b2

        for i in range(8):
            output = output + self.w2[i] * hidden[i]

        return sigmoid_value(output)

    def parameters(self):
        params = []

        for row in self.w1:
            params.extend(row)

        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)

        return params


network = TinyNetwork()

sample_output = network(X_moons[0])

print("Network architecture: 2 -> 8 -> 1")
print("Number of parameters:", len(network.parameters()))
print("First prediction:", sample_output.data)
print("Neural network created successfully!")

# =====================================
# STEP 9: BINARY CROSS-ENTROPY LOSS
# =====================================

print("\nLab 5 - BCE Loss")

def bce_loss(prediction, target):
    # Avoid log(0)
    p = min(max(prediction.data, 1e-7), 1.0 - 1e-7)

    loss_data = -(
        target * math.log(p)
        + (1.0 - target) * math.log(1.0 - p)
    )

    out = Value(loss_data, (prediction,), "BCE")

    def _backward():
        if 1e-7 < prediction.data < 1.0 - 1e-7:
            derivative = (
                -target / p
                + (1.0 - target) / (1.0 - p)
            )
            prediction.grad += derivative * out.grad

    out._backward = _backward
    return out


prediction = network(X_moons[0])
loss = bce_loss(prediction, y_moons[0])

print("Prediction:", prediction.data)
print("Actual label:", y_moons[0])
print("BCE loss:", loss.data)

loss.backward()

print("Gradient calculated successfully!")
print("BCE loss created successfully!")

# =====================================
# STEP 10: TRAIN OUR NEURAL NETWORK
# =====================================

import time

print("\nLab 5 - Training Neural Network")

# Create a fresh network for training
train_network = TinyNetwork()

learning_rate = 0.1
epochs = 30

start_time = time.perf_counter()

for epoch in range(epochs):
    total_loss = 0.0
    correct = 0

    for inputs, target in zip(X_moons, y_moons):
        # Clear previous parameter gradients
        for parameter in train_network.parameters():
            parameter.grad = 0.0

        # Forward pass
        prediction = train_network(inputs)

        # Calculate loss
        loss = bce_loss(prediction, target)

        # Backward pass
        loss.backward()

        # Update parameters
        for parameter in train_network.parameters():
            parameter.data -= learning_rate * parameter.grad

        total_loss += loss.data

        predicted_label = 1 if prediction.data >= 0.5 else 0

        if predicted_label == target:
            correct += 1

    if (epoch + 1) % 5 == 0:
        average_loss = total_loss / len(X_moons)
        accuracy = correct / len(X_moons)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2%}"
        )

training_time = time.perf_counter() - start_time

print(f"Training time: {training_time:.4f} seconds")
print("Neural network training completed successfully!")

# =====================================
# STEP 11: PYTORCH TRAINING
# =====================================

import torch.nn as nn

print("\nLab 5 - PyTorch Training")

torch.manual_seed(42)
torch.set_num_threads(1)

# Convert dataset to PyTorch tensors
X_torch = torch.tensor(X_moons, dtype=torch.float32)
y_torch = torch.tensor(y_moons, dtype=torch.float32).reshape(-1, 1)

# Same architecture: 2 -> 8 -> 1
torch_network = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(torch_network.parameters(), lr=0.1)

torch_start = time.perf_counter()

for epoch in range(30):
    total_torch_loss = 0.0

    # Train one sample at a time, like our Value engine
    for i in range(len(X_torch)):
        optimizer.zero_grad()

        prediction_torch = torch_network(X_torch[i:i+1])
        loss_torch = criterion(prediction_torch, y_torch[i:i+1])

        loss_torch.backward()
        optimizer.step()

        total_torch_loss += loss_torch.item()

    if (epoch + 1) % 5 == 0:
        print(
            f"PyTorch Epoch {epoch + 1}/30 | "
            f"Loss: {total_torch_loss / len(X_torch):.4f}"
        )

torch_training_time = time.perf_counter() - torch_start

print(f"PyTorch training time: {torch_training_time:.4f} seconds")
print(f"Our engine training time: {training_time:.4f} seconds")
print("PyTorch training completed successfully!")

# =====================================
# STEP 12: MEDIAN OF 5 TRAINING RUNS
# =====================================

import statistics

print("\nLab 5 - Median Training Benchmark")

our_times = []
pytorch_times = []

for run in range(5):
    # Our Value engine
    random.seed(42)
    model = TinyNetwork()

    start = time.perf_counter()

    for epoch in range(30):
        for inputs, target in zip(X_moons, y_moons):
            for p in model.parameters():
                p.grad = 0.0

            prediction = model(inputs)
            loss = bce_loss(prediction, target)
            loss.backward()

            for p in model.parameters():
                p.data -= 0.1 * p.grad

    our_times.append(time.perf_counter() - start)

    # PyTorch model
    torch.manual_seed(42)

    model_torch = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    optimizer_torch = torch.optim.SGD(
        model_torch.parameters(), lr=0.1
    )

    start = time.perf_counter()

    for epoch in range(30):
        for i in range(len(X_torch)):
            optimizer_torch.zero_grad()

            prediction = model_torch(X_torch[i:i+1])
            loss = criterion(prediction, y_torch[i:i+1])

            loss.backward()
            optimizer_torch.step()

    pytorch_times.append(time.perf_counter() - start)

    print(
        f"Run {run + 1}/5 | "
        f"Our engine: {our_times[-1]:.4f}s | "
        f"PyTorch: {pytorch_times[-1]:.4f}s"
    )

our_median = statistics.median(our_times)
pytorch_median = statistics.median(pytorch_times)

print("\nMedian results:")
print(f"Our engine median: {our_median:.4f} seconds")
print(f"PyTorch median: {pytorch_median:.4f} seconds")
print("Median benchmark completed successfully!")

# =====================================
# STEP 13: EAGER VS JIT TRACED BENCHMARK
# =====================================

print("\nLab 5 - Eager vs JIT Traced Benchmark")

# Use the trained PyTorch model
torch_network.eval()

example_input = torch.randn(128, 2)

# Create an optimized traced model
traced_network = torch.jit.trace(torch_network, example_input)

print("\nTraced computation graph:")
print(traced_network.inlined_graph)

# Warm-up before timing
with torch.no_grad():
    for _ in range(100):
        torch_network(example_input)
        traced_network(example_input)

eager_times = []
traced_times = []

with torch.no_grad():
    for run in range(5):
        start = time.perf_counter()

        for _ in range(1000):
            torch_network(example_input)

        eager_times.append(time.perf_counter() - start)

        start = time.perf_counter()

        for _ in range(1000):
            traced_network(example_input)

        traced_times.append(time.perf_counter() - start)

eager_median = statistics.median(eager_times)
traced_median = statistics.median(traced_times)

print(f"\nEager median: {eager_median:.6f} seconds")
print(f"JIT traced median: {traced_median:.6f} seconds")
print(f"Speedup (eager / traced): {eager_median / traced_median:.2f}x")

print("Eager vs JIT benchmark completed successfully!")

# =====================================
# STEP 14: TENSOR PROPERTIES
# =====================================

print("\nLab 5 - Tensor Properties")

# Create a tensor
tensor = torch.randn(128, 2)

print("Original tensor:")
print("Shape:", tensor.shape)
print("Dtype:", tensor.dtype)
print("Contiguous:", tensor.is_contiguous())

# Create a non-contiguous view
tensor_transposed = tensor.transpose(0, 1)

print("\nTransposed tensor:")
print("Shape:", tensor_transposed.shape)
print("Dtype:", tensor_transposed.dtype)
print("Contiguous:", tensor_transposed.is_contiguous())

# Convert back to contiguous memory
tensor_contiguous = tensor_transposed.contiguous()

print("\nAfter .contiguous():")
print("Shape:", tensor_contiguous.shape)
print("Dtype:", tensor_contiguous.dtype)
print("Contiguous:", tensor_contiguous.is_contiguous())

print("Tensor properties checked successfully!")

# =====================================
# STEP 15: CONTIGUITY BENCHMARK
# =====================================

print("\nLab 5 - Contiguity Benchmark")

torch.manual_seed(42)

# Both tensors have the same shape and values.
base = torch.randn(512, 512)
non_contiguous = base.transpose(0, 1)
contiguous = non_contiguous.contiguous()

print("Shape:", contiguous.shape)
print("Dtype:", contiguous.dtype)
print("Contiguous tensor:", contiguous.is_contiguous())
print("Non-contiguous tensor:", non_contiguous.is_contiguous())

# Warm-up
with torch.no_grad():
    for _ in range(50):
        torch.relu(contiguous)
        torch.relu(non_contiguous)

contiguous_times = []
non_contiguous_times = []

with torch.no_grad():
    for run in range(5):
        start = time.perf_counter()

        for _ in range(200):
            torch.relu(contiguous)

        contiguous_times.append(time.perf_counter() - start)

        start = time.perf_counter()

        for _ in range(200):
            torch.relu(non_contiguous)

        non_contiguous_times.append(time.perf_counter() - start)

contiguous_median = statistics.median(contiguous_times)
non_contiguous_median = statistics.median(non_contiguous_times)

print(f"Contiguous median: {contiguous_median:.6f} seconds")
print(f"Non-contiguous median: {non_contiguous_median:.6f} seconds")

print("Contiguity benchmark completed successfully!")

# =====================================
# STEP 16: SAVE LAB 5 RESULTS
# =====================================

from pathlib import Path

print("\nLab 5 - Saving Results")

results_folder = Path(__file__).resolve().parent.parent / "results"
results_folder.mkdir(parents=True, exist_ok=True)

results_file = results_folder / "benchmark_results.txt"

with open(results_file, "w", encoding="utf-8") as f:
    f.write("LAB 5 - BENCHMARK RESULTS\n")
    f.write("=========================\n\n")

    f.write("Gradient verification:\n")
    f.write(f"PyTorch max difference: {max_difference:.10f}\n")
    f.write(f"Numerical max difference: {max_numerical_difference:.10f}\n\n")

    f.write("Training benchmark (median of 5 runs):\n")
    f.write(f"Custom engine: {our_median:.6f} seconds\n")
    f.write(f"PyTorch: {pytorch_median:.6f} seconds\n\n")

    f.write("Eager vs JIT traced (median of 5 runs):\n")
    f.write(f"Eager: {eager_median:.6f} seconds\n")
    f.write(f"JIT traced: {traced_median:.6f} seconds\n")
    f.write(f"Speedup: {eager_median / traced_median:.2f}x\n\n")

    f.write("Tensor contiguity benchmark:\n")
    f.write(f"Contiguous: {contiguous_median:.6f} seconds\n")
    f.write(f"Non-contiguous: {non_contiguous_median:.6f} seconds\n")

print("Results saved to:", results_file)
print("Lab 5 results saved successfully!")
