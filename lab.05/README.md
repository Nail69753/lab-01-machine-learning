
# Lab 5 — Frameworks and Automatic Differentiation

## Objective
Implement a simple automatic differentiation engine and compare its results and performance with PyTorch.

## 1. Custom Autograd Engine
Implemented a `Value` class supporting:
- Addition and multiplication
- Power operations
- ReLU activation
- Reverse-mode automatic differentiation
- Topological sorting during backpropagation
- Gradient accumulation

## 2. Gradient Verification
Tested the expression:

`((a * b + c) ** 2).relu()`

The custom engine gradients were compared against:
- PyTorch autograd
- Numerical finite differences

Both checks passed with a maximum absolute difference below `1e-6`.

## 3. Neural Network Training
Created a 2 → 8 → 1 neural network using:
- ReLU hidden activation
- Sigmoid output activation
- Binary cross-entropy loss
- The `make_moons` dataset

Trained both the custom engine and a PyTorch model.

Training performance was measured using the median of five runs.

## 4. Eager vs JIT Traced
Compared standard PyTorch eager execution against `torch.jit.trace`.

Both models were warmed up before benchmarking.

JIT tracing can help when the model executes many small operations repeatedly, because it can reduce Python execution overhead. However, tracing does not guarantee a speedup for every model.

## 5. Tensor Properties
Inspected:
- Tensor shape
- Data type
- Memory contiguity

Compared contiguous and non-contiguous tensors using a ReLU benchmark.

## Results
Measured timings and gradient differences are saved in:

`results/benchmark_results.txt`

## Run
From the repository root:

```powershell
python "lab.05\src\main.py"
```
