# Lab 2: A Neural Network from Scratch

## 1. Objective

The objective of this lab is to implement and evaluate a neural network from scratch using NumPy, including forward propagation, cross-entropy loss, backpropagation, gradient checking, training, and performance measurements.

## 2. Dataset

The experiment uses the `load_digits` dataset from scikit-learn.

- Input features: 64
- Output classes: 10
- Training samples: 1257
- Testing samples: 540
- Train/test split: 70% / 30%
- Random state: 42

Pixel values were normalized by dividing by 16, and labels were one-hot encoded.

## 3. Neural Network Architecture

The network consists of:

- Input layer: 64 features
- First hidden layer: W neurons with ReLU
- Second hidden layer: W neurons with ReLU
- Output layer: 10 neurons with Softmax

Two hidden-layer widths were tested: W = 16 and W = 64.

Weights were initialized using a normal distribution with mean 0 and standard deviation 0.01. Biases were initialized to zero.

## 4. Forward Propagation and Loss

Forward propagation was implemented using NumPy matrix operations.

Softmax was used to convert output scores into probabilities. The output shape and probability sums were verified.

Cross-entropy was used as the loss function.

The initial loss was approximately 2.3026, which is close to ln(10), as expected for an untrained ten-class classifier.

## 5. Backpropagation and Gradient Checking

Backpropagation was implemented manually using the chain rule.

Numerical gradient checking was performed on 10 randomly selected weights.

Numerical gradient checking was performed using a perturbation of 1e-5.

All 10 out of 10 gradient checks passed with a relative error below 1e-6, confirming agreement between the analytical and numerical gradients.


## 6. Training Experiments

The network was trained using mini-batch gradient descent.

Training configuration:

- Epochs: 50
- Batch size: 32
- Hidden-layer widths: 16 and 64
- Learning rates: 0.001, 0.01 and 0.1

All six configurations were tested.

For W = 64 and learning rate = 0.1:

- Final training loss: 0.1566
- Final test accuracy: 93.70%

The loss curves for all configurations are saved in `results/loss_curves.png`.

Complete experimental results are available in `results/experiment_results.csv`.

## 7. Forward and Backward Timing

Forward propagation and backpropagation were measured separately using the median of 100 runs with a batch size of 32.

| Width | Forward Median (ms) | Backward Median (ms) |
|---|---:|---:|
| 16 | 0.0891 | 0.0929 |
| 64 | 0.0628 | 0.0835 |

Backward propagation was slightly slower than forward propagation in both measurements.

This is consistent with backward propagation requiring additional matrix multiplications to compute gradients. The measurements may also be affected