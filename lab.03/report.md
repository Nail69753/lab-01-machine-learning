# Lab 3 — Architectures and Their Costs

## 1. Objective

The objective of this laboratory is to implement and compare three neural network architectures using PyTorch: a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and an Attention-based Transformer model.

The comparison focuses on classification accuracy, parameter count, computational cost (FLOPs), and inference latency.

## 2. Dataset

The Fashion-MNIST dataset was used for image classification. Each image is a 28 × 28 grayscale image belonging to one of ten clothing categories.

The dataset was divided into:

- Training: 55,000 images
- Validation: 5,000 images
- Test: 10,000 images

The images were converted to tensors and normalized using a mean of 0.2860 and a standard deviation of 0.3530. The batch size was 128.

## 3. Model Architectures

### MLP

The MLP consists of a flattened input layer followed by fully connected layers with 256 and 128 hidden units, ReLU activations, and a final 10-class output layer.

### CNN

The CNN uses two convolutional layers with 16 and 32 output channels, ReLU activations, max-pooling layers, and a final fully connected classification layer.

### Attention

The Attention model divides each image into 16 patches of size 7 × 7. Each patch is projected to a 64-dimensional embedding. Two Transformer encoder layers with four attention heads are applied, followed by mean pooling and a classification layer.

## 4. Parameter Counts

The parameter counts obtained from the model implementations are:

| Model | Parameters |
|---|---:|
| MLP | 235,146 |
CNN | 20,490
| Attention | 70,794 |

The prescribed architectures do not all fall within the assignment's suggested range of 100,000–150,000 parameters. The exact layer configurations were preserved for consistency with the provided specifications.

## 5. Computational Cost


### THOP Profiling Results

| Model | THOP MACs | Approximate FLOPs | THOP Parameters |
|---|---:|---:|---:|
| MLP | 234,752 | 469,504 | 235,146 |
| CNN | 1,031,744 | 2,063,488 | 20,490 |
| Attention | 50,816 | 101,632 | 3,850 |

Approximate FLOPs were calculated as two times the
MAC count reported by THOP.

For the MLP, the theoretical FLOPs of the first linear
layer are 401,408, while THOP reports approximately
469,504 FLOPs for the complete MLP.

For the CNN, the theoretical FLOPs of the first
convolutional layer are 225,792, while THOP reports
approximately 2,063,488 FLOPs for the complete CNN.

These values are not directly equivalent because the
hand calculations describe individual layers, while
THOP profiles the complete model.

THOP significantly underestimates the Attention model
because it does not fully account for Transformer
operations, including attention matrix multiplications.
Its reported parameter count of 3,850 is also incomplete.

Therefore, the Attention model's THOP results should
not be interpreted as its actual total computational cost.


## 6. Training and Evaluation


| Model | Test Accuracy |
|---|---:|
| MLP | 87.40% |
| CNN | 89.49% |
| Attention | 84.15% |

The CNN achieved the highest test accuracy at 89.49%,
followed by the MLP at 87.40% and the Attention model
at 84.15%.


## 7. Inference Latency


| Model | Batch 1 Median (ms) | Batch 128 Median (ms) |
|---|---:|---:|
| MLP | 0.1977 | 0.7035 |
| CNN | 0.4429 | 9.6692 |
| Attention | 0.9840 | 6.4060 |

The MLP had the lowest inference latency for both batch
sizes. The CNN achieved the highest accuracy but required
more inference time. The Attention model was slower than
the MLP and had the lowest test accuracy.

These results demonstrate that parameter count alone
does not determine inference latency. Different
architectures have different computation and memory
access patterns.

## 8. Accuracy and Cost Trade-offs

Based on the measured results, the CNN offers the best
classification accuracy (89.49%), while the MLP provides
the lowest inference latency.

The CNN is a strong choice when accuracy is the main
priority. The MLP is more suitable when inference speed
is critical.

The Attention model did not outperform the other
architectures under the five-epoch training setup.


- `results/accuracy_vs_parameters.png`
- `results/accuracy_vs_latency.png`

The CNN uses convolutional filters to share parameters across spatial locations, allowing efficient feature extraction with relatively few learned weights.

The MLP uses dense matrix multiplications and does not explicitly exploit the spatial structure of images.

The Attention model can represent relationships between image patches, but its attention operations introduce additional computational and memory-access costs.

The final cost-effectiveness comparison depends on the measured test accuracies and inference latencies.

## 9. Conclusion

This laboratory demonstrates that neural network architectures can differ significantly in their parameter counts, computational operations, and inference performance.

The experiments provide a basis for comparing MLP, CNN, and Attention models under a common dataset and training protocol. The final assessment should consider accuracy, latency, and computational cost together rather than relying on parameter count alone.