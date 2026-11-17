# Individual Project Proposal

**Project Title:** Efficient Fashion Image Classification Using a Lightweight CNN

**Student Name:** [Your Full Name]  
**Student ID:** [Your Student ID]

## 1. Problem Statement

The goal of this project is to develop a lightweight deep learning model that accurately classifies clothing images while remaining suitable for deployment on resource-constrained devices. The project will investigate how model optimization affects classification accuracy, inference latency, and model size.

## 2. Dataset

The project will use **FashionMNIST**, which contains 60,000 training images and 10,000 test images across 10 clothing categories. Each image is grayscale with a resolution of 28 × 28 pixels.

## 3. Model and Success Metrics

The baseline will be the CNN developed in Lab 3, which achieved approximately **89.49% test accuracy**.

The project's targets are:

- **Test accuracy:** At least 89%.
- **Inference latency:** No more than 50 ms per image.
- **Model size:** No more than 5 MB.

Accuracy will be evaluated on the FashionMNIST test set. Inference latency and model size will be measured on the selected deployment environment.

## 4. Deployment Constraints

The final model should support CPU-based inference without requiring a dedicated GPU. It must satisfy the latency and model-size constraints while maintaining acceptable classification accuracy.

## 5. Planned Optimization

Three optimization techniques will be investigated:

- **Quantization:** Reduce numerical precision to decrease model size and potentially improve inference speed.
- **Pruning:** Remove less important model parameters to reduce computational complexity.
- **Knowledge distillation:** Train a smaller student model using predictions from a larger teacher model.

The optimized models will be compared against the baseline CNN using accuracy, latency, and model size.

## 6. Expected Outcome

The expected result is a compact clothing-classification model that achieves competitive accuracy while meeting the deployment requirements. The project will document the trade-offs between predictive performance and computational efficiency.