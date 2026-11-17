# Individual Project Proposal

**Project Title:** Car Brand and Model Recognition Using Deep Learning

**Student Name:** Nail Murghuzlu  
**Student ID:** 69753  
**Course Assignment:** Lab 6 — Task 8

### 1. Problem Statement
The goal of this project is to develop a deep learning system that automatically identifies a car's brand and model from an image. The system will use image classification to distinguish between different vehicle models and provide predictions efficiently.

### 2. Dataset
The project will use the **Stanford Cars Dataset**, which contains 16,185 images across 196 car categories. The images include different vehicle makes, models, and production years. The dataset will be divided into training and testing sets, with image preprocessing and data augmentation applied during training.

### 3. Success Metrics
The primary evaluation metric will be classification accuracy on the test dataset. The initial target is **at least 80% test accuracy**. Model size and inference latency will also be measured to evaluate deployment suitability.

### 4. Deployment Constraints
The optimized model must satisfy the following requirements:

- **Inference latency:** ≤ 50 milliseconds per image
- **Model size:** ≤ 5 MB

Performance will be evaluated on CPU hardware using a consistent input size and batch size of one.

### 5. Planned Optimization
The project will begin with a pretrained convolutional neural network adapted to car classification. Two optimization methods are planned:

- **Knowledge distillation:** Transfer knowledge from a larger teacher model to a smaller student model.
- **Quantization:** Reduce numerical precision to decrease model size and potentially improve inference speed.

The final model will be compared with the baseline in terms of accuracy, latency, and storage requirements.

### 6. Expected Outcome
The expected outcome is a compact car recognition model that balances classification accuracy with efficient deployment. The project will document the trade-offs between prediction quality, model size, and inference speed.