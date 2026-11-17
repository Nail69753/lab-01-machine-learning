# Individual Project Proposal

**Project Title:** Car Brand and Model Recognition Using Deep Learning  
**Student Name:** Nail Murghuzlu  
**Student ID:** 69753

## 1. Problem

In this project, I want to create an AI model that can recognize the brand and model of a car from a picture. Some cars look very similar, so identifying the correct model is not always easy. I will use deep learning to solve this problem.

The main idea is to build a model that gives good predictions but does not require too much memory or processing power.

## 2. Dataset

I plan to use the **Stanford Cars Dataset**. It contains 16,185 car images from 196 different categories.

The pictures include different brands, models, and production years. Before training, I will resize the images and normalize their pixel values. I may also use data augmentation to improve the results.

## 3. Model and Success Metric

I will start with a pretrained CNN model, possibly MobileNetV3, and adapt it for car classification.

My target is to achieve **at least 80% test accuracy**. I will compare the predicted car categories with the correct labels to measure performance.

I will also check how fast the model works and how much storage it needs.

## 4. Deployment Constraints

The final model should be small enough to run on a normal computer without a powerful GPU.

The requirements are:

- **Inference latency:** 50 ms or less per image.
- **Model size:** 5 MB or less.

I will test the model on a CPU and measure its average prediction time.

## 5. Planned Optimization

I plan to use two optimization methods.

**Knowledge distillation** will help train a smaller model using information from a larger model.

**Quantization** will reduce the precision of model weights to make the model smaller and possibly faster.

After optimization, I will compare the new model with the original one using accuracy, size, and latency.

## 6. Expected Result

I expect to develop a small car recognition model that can identify different vehicle brands and models with reasonable accuracy.

The main challenge will be keeping good accuracy while reducing model size and prediction time.

This project will help me understand how deep learning models can be optimized for practical applications.