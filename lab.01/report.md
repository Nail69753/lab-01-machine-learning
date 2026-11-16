# Lab 1 — Machine Learning Baseline and Performance Measurement

## 1. Objective

The objective of this laboratory exercise is to establish baseline performance measurements for two machine learning algorithms: Logistic Regression and Random Forest.

The experiment evaluates classification accuracy, training time, inference latency, serialized model size, and process memory usage.

## 2. Experimental Environment

- **Python version:** 3.11.8
- **Operating system:** Windows
- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset (scikit-learn)
- **Training split:** 70%
- **Testing split:** 30%
- **Random seed:** 42

The experiment uses NumPy 1.26.4, scikit-learn 1.5.0, PyTorch 2.3.1, and other dependencies documented in `results/versions.txt`.

PyArrow 15.0.2 was used instead of the initially specified 16.1.0 because of a dependency conflict with MLflow 2.14.1.

## 3. Methodology

Two classification models were evaluated:

1. Logistic Regression with a maximum of 1,000 iterations.
2. Random Forest with 100 estimators.

The dataset was divided using stratified sampling with a fixed random state.

Each model was warmed up before performance measurement. Training was repeated five times, and median training time was recorded.

Inference latency was measured over 100 predictions using a single test sample. The median inference time was reported.

Model sizes were measured from serialized Joblib files.

Process memory was sampled using psutil. The recorded values represent the maximum sampled process resident memory, rather than an exact peak memory measurement.

## 4. Experimental Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 94.15% | 93.57% |
| Training time (ms) | 849.8851 | 260.4682 |
| Inference time (ms) | 0.2551 | 6.5673 |
| Model size (KB) | 1.0303 | 284.0713 |
| Sampled memory (MB) | 229.8633 | 230.9531 |

## 5. Discussion

Logistic Regression achieved the highest classification accuracy at 94.15%, compared with 93.57% for Random Forest.

Random Forest demonstrated a shorter training time, completing training in approximately 260.47 ms compared with 849.89 ms for Logistic Regression.

However, Logistic Regression provided substantially lower inference latency and required significantly less serialized storage.

The sampled process memory values were similar for both models. These values include memory used by the Python process and loaded libraries, so they should not be interpreted as the models' individual memory requirements.

The results reflect this particular dataset, experimental configuration, and execution environment.

## 6. Conclusion

Both models successfully completed the baseline classification experiment.

Logistic Regression offered the best combination of classification accuracy, inference speed, and serialized model size in this experiment.

Random Forest demonstrated faster training but required more storage and inference time.

Overall, Logistic Regression is the preferable baseline for this experiment when inference efficiency and model size are priorities.

## 7. Generated Artifacts

The experiment produced the following files:

- `results/baseline_accuracy.csv`
- `results/measurements.csv`
- `results/logistic_regression.joblib`
- `results/random_forest.joblib`
- `results/versions.txt`

These artifacts document the measured results, trained models, and software environment.