# Lab 1 — Machine Learning Baseline

## Project Description

This project compares two machine learning classification models using the Breast Cancer Wisconsin Diagnostic Dataset.

The models are:
- Logistic Regression
- Random Forest

## Requirements

- Python 3.11.8
- Dependencies listed in `requirements.txt`

## Installation

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Install dependencies using the virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the Experiment

Run the baseline experiment:

```powershell
.\.venv\Scripts\python.exe lab.01\src\main.py
```

Print package versions:

```powershell
.\.venv\Scripts\python.exe lab.01\src\print_versions.py
```

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 94.15% |
| Random Forest | 93.57% |

Detailed performance measurements are stored in `lab.01/results/measurements.csv`.

## Project Structure

```text
repo/
├── data/
├── lab.01/
│   ├── src/
│   │   ├── main.py
│   │   └── print_versions.py
│   ├── results/
│   │   ├── baseline_accuracy.csv
│   │   ├── measurements.csv
│   │   ├── logistic_regression.joblib
│   │   ├── random_forest.joblib
│   │   └── versions.txt
│   └── report.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes

PyArrow 15.0.2 was used instead of 16.1.0 due to a dependency conflict with MLflow 2.14.1.

## Conclusion

Logistic Regression achieved slightly higher accuracy and substantially lower inference latency and serialized model size, while Random Forest trained faster in this experiment.