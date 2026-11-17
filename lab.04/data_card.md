# Data Card — UCI Adult Dataset

## 1. Dataset Overview

- **Dataset:** UCI Adult (Census Income)
- **Source:** UCI Machine Learning Repository, accessed through OpenML
- **OpenML version:** 2
- **Rows:** 48,842
- **Columns:** 15 (14 input features and 1 target)
- **Task:** Binary classification

## 2. Problem Definition

The objective is to predict whether an individual's annual income exceeds $50,000 using demographic, education, employment, and financial information.

The target column is `class`, with two possible labels: `<=50K` and `>50K`.

Model performance is evaluated using accuracy and F1 score.

## 3. Data Processing

The dataset is divided into 70% training data and 30% test data using a stratified split.

The following operations are performed:

- Exact duplicate rows are removed from the training set only.
- Missing numeric values are imputed using training medians.
- Missing categorical values are imputed using training modes.
- Numeric features are clipped using the 1st and 99th percentiles calculated from training data.
- Categorical features are one-hot encoded.
- Two derived features are created: `capital_net` and `hours_per_age`.

The test dataset is not used to fit preprocessing transformations.

## 4. Data Quality

Validation checks include expected columns, missing values, numeric ranges, and duplicate rows.

A label-quality audit checks for identical feature records with conflicting target labels.

A reproducible sample of 50 records is selected for manual review against the dataset documentation. The manual review must be completed and recorded separately.

## 5. Bias and Fairness

The dataset contains demographic attributes, including race and sex, and reflects historical socioeconomic patterns.

Models trained on this dataset may reproduce or amplify existing inequalities. Accuracy and F1 score alone do not establish fairness across demographic groups.

## 6. Privacy

The dataset contains individual-level census-derived records. Even though it is publicly distributed, records should be handled responsibly.

The model should not be used to make consequential decisions about individuals without additional validation, fairness assessment, and appropriate oversight.

## 7. License and Usage

The dataset originates from the UCI Machine Learning Repository and is accessed through OpenML.

The applicable dataset license and attribution requirements should be verified from the official dataset pages before redistribution or commercial use.

Sources:

- https://archive.ics.uci.edu/dataset/2/adult
- https://www.openml.org/d/1590

## 8. Reproducibility

The project uses `random_state=42` for reproducible splitting and sampling.

Configuration settings are documented in `config.yaml`.

The fitted preprocessing pipeline is saved to `results/preprocessor.joblib`, and a SHA256 digest of processed data is saved to `results/dataset_hash.txt`.