import yaml

with open("lab.04/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully!")
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# =====================================
# STEP 1: LOAD DATASET
# =====================================

print("Step 1: Loading UCI Adult Dataset")


adult = fetch_openml(
    name=config["dataset"]["name"],
    version=config["dataset"]["version"],
    as_frame=True
)


raw_data = adult.frame.copy(deep=True)

print("Rows:", raw_data.shape[0])
print("Columns:", raw_data.shape[1])


# =====================================
# STEP 2: DATA VALIDATION
# =====================================

print("\nStep 2: Data Validation")

numeric_columns = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

target_column = "class"

expected_columns = (
    numeric_columns
    + categorical_columns
    + [target_column]
)

missing_columns = [
    col for col in expected_columns
    if col not in raw_data.columns
]

print("Missing columns:", missing_columns)

print("\nColumn data types:")
print(raw_data.dtypes)

ranges = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356)
}

print("\nRange violations:")

for column, (minimum, maximum) in ranges.items():
    values = pd.to_numeric(
        raw_data[column],
        errors="coerce"
    )

    violations = (
        (values < minimum)
        | (values > maximum)
    ).sum()

    print(f"{column}: {violations}")

print("\nMissing values:")

for column in raw_data.columns:
    missing = (
        raw_data[column].isna()
        | raw_data[column]
        .astype(str)
        .str.strip()
        .eq("?")
    ).sum()

    print(f"{column}: {missing}")

print(
    "\nExact duplicate rows:",
    raw_data.duplicated().sum()
)

print("Validation completed!")


# =====================================
# STEP 3: TRAIN / TEST SPLIT
# =====================================

print("\nStep 3: Train/Test Split")

X = raw_data.drop(
    columns=[target_column]
).copy()

y = raw_data[target_column].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["split"]["test_size"],
    random_state=config["split"]["random_state"],
    stratify=y
)


print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

print("Split completed successfully!")


# =====================================
# STEP 4: REMOVE TRAIN DUPLICATES
# =====================================

print("\nStep 4: Remove Training Duplicates")

train_combined = X_train.copy()
train_combined[target_column] = y_train

duplicates_before = (
    train_combined.duplicated().sum()
)

train_combined = (
    train_combined.drop_duplicates()
)

X_train_clean = train_combined.drop(
    columns=[target_column]
)

y_train_clean = (
    train_combined[target_column]
)

print(
    "Training duplicates removed:",
    duplicates_before
)

print(
    "Training rows after cleaning:",
    len(X_train_clean)
)

print(
    "Test rows unchanged:",
    len(X_test)
)


# =====================================
# STEP 5: PREPROCESSING
# =====================================

print("\nStep 5: Preprocessing")


class PercentileClipper(
    BaseEstimator,
    TransformerMixin
):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        self.lower_ = np.percentile(
            X, 1, axis=0
        )

        self.upper_ = np.percentile(
            X, 99, axis=0
        )

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        return np.clip(
            X,
            self.lower_,
            self.upper_
        )


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "clipper",
        PercentileClipper()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


def prepare_missing_values(data):
    result = data.copy()

    for column in categorical_columns:
        result[column] = (
            result[column]
            .astype("object")
            .replace("?", np.nan)
        )

    return result


X_train_prepared = prepare_missing_values(
    X_train_clean
)

X_test_prepared = prepare_missing_values(
    X_test
)

X_train_processed = (
    preprocessor.fit_transform(
        X_train_prepared
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test_prepared
    )
)

print(
    "Processed train shape:",
    X_train_processed.shape
)

print(
    "Processed test shape:",
    X_test_processed.shape
)


# =====================================
# STEP 6: SAVE PREPROCESSOR
# =====================================

print("\nStep 6: Save Preprocessor")

results_dir = Path(config["output"]["directory"])


results_dir.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    preprocessor,
    results_dir / "preprocessor.joblib"
)

print("Preprocessor saved successfully!")
print("\nLab 4 Steps 1-6 completed!")

# =====================================
# STEP 7: FEATURE ENGINEERING
# AND LOGISTIC REGRESSION
# =====================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import csv

print("\nStep 7: Feature Engineering")


def add_features(data):
    result = data.copy()

    result["capital_net"] = (
        result["capital-gain"]
        - result["capital-loss"]
    )

    result["hours_per_age"] = (
        result["hours-per-week"]
        / (result["age"] - 16)
    )

    return result


def evaluate_model(train_features, test_features):
    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )

    model.fit(train_features, y_train_clean)

    predictions = model.predict(test_features)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=">50K"
    )

    return accuracy, f1


# Baseline: without engineered features
baseline_accuracy, baseline_f1 = evaluate_model(
    X_train_processed,
    X_test_processed
)

print("\nBaseline results:")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"F1 Score: {baseline_f1:.4f}")


# Add two new features
X_train_engineered = add_features(
    X_train_prepared
)

X_test_engineered = add_features(
    X_test_prepared
)

engineered_numeric_columns = numeric_columns + [
    "capital_net",
    "hours_per_age"
]

engineered_preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "clipper",
                PercentileClipper()
            )
        ]),
        engineered_numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])

X_train_engineered_processed = (
    engineered_preprocessor.fit_transform(
        X_train_engineered
    )
)

X_test_engineered_processed = (
    engineered_preprocessor.transform(
        X_test_engineered
    )
)

engineered_accuracy, engineered_f1 = evaluate_model(
    X_train_engineered_processed,
    X_test_engineered_processed
)

print("\nWith engineered features:")
print(f"Accuracy: {engineered_accuracy:.4f}")
print(f"F1 Score: {engineered_f1:.4f}")


# Save comparison results
comparison_results = [
    {
        "model": "Baseline",
        "accuracy": baseline_accuracy,
        "f1": baseline_f1
    },
    {
        "model": "Engineered Features",
        "accuracy": engineered_accuracy,
        "f1": engineered_f1
    }
]

with open(
    results_dir / "model_comparison.csv",
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model", "accuracy", "f1"]
    )
    writer.writeheader()
    writer.writerows(comparison_results)

print("\nModel comparison saved!")

# =====================================
# STEP 8: LABEL QUALITY AUDIT
# =====================================

print("\nStep 8: Label Quality Audit")

# Find feature-identical rows with conflicting labels
feature_columns = [
    col for col in raw_data.columns
    if col != target_column
]

label_counts = raw_data.groupby(
    feature_columns,
    dropna=False,
    observed=True
)[target_column].nunique(dropna=False)

conflicting_groups = label_counts[
    label_counts > 1
]

print(
    "Duplicate feature groups with conflicting labels:",
    len(conflicting_groups)
)

# Count all rows belonging to conflicting groups
conflicting_rows = raw_data.merge(
    conflicting_groups.reset_index()[
        feature_columns
    ],
    on=feature_columns,
    how="inner"
)

print(
    "Rows in conflicting groups:",
    len(conflicting_rows)
)

# Save conflicting rows for inspection
conflicting_rows.to_csv(
    results_dir / "label_conflicts.csv",
    index=False
)

# Draw 50 reproducible random rows
audit_sample = raw_data.sample(
    n=50,
    random_state=42
).copy()

# Check numeric ranges in sampled rows
audit_sample["range_check"] = "PASS"

for column, (minimum, maximum) in ranges.items():
    values = pd.to_numeric(
        audit_sample[column],
        errors="coerce"
    )

    invalid = (
        values.isna()
        | (values < minimum)
        | (values > maximum)
    )

    audit_sample.loc[
        invalid, "range_check"
    ] = "REVIEW"

# Flag unexpected target labels
valid_labels = {"<=50K", ">50K"}

invalid_labels = ~audit_sample[
    target_column
].astype(str).isin(valid_labels)

audit_sample.loc[
    invalid_labels, "range_check"
] = "REVIEW"


# Preserve completed manual audit results
audit_file = results_dir / "label_audit_50.csv"

audit_sample["manual_review"] = "PENDING"

if audit_file.exists():
    previous_audit = pd.read_csv(audit_file)

    # Match the exact sampled rows before restoring reviews
    original_columns = raw_data.columns.tolist()

    if all(col in previous_audit.columns for col in original_columns):
      
        old_rows = (
            previous_audit[original_columns]
            .astype("object")
            .fillna("<MISSING>")
            .astype(str)
        )

      
        new_rows = (
            audit_sample[original_columns]
            .astype("object")
            .fillna("<MISSING>")
            .astype(str)
        )

        if (
            old_rows.reset_index(drop=True).equals(
                new_rows.reset_index(drop=True)
            )
            and previous_audit["manual_review"].eq("PASS").all()
        ):
            audit_sample["manual_review"] = "PASS"

audit_sample.to_csv(
    audit_file,
    index=False
)


audit_sample.to_csv(
    results_dir / "label_audit_50.csv",
    index=False
)

print(
    "Random rows selected for audit:",
    len(audit_sample)
)

print(
    "Rows flagged for review:",
    (audit_sample["range_check"] == "REVIEW").sum()
)

print("Manual review status: PENDING")
print("Label quality audit files saved!")

# =====================================
# STEP 9: SHA256 DATASET HASH
# =====================================

import hashlib

print("\nStep 9: Dataset SHA256 Hash")

# Hash the processed training and test data,
# including their corresponding labels.
def update_hash(hasher, features, labels):
    if hasattr(features, "toarray"):
        features = features.toarray()

    features = np.asarray(features, dtype=np.float64)

    hasher.update(
        np.ascontiguousarray(features).tobytes()
    )

    label_text = "\n".join(
        labels.astype(str).tolist()
    )

    hasher.update(label_text.encode("utf-8"))


hasher = hashlib.sha256()

update_hash(
    hasher,
    X_train_engineered_processed,
    y_train_clean
)

update_hash(
    hasher,
    X_test_engineered_processed,
    y_test
)

dataset_hash = hasher.hexdigest()

print("Processed dataset SHA256:", dataset_hash)

with open(
    results_dir / "dataset_hash.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(dataset_hash + "\n")

print("SHA256 hash saved successfully!")

# =====================================
# STEP 10: SAVE VALIDATION REPORT
# =====================================

import json

print("\nStep 10: Saving Validation Report")

validation_report = {
    "dataset": "UCI Adult",
    "total_rows": int(len(raw_data)),
    "total_columns": int(len(raw_data.columns)),
    "missing_expected_columns": missing_columns,
    "column_types": {
        col: str(raw_data[col].dtype)
        for col in raw_data.columns
    },
    "missing_values": {},
    "range_violations": {},
    "exact_duplicate_rows": int(
        raw_data.duplicated().sum()
    ),
    "training_duplicates_removed": int(
        duplicates_before
    ),
    "conflicting_label_groups": int(
        len(conflicting_groups)
    ),
    "conflicting_label_rows": int(
        len(conflicting_rows)
    ),
    "manual_audit_sample_size": int(
        len(audit_sample)
    ),
   
"manual_audit_status": (
    "COMPLETED"
    if audit_sample["manual_review"].eq("PASS").all()
    else "PENDING"
)

}

for column in raw_data.columns:
    missing = (
        raw_data[column].isna()
        | raw_data[column]
        .astype(str)
        .str.strip()
        .eq("?")
    ).sum()

    validation_report["missing_values"][column] = (
        int(missing)
    )

for column, (minimum, maximum) in ranges.items():
    values = pd.to_numeric(
        raw_data[column],
        errors="coerce"
    )

    violations = (
        (values < minimum)
        | (values > maximum)
    ).sum()

    validation_report["range_violations"][column] = (
        int(violations)
    )

with open(
    results_dir / "validation_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        validation_report,
        file,
        indent=4
    )

print("Validation report saved successfully!")
