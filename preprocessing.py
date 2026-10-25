from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"
TARGET_COLUMN = "loan_status"

FEATURE_ORDER = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def build_category_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        column: {value: index for index, value in enumerate(sorted(df[column].dropna().unique()))}
        for column in CATEGORICAL_FEATURES
    }


def normalize_numeric_features(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for column in NUMERIC_FEATURES:
        min_value = float(df[column].min())
        max_value = float(df[column].max())
        params[column] = {"min": min_value, "max": max_value}

        denominator = max_value - min_value
        df[column] = 0.0 if denominator == 0 else (df[column] - min_value) / denominator

    return params


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    initial_size = len(df)
    df = df.dropna().copy()
    removed_rows = initial_size - len(df)
    print(f"Data cleaning complete: removed {removed_rows} rows with missing values.")

    category_maps = build_category_maps(df)
    for column, mapping in category_maps.items():
        df[column] = df[column].map(mapping)

    numeric_params = normalize_numeric_features(df)

    params = {
        **numeric_params,
        "category_maps": category_maps,
        "feature_order": FEATURE_ORDER,
        "target_column": TARGET_COLUMN,
    }
    with (BASE_DIR / "params.json").open("w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=False, indent=2)

    x = df[FEATURE_ORDER].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = next(skf.split(x, y))

    np.save(BASE_DIR / "X_train.npy", x[train_index])
    np.save(BASE_DIR / "X_test.npy", x[test_index])
    np.save(BASE_DIR / "y_train.npy", y[train_index])
    np.save(BASE_DIR / "y_test.npy", y[test_index])

    print("Preprocessing complete.")
    print("Created X_train.npy, X_test.npy, y_train.npy, y_test.npy, and params.json.")


if __name__ == "__main__":
    main()
