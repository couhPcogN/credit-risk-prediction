from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
from django.conf import settings


ML_PROJECT_ROOT = Path(getattr(settings, "ML_PROJECT_ROOT", Path(settings.BASE_DIR).parent))
if str(ML_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_PROJECT_ROOT))

from model import KNN, LinearRegression, LogisticRegression


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

CATEGORY_MAPS = {
    "person_home_ownership": {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3,
    },
    "loan_intent": {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3,
        "PERSONAL": 4,
        "VENTURE": 5,
    },
    "loan_grade": {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
    },
    "cb_person_default_on_file": {
        "N": 0,
        "Y": 1,
    },
}

FIELD_LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_home_ownership": "Home ownership",
    "person_emp_length": "Employment length",
    "loan_intent": "Loan purpose",
    "loan_grade": "Credit grade",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan-to-income ratio",
    "cb_person_default_on_file": "Previous default",
    "cb_person_cred_hist_length": "Credit history length",
}

FORM_OPTIONS = {
    "person_home_ownership": [
        {"value": "MORTGAGE", "label": "Mortgage"},
        {"value": "OTHER", "label": "Other"},
        {"value": "OWN", "label": "Own"},
        {"value": "RENT", "label": "Rent"},
    ],
    "loan_intent": [
        {"value": "DEBTCONSOLIDATION", "label": "Debt consolidation"},
        {"value": "EDUCATION", "label": "Education"},
        {"value": "HOMEIMPROVEMENT", "label": "Home improvement"},
        {"value": "MEDICAL", "label": "Medical"},
        {"value": "PERSONAL", "label": "Personal"},
        {"value": "VENTURE", "label": "Business"},
    ],
    "loan_grade": [
        {"value": "A", "label": "A - best"},
        {"value": "B", "label": "B"},
        {"value": "C", "label": "C"},
        {"value": "D", "label": "D"},
        {"value": "E", "label": "E"},
        {"value": "F", "label": "F"},
        {"value": "G", "label": "G - highest risk"},
    ],
    "cb_person_default_on_file": [
        {"value": "N", "label": "No"},
        {"value": "Y", "label": "Yes"},
    ],
}

DEFAULT_INPUTS = {
    "person_age": 30,
    "person_income": 65000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 12000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.18,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6,
}

FORM_FIELD_ALIASES = {
    "age": "person_age",
    "income": "person_income",
    "home_ownership": "person_home_ownership",
    "emp_length": "person_emp_length",
    "loan_intent": "loan_intent",
    "loan_grade": "loan_grade",
    "loan_amnt": "loan_amnt",
    "int_rate": "loan_int_rate",
    "loan_percent_income": "loan_percent_income",
    "default_on_file": "cb_person_default_on_file",
    "cred_hist_length": "cb_person_cred_hist_length",
}

CANONICAL_TO_FORM_FIELD = {canonical: form for form, canonical in FORM_FIELD_ALIASES.items()}

FIELD_LIMITS = {
    "person_age": (18, 100),
    "person_income": (0, 10_000_000),
    "person_emp_length": (0, 60),
    "loan_amnt": (0, 1_000_000),
    "loan_int_rate": (0, 100),
    "loan_percent_income": (0, 1),
    "cb_person_cred_hist_length": (0, 50),
}

REPORT_IMAGES = [
    {
        "filename": "comparison_80_20_chart.png",
        "title": "Model comparison",
        "description": "Accuracy, Precision, Recall, and F1-score for all three models.",
    },
    {
        "filename": "roc_curve_comparison.png",
        "title": "ROC and AUC",
        "description": "The ability to distinguish safe and default profiles.",
    },
    {
        "filename": "feature_importance_logistic.png",
        "title": "Feature importance",
        "description": "The most influential Logistic Regression coefficients.",
    },
    {
        "filename": "confusion_matrix_knn.png",
        "title": "KNN confusion matrix",
        "description": "The four correct and incorrect prediction categories for KNN.",
    },
    {
        "filename": "eda_class_distribution.png",
        "title": "Class distribution",
        "description": "The proportion of safe and default profiles in the dataset.",
    },
    {
        "filename": "eda_correlation_matrix.png",
        "title": "Feature correlations",
        "description": "Relationships between numerical variables in the dataset.",
    },
]

REPORT_IMAGE_NAMES = {report["filename"] for report in REPORT_IMAGES}


@dataclass(frozen=True)
class PredictionResult:
    label: str
    summary: str
    risk_probability: float
    risk_percent: float
    is_high_risk: bool
    nearest_risk_count: int
    nearest_total: int
    notes: list[str]
    inputs: dict[str, float | str]


@dataclass(frozen=True)
class DatasetSummary:
    total_count: int
    train_count: int
    test_count: int
    safe_count: int
    risk_count: int
    risk_rate: float
    feature_count: int


class CreditRiskService:
    def __init__(self) -> None:
        self.root_dir = ML_PROJECT_ROOT
        self.params_path = self.root_dir / "params.json"
        self.dataset_path = self.root_dir / "data" / "credit_risk_dataset.csv"

        self.params = self._load_params()
        self.numeric_params = self._extract_numeric_params(self.params)
        self.category_maps = self.params.get("category_maps", CATEGORY_MAPS)

        self.x_train = np.load(self.root_dir / "X_train.npy")
        self.y_train = np.load(self.root_dir / "y_train.npy")
        self.x_test = np.load(self.root_dir / "X_test.npy")
        self.y_test = np.load(self.root_dir / "y_test.npy")

        self.model = KNN(k=5)
        self.model.fit(self.x_train, self.y_train)
        self.linear_model = LinearRegression(lr=0.1, iterations=1000)
        self.linear_model.fit(self.x_train, self.y_train)
        self.logistic_model = LogisticRegression(lr=0.1, max_iterations=5000)
        self.logistic_model.fit(self.x_train, self.y_train)

        self.summary = self._build_summary()
        self.dataset_profile = self._build_dataset_profile()

    def _load_params(self) -> dict:
        with self.params_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _extract_numeric_params(self, params: Mapping[str, object]) -> dict[str, dict[str, float]]:
        numeric_params: dict[str, dict[str, float]] = {}
        for feature, value in params.items():
            if isinstance(value, dict) and {"min", "max"}.issubset(value):
                numeric_params[feature] = {
                    "min": float(value["min"]),
                    "max": float(value["max"]),
                }
        return numeric_params

    def _build_summary(self) -> DatasetSummary:
        y_all = np.concatenate([self.y_train, self.y_test])
        risk_count = int(np.sum(y_all == 1))
        safe_count = int(np.sum(y_all == 0))
        total_count = int(len(y_all))
        risk_rate = (risk_count / total_count * 100) if total_count else 0.0
        return DatasetSummary(
            total_count=total_count,
            train_count=int(len(self.y_train)),
            test_count=int(len(self.y_test)),
            safe_count=safe_count,
            risk_count=risk_count,
            risk_rate=risk_rate,
            feature_count=len(FEATURE_ORDER),
        )

    def _build_dataset_profile(self) -> dict[str, list[dict[str, float | int | str]]]:
        if not self.dataset_path.exists():
            return {"grades": [], "intents": []}

        grade_counts: Counter[str] = Counter()
        intent_counts: Counter[str] = Counter()

        with self.dataset_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                grade_counts[row.get("loan_grade", "")] += 1
                intent_counts[row.get("loan_intent", "")] += 1

        return {
            "grades": self._counter_to_rows(grade_counts, FORM_OPTIONS["loan_grade"]),
            "intents": self._counter_to_rows(intent_counts, FORM_OPTIONS["loan_intent"]),
        }

    def _counter_to_rows(
        self,
        counter: Counter[str],
        options: list[dict[str, str]],
    ) -> list[dict[str, float | int | str]]:
        total = sum(counter.values()) or 1
        labels = {option["value"]: option["label"] for option in options}
        rows = []
        for key, count in counter.most_common():
            if not key:
                continue
            rows.append({
                "value": key,
                "label": labels.get(key, key),
                "count": int(count),
                "percent": round(count / total * 100, 1),
            })
        return rows

    def display_inputs(self, data: Mapping[str, object] | None = None) -> dict[str, object]:
        values: dict[str, object] = DEFAULT_INPUTS.copy()
        if not data:
            return values

        for field in DEFAULT_INPUTS:
            raw_value = self._raw_value(data, field)
            if raw_value not in (None, ""):
                values[field] = raw_value
        return values

    def parse_inputs(self, data: Mapping[str, object]) -> dict[str, float | str]:
        values: dict[str, float | str] = {}

        for feature in NUMERIC_FEATURES:
            raw_value = self._raw_value(data, feature)
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{FIELD_LABELS[feature]} must be a valid number.") from exc

            lower, upper = FIELD_LIMITS[feature]
            if value < lower or value > upper:
                raise ValueError(f"{FIELD_LABELS[feature]} must be between {lower:g} and {upper:g}.")
            values[feature] = value

        for feature, choices in self.category_maps.items():
            value = str(self._raw_value(data, feature))
            if value not in choices:
                raise ValueError(f"{FIELD_LABELS[feature]} is not valid.")
            values[feature] = value

        return values

    def _raw_value(self, data: Mapping[str, object], canonical_field: str) -> object:
        value = data.get(canonical_field)
        if value not in (None, ""):
            return value

        form_field = CANONICAL_TO_FORM_FIELD.get(canonical_field)
        if form_field:
            value = data.get(form_field)
            if value not in (None, ""):
                return value

        return DEFAULT_INPUTS[canonical_field]

    def build_feature_vector(self, inputs: Mapping[str, float | str]) -> np.ndarray:
        vector = []
        for feature in FEATURE_ORDER:
            if feature in NUMERIC_FEATURES:
                vector.append(self.normalize_numeric_value(feature, float(inputs[feature])))
            else:
                vector.append(float(self.encode_category(feature, str(inputs[feature]))))
        return np.array(vector, dtype=float).reshape(1, -1)

    def normalize_numeric_value(self, feature: str, value: float) -> float:
        params = self.numeric_params[feature]
        min_value = params["min"]
        max_value = params["max"]
        denominator = max_value - min_value
        if denominator == 0:
            return 0.0
        return (value - min_value) / denominator

    def encode_category(self, feature: str, value: str) -> int:
        return int(self.category_maps[feature][value])

    def predict(self, data: Mapping[str, object]) -> PredictionResult:
        inputs = self.parse_inputs(data)
        vector = self.build_feature_vector(inputs)
        prediction = int(self.model.predict(vector)[0])
        risk_probability, nearest_risk_count, nearest_total = self._knn_risk_probability(vector[0])
        is_high_risk = prediction == 1

        return PredictionResult(
            label="Default" if is_high_risk else "Safe",
            summary=(
                "This profile has elevated risk and requires additional review."
                if is_high_risk
                else "This profile is currently classified as relatively safe."
            ),
            risk_probability=risk_probability,
            risk_percent=round(risk_probability * 100, 1),
            is_high_risk=is_high_risk,
            nearest_risk_count=nearest_risk_count,
            nearest_total=nearest_total,
            notes=self._risk_notes(inputs, risk_probability),
            inputs=dict(inputs),
        )

    def predict_all(self, data: Mapping[str, object]) -> dict[str, dict[str, float | str]]:
        inputs = self.parse_inputs(data)
        vector = self.build_feature_vector(inputs)

        knn_prediction = int(self.model.predict(vector)[0])
        knn_risk_probability, _, _ = self._knn_risk_probability(vector[0])

        linear_prediction = int(self.linear_model.predict(vector)[0])
        linear_raw_score = float((np.dot(vector, self.linear_model.weights) + self.linear_model.bias)[0])
        linear_risk_probability = self._clip_probability(linear_raw_score)

        logistic_risk_probability = float(self.logistic_model.predict_prob(vector)[0])
        logistic_prediction = int(self.logistic_model.predict(vector)[0])

        return {
            "KNN": self._format_model_result(knn_prediction, knn_risk_probability),
            "Linear Regression": self._format_model_result(linear_prediction, linear_risk_probability),
            "Logistic Regression": self._format_model_result(logistic_prediction, logistic_risk_probability),
        }

    def _format_model_result(self, prediction: int, risk_probability: float) -> dict[str, float | str]:
        return {
            "label": "Default" if prediction == 1 else "Safe",
            "risk": round(self._clip_probability(risk_probability) * 100, 1),
        }

    def _clip_probability(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _knn_risk_probability(self, vector: np.ndarray) -> tuple[float, int, int]:
        distances = np.sqrt(np.sum((self.x_train - vector) ** 2, axis=1))
        k = min(self.model.k, len(distances))
        nearest_indices = np.argpartition(distances, k - 1)[:k]
        nearest_labels = self.y_train[nearest_indices]
        risk_count = int(np.sum(nearest_labels == 1))
        return risk_count / k if k else 0.0, risk_count, k

    def _risk_notes(self, inputs: Mapping[str, float | str], risk_probability: float) -> list[str]:
        notes: list[str] = []
        if float(inputs["loan_percent_income"]) >= 0.35:
            notes.append("The loan-to-income ratio is relatively high.")
        if str(inputs["cb_person_default_on_file"]) == "Y":
            notes.append("The profile has a previous default on record.")
        if str(inputs["loan_grade"]) in {"D", "E", "F", "G"}:
            notes.append("The credit grade requires additional attention.")
        if float(inputs["loan_int_rate"]) >= 15:
            notes.append("A high interest rate is commonly associated with greater risk.")
        if risk_probability == 0:
            notes.append("All five nearest training profiles are classified as safe.")
        elif risk_probability == 1:
            notes.append("All five nearest training profiles are classified as default.")
        if not notes:
            notes.append("The main signals are within a stable range compared with the training set.")
        return notes[:4]

    def class_distribution(self) -> list[dict[str, float | int | str]]:
        total = self.summary.total_count or 1
        return [
            {
                "label": "Safe",
                "count": self.summary.safe_count,
                "percent": round(self.summary.safe_count / total * 100, 1),
            },
            {
                "label": "Default",
                "count": self.summary.risk_count,
                "percent": round(self.summary.risk_count / total * 100, 1),
            },
        ]

    def reports(self) -> list[dict[str, str | bool]]:
        reports = []
        for report in REPORT_IMAGES:
            path = resolve_report_path(report["filename"])
            reports.append({**report, "available": path is not None})
        return reports


def resolve_report_path(filename: str) -> Path | None:
    if filename not in REPORT_IMAGE_NAMES:
        return None

    path = ML_PROJECT_ROOT / "web_app" / "prediction" / "static" / "images" / filename
    if not path.exists() or not path.is_file():
        return None
    return path


@lru_cache(maxsize=1)
def get_credit_risk_service() -> CreditRiskService:
    return CreditRiskService()
