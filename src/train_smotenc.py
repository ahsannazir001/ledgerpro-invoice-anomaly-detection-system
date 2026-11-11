
"""Evaluate SMOTENC + Gradient Boosting using stratified CV."""

from pathlib import Path

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "invoices_clean.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "smotenc_evaluation.csv"

TARGET = "is_fraud"
RANDOM_STATE = 42
N_SPLITS = 5

NUMERIC_COLUMNS = [
    "amount",
    "payment_terms_days",
    "line_item_count",
    "invoice_year",
    "invoice_month",
    "invoice_day_of_week",
    "due_days_after_invoice",
]

CATEGORICAL_COLUMNS = [
    "vendor_id",
    "category",
    "payment_method",
    "country",
]


# ---------------------------------------------------------
# 2. Load and prepare dataset
# ---------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {
        "invoice_id",
        "invoice_date",
        "due_date",
        TARGET,
        *NUMERIC_COLUMNS[:3],
        *CATEGORICAL_COLUMNS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty or df[TARGET].isna().any():
        raise ValueError("Dataset is empty or target contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both classes: 0 and 1.")

    X = df.drop(columns=[TARGET, "invoice_id"]).copy()

    invoice_dates = pd.to_datetime(
        X.pop("invoice_date"),
        errors="coerce",
    )

    due_dates = pd.to_datetime(
        X.pop("due_date"),
        errors="coerce",
    )

    X["invoice_year"] = invoice_dates.dt.year
    X["invoice_month"] = invoice_dates.dt.month
    X["invoice_day_of_week"] = invoice_dates.dt.dayofweek

    X["due_days_after_invoice"] = (
        due_dates - invoice_dates
    ).dt.days

    for column in CATEGORICAL_COLUMNS:
        X[column] = X[column].astype("string")

    y = df[TARGET].astype(int)

    return X, y


# ---------------------------------------------------------
# 3. Build SMOTENC-compatible preprocessing
# ---------------------------------------------------------

def build_pipeline():

    # First transform:
    # Numerical values are imputed.
    # Categorical values are encoded as integer categories.

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    initial_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, NUMERIC_COLUMNS),
            ("categorical", categorical_preprocessor, CATEGORICAL_COLUMNS),
        ],
        sparse_threshold=0.0,
    )

    # After initial preprocessing:
    # Numerical columns come first, followed by categorical columns.

    categorical_indices = list(
        range(
            len(NUMERIC_COLUMNS),
            len(NUMERIC_COLUMNS) + len(CATEGORICAL_COLUMNS),
        )
    )

    sampler = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=0.5,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    # Second transform:
    # One-hot encode categorical columns after SMOTENC.

    final_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                list(range(len(NUMERIC_COLUMNS))),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_indices,
            ),
        ],
        sparse_threshold=0.0,
    )

    classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )

    return ImbPipeline(
        steps=[
            ("initial_preprocessor", initial_preprocessor),
            ("smotenc", sampler),
            ("final_preprocessor", final_preprocessor),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------
# 4. Stratified cross-validation
# ---------------------------------------------------------

def evaluate_model():

    X, y = load_data()

    print("Dataset shape:", X.shape)
    print("\nOriginal class distribution:")
    print(y.value_counts())

    if y.value_counts().min() < N_SPLITS:
        raise ValueError("Insufficient minority samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = build_pipeline()

        print(f"\nTraining fold {fold}/{N_SPLITS}...")

        model.fit(X_train, y_train)

        # Validation data is never resampled.
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        metrics = {
            "fold": fold,
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "pr_auc": average_precision_score(
                y_val, probabilities
            ),
            "roc_auc": roc_auc_score(
                y_val, probabilities
            ),
        }

        fold_results.append(metrics)

        print(
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    # -----------------------------------------------------
    # 5. Aggregate results
    # -----------------------------------------------------

    results = pd.DataFrame(fold_results)

    summary = results.drop(columns=["fold"]).agg(
        ["mean", "std"]
    ).round(4)

    print("\n========== SMOTENC CROSS-VALIDATION RESULTS ==========")
    print(summary)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(REPORT_PATH, index=False)

    print("\nFold results saved to:", REPORT_PATH)


if __name__ == "__main__":
    evaluate_model()
