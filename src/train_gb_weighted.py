
"""Evaluate weighted Gradient Boosting using stratified cross-validation."""

from pathlib import Path

import pandas as pd

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "invoices_clean.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "gb_weighted_evaluation.csv"

TARGET = "is_fraud"
RANDOM_STATE = 42
N_SPLITS = 5


# ---------------------------------------------------------
# 2. Load and prepare data
# ---------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "invoice_id",
        "invoice_date",
        "due_date",
        "amount",
        "payment_terms_days",
        "line_item_count",
        "vendor_id",
        "category",
        "payment_method",
        "country",
        TARGET,
    }

    missing = required_columns - set(df.columns)

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

    categorical_columns = [
        "vendor_id",
        "category",
        "payment_method",
        "country",
    ]

    for column in categorical_columns:
        X[column] = X[column].astype("string")

    y = df[TARGET].astype(int)

    return X, y


# ---------------------------------------------------------
# 3. Build preprocessing pipeline
# ---------------------------------------------------------

def build_pipeline():

    numeric_columns = [
        "amount",
        "payment_terms_days",
        "line_item_count",
        "invoice_year",
        "invoice_month",
        "invoice_day_of_week",
        "due_days_after_invoice",
    ]

    categorical_columns = [
        "vendor_id",
        "category",
        "payment_method",
        "country",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------
# 4. Stratified cross-validation
# ---------------------------------------------------------

def evaluate_model():

    X, y = load_data()

    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())

    if y.value_counts().min() < N_SPLITS:
        raise ValueError("Insufficient minority samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Calculate weights using training labels only.
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        model = build_pipeline()

        print(f"\nTraining fold {fold}/{N_SPLITS}...")

        # Pipeline passes sample weights to the final classifier.
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights,
        )

        # Validation fold remains untouched.
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

        results.append(metrics)

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

    results_df = pd.DataFrame(results)

    summary = results_df.drop(columns=["fold"]).agg(
        ["mean", "std"]
    ).round(4)

    print("\n========== WEIGHTED GRADIENT BOOSTING RESULTS ==========")
    print(summary)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(REPORT_PATH, index=False)

    print("\nFold results saved to:", REPORT_PATH)


if __name__ == "__main__":
    evaluate_model()
