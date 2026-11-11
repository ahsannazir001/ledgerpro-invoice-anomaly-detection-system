
"""Compare baseline fraud classifiers using stratified cross-validation."""

from pathlib import Path

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "invoices_clean.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"

TARGET = "is_fraud"
RANDOM_STATE = 42
N_SPLITS = 5


# ---------------------------------------------------------
# 2. Load and prepare dataset
# ---------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both classes 0 and 1.")

    y = df[TARGET].astype(int)

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

    return X, y


# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------

def build_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------
# 4. Define candidate models
# ---------------------------------------------------------

def build_models(preprocessor):

    return {
        "RandomForest_ClassWeight": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "RandomForest_SMOTE": ImbPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=0.5,
                        random_state=RANDOM_STATE,
                        k_neighbors=5,
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "GradientBoosting_SMOTE": ImbPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=0.5,
                        random_state=RANDOM_STATE,
                        k_neighbors=5,
                    ),
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------
# 5. Cross-validation
# ---------------------------------------------------------

def compare_models():

    X, y = load_data()

    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())

    minority_count = int(y.value_counts().min())

    if minority_count < N_SPLITS:
        raise ValueError("Not enough minority samples for cross-validation.")

    preprocessor = build_preprocessor(X)
    models = build_models(preprocessor)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, model in models.items():

        print(f"\n{'=' * 55}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 55}")

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            if "smote" in model.named_steps:
                train_minority = int(y_train.value_counts().min())

                if train_minority <= 5:
                    raise ValueError(
                        "Not enough minority samples for SMOTE."
                    )

            fitted_model = clone(model)

            fitted_model.fit(X_train, y_train)

            y_pred = fitted_model.predict(X_val)
            y_prob = fitted_model.predict_proba(X_val)[:, 1]

            precision = precision_score(
                y_val, y_pred, zero_division=0
            )

            recall = recall_score(
                y_val, y_pred, zero_division=0
            )

            f1 = f1_score(
                y_val, y_pred, zero_division=0
            )

            pr_auc = average_precision_score(y_val, y_prob)
            roc_auc = roc_auc_score(y_val, y_prob)

            fold_result = {
                "model": model_name,
                "fold": fold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }

            fold_metrics.append(fold_result)

            print(
                f"Fold {fold}: "
                f"Precision={precision:.4f}, "
                f"Recall={recall:.4f}, "
                f"F1={f1:.4f}, "
                f"PR-AUC={pr_auc:.4f}, "
                f"ROC-AUC={roc_auc:.4f}"
            )

        fold_df = pd.DataFrame(fold_metrics)

        results.append(
            {
                "model": model_name,
                "precision_mean": fold_df["precision"].mean(),
                "precision_std": fold_df["precision"].std(),
                "recall_mean": fold_df["recall"].mean(),
                "recall_std": fold_df["recall"].std(),
                "f1_mean": fold_df["f1"].mean(),
                "f1_std": fold_df["f1"].std(),
                "pr_auc_mean": fold_df["pr_auc"].mean(),
                "roc_auc_mean": fold_df["roc_auc"].mean(),
            }
        )

    # -----------------------------------------------------
    # 6. Display and save comparison
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="f1_mean",
        ascending=False,
    ).reset_index(drop=True)

    print("\n\n========== MODEL COMPARISON ==========")
    print(results_df.round(4).to_string(index=False))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(REPORT_PATH, index=False)

    print(f"\nComparison report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    compare_models()
