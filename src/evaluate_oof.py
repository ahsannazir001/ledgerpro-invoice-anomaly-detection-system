
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5


def load_and_prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "invoice_id",
        "invoice_date",
        "due_date",
        "is_fraud",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["is_fraud"].isna().any():
        raise ValueError("Target column contains missing values.")

    y = df["is_fraud"].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only binary labels 0 and 1.")

    if y.value_counts().min() < N_SPLITS:
        raise ValueError(
            f"Each class must have at least {N_SPLITS} samples for "
            f"{N_SPLITS}-fold stratified cross-validation."
        )

    invoice_ids = df["invoice_id"].copy()

    X = df.drop(columns=["is_fraud", "invoice_id"]).copy()

    X["invoice_date"] = pd.to_datetime(
        X["invoice_date"], errors="coerce"
    )
    X["due_date"] = pd.to_datetime(
        X["due_date"], errors="coerce"
    )

    X["invoice_year"] = X["invoice_date"].dt.year
    X["invoice_month"] = X["invoice_date"].dt.month
    X["invoice_day_of_week"] = X["invoice_date"].dt.dayofweek

    X["due_days_after_invoice"] = (
        X["due_date"] - X["invoice_date"]
    ).dt.days

    X = X.drop(columns=["invoice_date", "due_date"])

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in categorical_columns:
        X[column] = X[column].astype("string")

    return X, y, invoice_ids, numeric_columns, categorical_columns


def build_model(numeric_columns, categorical_columns):
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    model = ImbPipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "smote",
                SMOTE(
                    sampling_strategy=0.5,
                    k_neighbors=5,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, invoice_ids, numeric_columns, categorical_columns = (
        load_and_prepare_data()
    )

    print("=" * 60)
    print("LEDGERPRO - OUT-OF-FOLD MODEL EVALUATION")
    print("=" * 60)
    print(f"Dataset rows: {len(X)}")
    print(f"Legitimate invoices: {(y == 0).sum()}")
    print(f"Fraudulent invoices: {(y == 1).sum()}")
    print(f"Cross-validation folds: {N_SPLITS}")
    print(f"Classification threshold: {THRESHOLD}")
    print()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.full(len(X), np.nan)
    fold_results = []

    for fold_number, (train_indices, valid_indices) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]

        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]

        model = build_model(numeric_columns, categorical_columns)

        # Preprocessing and SMOTE are fitted only on this training fold.
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        oof_probabilities[valid_indices] = probabilities

        precision = precision_score(
            y_valid, predictions, zero_division=0
        )
        recall = recall_score(
            y_valid, predictions, zero_division=0
        )
        f1 = f1_score(
            y_valid, predictions, zero_division=0
        )

        fold_results.append(
            {
                "fold": fold_number,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fraud_samples": int(y_valid.sum()),
            }
        )

        print(
            f"Fold {fold_number}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(
            "Some rows did not receive an out-of-fold prediction."
        )

    oof_predictions = (
        oof_probabilities >= THRESHOLD
    ).astype(int)

    pooled_precision = precision_score(
        y, oof_predictions, zero_division=0
    )
    pooled_recall = recall_score(
        y, oof_predictions, zero_division=0
    )
    pooled_f1 = f1_score(
        y, oof_predictions, zero_division=0
    )
    pooled_pr_auc = average_precision_score(
        y, oof_probabilities
    )
    pooled_roc_auc = roc_auc_score(
        y, oof_probabilities
    )

    matrix = confusion_matrix(
        y, oof_predictions, labels=[0, 1]
    )

    predictions_df = pd.DataFrame(
        {
            "invoice_id": invoice_ids,
            "actual": y,
            "probability": oof_probabilities,
            "predicted": oof_predictions,
        }
    )

    predictions_path = REPORTS_DIR / "oof_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    fold_df = pd.DataFrame(fold_results)
    fold_path = REPORTS_DIR / "oof_fold_metrics.csv"
    fold_df.to_csv(fold_path, index=False)

    report_lines = [
        "LEDGERPRO - OUT-OF-FOLD EVALUATION",
        "=" * 45,
        f"Rows evaluated: {len(X)}",
        f"Cross-validation folds: {N_SPLITS}",
        f"Threshold: {THRESHOLD}",
        "",
        "POOLED OUT-OF-FOLD METRICS",
        "-" * 45,
        f"Precision: {pooled_precision:.4f}",
        f"Recall: {pooled_recall:.4f}",
        f"F1-score: {pooled_f1:.4f}",
        f"PR-AUC (Average Precision): {pooled_pr_auc:.4f}",
        f"ROC-AUC: {pooled_roc_auc:.4f}",
        "",
        "CONFUSION MATRIX",
        "-" * 45,
        str(matrix),
        "",
        "CLASSIFICATION REPORT",
        "-" * 45,
        classification_report(
            y,
            oof_predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        ),
        "",
        "FOLD METRICS",
        "-" * 45,
        fold_df.to_string(index=False),
        "",
        "NOTE: These are out-of-fold cross-validation estimates.",
        "The validation fold was not used for fitting preprocessing or SMOTE.",
        "Threshold tuning is not performed in this evaluation.",
    ]

    report_path = REPORTS_DIR / "oof_evaluation.txt"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 60)
    print(f"Precision: {pooled_precision:.4f}")
    print(f"Recall:    {pooled_recall:.4f}")
    print(f"F1-score:  {pooled_f1:.4f}")
    print(f"PR-AUC:    {pooled_pr_auc:.4f}")
    print(f"ROC-AUC:   {pooled_roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            oof_predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("Saved files:")
    print(f"- {predictions_path}")
    print(f"- {fold_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
