
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "external_test"
    / "invoices_external_test_raw.csv"
)

PIPELINE_PATH = ROOT / "models" / "final_pipeline.joblib"

REPORT_DIR = ROOT / "reports"

THRESHOLD = 0.50

FEATURES = [
    "amount",
    "payment_terms_days",
    "line_item_count",
    "invoice_year",
    "invoice_month",
    "invoice_day_of_week",
    "due_days_after_invoice",
]


def load_external_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"External dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "invoice_id",
        "invoice_date",
        "due_date",
        "amount",
        "payment_terms_days",
        "line_item_count",
        "is_fraud",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs detected.")

    if df.isnull().any().any():
        raise ValueError(
            "External dataset contains missing values."
        )

    if not df["is_fraud"].isin([0, 1]).all():
        raise ValueError("Invalid target labels.")

    if df["is_fraud"].nunique() != 2:
        raise ValueError(
            "External dataset must contain both classes."
        )

    if not df["amount"].between(50, 250_000).all():
        raise ValueError("Invalid invoice amounts.")

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], errors="raise"
    )

    df["due_date"] = pd.to_datetime(
        df["due_date"], errors="raise"
    )

    if (df["due_date"] < df["invoice_date"]).any():
        raise ValueError("Invalid invoice due dates.")

    actual_terms = (
        df["due_date"] - df["invoice_date"]
    ).dt.days

    if not actual_terms.equals(
        df["payment_terms_days"].astype("int64")
    ):
        raise ValueError("Payment terms consistency failed.")

    df["invoice_year"] = df["invoice_date"].dt.year
    df["invoice_month"] = df["invoice_date"].dt.month

    df["invoice_day_of_week"] = (
        df["invoice_date"].dt.dayofweek
    )

    df["due_days_after_invoice"] = (
        df["due_date"] - df["invoice_date"]
    ).dt.days

    X = df[FEATURES].copy()
    y = df["is_fraud"].astype(int)

    return df, X, y


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found: {PIPELINE_PATH}"
        )

    print("Loading independent external test dataset...")

    df, X_external, y_external = load_external_dataset()

    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(y_external.value_counts().sort_index())

    # Load the already-trained pipeline.
    # Only load trusted artifacts created by this project.
    print("\nLoading previously trained model...")
    pipeline = joblib.load(PIPELINE_PATH)

    print("Running external predictions...")

    # No fit() or fit_transform() is called here.
    probabilities = pipeline.predict_proba(X_external)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Independent evaluation metrics
    metrics = {
        "accuracy": float(
            accuracy_score(y_external, predictions)
        ),
        "precision": float(
            precision_score(
                y_external, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_external, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_external, predictions, zero_division=0
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_external, probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_external, probabilities
            )
        ),
    }

    matrix = confusion_matrix(
        y_external,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        y_external,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )

    # Save predictions with invoice identifiers.
    predictions_df = pd.DataFrame({
        "invoice_id": df["invoice_id"],
        "actual": y_external,
        "predicted": predictions,
        "fraud_probability": probabilities,
    })

    predictions_path = (
        REPORT_DIR / "external_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # Save metrics.
    metrics_path = (
        REPORT_DIR / "external_test_metrics.json"
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save detailed report.
    report_path = (
        REPORT_DIR / "external_test_evaluation.txt"
    )

    with report_path.open("w", encoding="utf-8") as file:
        file.write("LEDGERPRO - INDEPENDENT EXTERNAL TEST\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Dataset records: {len(df)}\n")
        file.write(f"Fraud cases: {int(y_external.sum())}\n")
        file.write(f"Threshold: {THRESHOLD}\n")
        file.write("Model: Previously trained final_pipeline.joblib\n")
        file.write("Model retraining: NO\n\n")

        file.write("EVALUATION METRICS\n")
        for name, value in metrics.items():
            file.write(f"{name}: {value:.6f}\n")

        file.write("\nCONFUSION MATRIX\n")
        file.write(
            "Rows = actual; columns = predicted\n"
        )
        file.write(str(matrix))

        file.write("\n\nCLASSIFICATION REPORT\n")
        file.write(report)

    # Display results.
    print("\n" + "=" * 60)
    print("INDEPENDENT EXTERNAL TEST RESULTS")
    print("=" * 60)

    for name, value in metrics.items():
        print(f"{name.upper():<15}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    print("\nSaved reports:")
    print(metrics_path)
    print(report_path)
    print(predictions_path)

    if metrics["f1"] > 0.70:
        print("\nF1 TARGET ACHIEVED on independent synthetic test data.")
    else:
        print("\nF1 TARGET NOT ACHIEVED on independent synthetic test data.")

    print("\nExternal evaluation completed.")


if __name__ == "__main__":
    main()
