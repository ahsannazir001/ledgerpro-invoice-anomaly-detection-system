
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = ROOT / "reports" / "oof_predictions.csv"
DATASET_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
REPORTS_DIR = ROOT / "reports"


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"OOF predictions not found: {PREDICTIONS_PATH}"
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {DATASET_PATH}"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    dataset = pd.read_csv(DATASET_PATH)

    required_prediction_columns = {
        "invoice_id",
        "actual",
        "probability",
        "predicted",
    }

    missing = required_prediction_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    required_dataset_columns = {
        "invoice_id",
        "amount",
        "is_fraud",
    }

    missing = required_dataset_columns - set(dataset.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    if predictions["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs found in predictions.")

    if dataset["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs found in dataset.")

    if len(predictions) != len(dataset):
        raise ValueError(
            "Prediction and dataset row counts do not match."
        )

    predictions["actual"] = predictions["actual"].astype(int)
    predictions["predicted"] = predictions["predicted"].astype(int)

    predictions["probability"] = pd.to_numeric(
        predictions["probability"],
        errors="coerce",
    )

    if predictions["probability"].isna().any():
        raise ValueError("Invalid probability values found.")

    if not predictions["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not predictions["actual"].isin([0, 1]).all():
        raise ValueError("Actual labels must be binary.")

    if not predictions["predicted"].isin([0, 1]).all():
        raise ValueError("Predicted labels must be binary.")

    # Attach invoice amount for business-level error analysis.
    analysis = predictions.merge(
        dataset[["invoice_id", "amount", "is_fraud"]],
        on="invoice_id",
        how="left",
        validate="one_to_one",
    )

    if analysis["amount"].isna().any():
        raise ValueError("Some invoice amounts could not be matched.")

    if not (
        analysis["actual"] == analysis["is_fraud"]
    ).all():
        raise ValueError(
            "Prediction actual labels do not match the clean dataset."
        )

    # Identify the four prediction outcomes.
    false_negatives = analysis[
        (analysis["actual"] == 1)
        & (analysis["predicted"] == 0)
    ].copy()

    false_positives = analysis[
        (analysis["actual"] == 0)
        & (analysis["predicted"] == 1)
    ].copy()

    true_positives = analysis[
        (analysis["actual"] == 1)
        & (analysis["predicted"] == 1)
    ].copy()

    true_negatives = analysis[
        (analysis["actual"] == 0)
        & (analysis["predicted"] == 0)
    ].copy()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    false_negatives = false_negatives.sort_values(
        "probability",
        ascending=False,
    )

    false_positives = false_positives.sort_values(
        "probability",
        ascending=False,
    )

    false_negatives.to_csv(
        REPORTS_DIR / "false_negatives.csv",
        index=False,
    )

    false_positives.to_csv(
        REPORTS_DIR / "false_positives.csv",
        index=False,
    )

    # Generate summary statistics.
    summary = {
        "True Positives": len(true_positives),
        "True Negatives": len(true_negatives),
        "False Positives": len(false_positives),
        "False Negatives": len(false_negatives),
    }

    lines = [
        "LEDGERPRO - OOF ERROR ANALYSIS",
        "=" * 45,
        "",
        "PREDICTION OUTCOMES",
        "-" * 45,
    ]

    for label, count in summary.items():
        lines.append(f"{label}: {count}")

    lines.extend(
        [
            "",
            "FALSE NEGATIVE ANALYSIS",
            "-" * 45,
            f"Missed fraud invoices: {len(false_negatives)}",
        ]
    )

    if not false_negatives.empty:
        lines.extend(
            [
                f"Average amount: {false_negatives['amount'].mean():.2f}",
                f"Median amount: {false_negatives['amount'].median():.2f}",
                f"Maximum amount: {false_negatives['amount'].max():.2f}",
                f"Minimum amount: {false_negatives['amount'].min():.2f}",
                f"Average probability: {false_negatives['probability'].mean():.4f}",
                "",
                "Highest probability missed fraud invoices:",
                false_negatives[
                    ["invoice_id", "amount", "probability"]
                ].head(10).to_string(index=False),
            ]
        )

    lines.extend(
        [
            "",
            "FALSE POSITIVE ANALYSIS",
            "-" * 45,
            f"Legitimate invoices flagged as fraud: {len(false_positives)}",
        ]
    )

    if not false_positives.empty:
        lines.extend(
            [
                f"Average amount: {false_positives['amount'].mean():.2f}",
                f"Median amount: {false_positives['amount'].median():.2f}",
                f"Maximum amount: {false_positives['amount'].max():.2f}",
                f"Minimum amount: {false_positives['amount'].min():.2f}",
                f"Average probability: {false_positives['probability'].mean():.4f}",
                "",
                "Highest probability false positives:",
                false_positives[
                    ["invoice_id", "amount", "probability"]
                ].head(10).to_string(index=False),
            ]
        )

    report_path = REPORTS_DIR / "oof_error_analysis.txt"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("LEDGERPRO - OOF ERROR ANALYSIS")
    print("=" * 55)

    for label, count in summary.items():
        print(f"{label}: {count}")

    print("\nFalse Negative Amount Statistics:")
    if not false_negatives.empty:
        print(
            false_negatives["amount"].describe().round(2)
        )
    else:
        print("No false negatives found.")

    print("\nFalse Positive Amount Statistics:")
    if not false_positives.empty:
        print(
            false_positives["amount"].describe().round(2)
        )
    else:
        print("No false positives found.")

    print("\nSaved reports:")
    print("- reports/false_negatives.csv")
    print("- reports/false_positives.csv")
    print("- reports/oof_error_analysis.txt")


if __name__ == "__main__":
    main()
