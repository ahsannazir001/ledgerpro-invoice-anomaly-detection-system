
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
PREDICTIONS_PATH = ROOT / "reports" / "oof_predictions.csv"
REPORTS_DIR = ROOT / "reports"

CATEGORICAL_FEATURES = [
    "vendor_id",
    "category",
    "payment_method",
    "country",
]


def calculate_rates(df, feature):
    """
    Calculate fraud and prediction error rates for each feature value.
    """

    result = (
        df.groupby(feature, dropna=False)
        .agg(
            total_invoices=("invoice_id", "count"),
            actual_fraud=("actual", "sum"),
            missed_fraud=("false_negative", "sum"),
            false_positives=("false_positive", "sum"),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = (
        result["actual_fraud"]
        / result["total_invoices"]
        * 100
    )

    result["fraud_miss_rate_pct"] = (
        result["missed_fraud"]
        / result["actual_fraud"].replace(0, pd.NA)
        * 100
    )

    result["false_positive_rate_pct"] = (
        result["false_positives"]
        / (
            result["total_invoices"]
            - result["actual_fraud"]
        ).replace(0, pd.NA)
        * 100
    )

    result["missed_fraud_share_pct"] = (
        result["missed_fraud"]
        / max(int(df["false_negative"].sum()), 1)
        * 100
    )

    result["false_positive_share_pct"] = (
        result["false_positives"]
        / max(int(df["false_positive"].sum()), 1)
        * 100
    )

    return result.sort_values(
        "total_invoices",
        ascending=False,
    )


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    dataset = pd.read_csv(DATASET_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_dataset = {
        "invoice_id",
        "is_fraud",
        *CATEGORICAL_FEATURES,
    }

    required_predictions = {
        "invoice_id",
        "actual",
        "predicted",
        "probability",
    }

    missing_dataset = required_dataset - set(dataset.columns)
    missing_predictions = required_predictions - set(predictions.columns)

    if missing_dataset:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing_dataset)}"
        )

    if missing_predictions:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_predictions)}"
        )

    if dataset["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs in dataset.")

    if predictions["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs in predictions.")

    if len(dataset) != len(predictions):
        raise ValueError("Dataset and prediction row counts differ.")

    df = predictions.merge(
        dataset,
        on="invoice_id",
        how="left",
        validate="one_to_one",
    )

    if df["is_fraud"].isna().any():
        raise ValueError("Some predictions have no matching invoice.")

    df["actual"] = df["actual"].astype(int)
    df["predicted"] = df["predicted"].astype(int)
    df["is_fraud"] = df["is_fraud"].astype(int)

    if not (df["actual"] == df["is_fraud"]).all():
        raise ValueError("Actual labels do not match the dataset.")

    if not df["actual"].isin([0, 1]).all():
        raise ValueError("Actual labels must be binary.")

    if not df["predicted"].isin([0, 1]).all():
        raise ValueError("Predicted labels must be binary.")

    df["false_negative"] = (
        (df["actual"] == 1)
        & (df["predicted"] == 0)
    ).astype(int)

    df["false_positive"] = (
        (df["actual"] == 0)
        & (df["predicted"] == 1)
    ).astype(int)

    total_invoices = len(df)
    total_fraud = int(df["actual"].sum())
    total_legitimate = total_invoices - total_fraud

    total_missed = int(df["false_negative"].sum())
    total_false_positives = int(df["false_positive"].sum())

    overall_fraud_rate = total_fraud / total_invoices * 100

    overall_miss_rate = (
        total_missed / total_fraud * 100
        if total_fraud else 0
    )

    overall_fp_rate = (
        total_false_positives / total_legitimate * 100
        if total_legitimate else 0
    )

    overall_summary = pd.DataFrame(
        [
            {
                "total_invoices": total_invoices,
                "actual_fraud": total_fraud,
                "legitimate_invoices": total_legitimate,
                "missed_fraud": total_missed,
                "false_positives": total_false_positives,
                "overall_fraud_rate_pct": overall_fraud_rate,
                "overall_fraud_miss_rate_pct": overall_miss_rate,
                "overall_false_positive_rate_pct": overall_fp_rate,
            }
        ]
    )

    all_results = []
    report_lines = [
        "LEDGERPRO - ERROR RATE COMPARISON",
        "=" * 65,
        "",
        "OVERALL DATASET BASELINE",
        "-" * 65,
        overall_summary.round(4).to_string(index=False),
        "",
    ]

    for feature in CATEGORICAL_FEATURES:
        result = calculate_rates(df, feature)
        result.insert(0, "feature", feature)

        all_results.append(result)

        report_lines.extend(
            [
                "",
                f"FEATURE: {feature.upper()}",
                "-" * 65,
                result.round(4).to_string(index=False),
                "",
            ]
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_path = REPORTS_DIR / "error_rate_comparison.csv"
    overall_path = REPORTS_DIR / "overall_error_rates.csv"
    report_path = REPORTS_DIR / "error_rate_comparison.txt"

    combined.to_csv(combined_path, index=False)
    overall_summary.to_csv(overall_path, index=False)

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("LEDGERPRO - ERROR RATE COMPARISON")
    print("=" * 65)

    print("\nOVERALL BASELINE")
    print(overall_summary.round(4).to_string(index=False))

    for feature in CATEGORICAL_FEATURES:
        print(f"\nFEATURE: {feature.upper()}")
        print(
            combined[combined["feature"] == feature]
            .round(4)
            .to_string(index=False)
        )

    print("\nSaved reports:")
    print("- reports/error_rate_comparison.csv")
    print("- reports/overall_error_rates.csv")
    print("- reports/error_rate_comparison.txt")


if __name__ == "__main__":
    main()
