
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

NUMERIC_FEATURES = [
    "amount",
    "payment_terms_days",
    "line_item_count",
]


def summarize_feature(df, feature, group_name):
    """Create a frequency and fraud/error summary for one feature."""

    if feature not in df.columns or df.empty:
        return None

    result = (
        df.groupby(feature, dropna=False)
        .agg(
            invoice_count=("invoice_id", "count"),
            average_amount=("amount", "mean"),
            median_amount=("amount", "median"),
        )
        .reset_index()
        .sort_values("invoice_count", ascending=False)
    )

    result.insert(0, "error_group", group_name)

    return result


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
        *NUMERIC_FEATURES,
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
            f"Dataset missing columns: {sorted(missing_dataset)}"
        )

    if missing_predictions:
        raise ValueError(
            f"Predictions missing columns: {sorted(missing_predictions)}"
        )

    if dataset["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs in dataset.")

    if predictions["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs in predictions.")

    # Match predictions to the original invoice features.
    analysis = predictions.merge(
        dataset,
        on="invoice_id",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", "_dataset"),
    )

    if len(analysis) != len(dataset):
        raise ValueError("Dataset and prediction row counts do not match.")

    if analysis["amount"].isna().any():
        raise ValueError("Some invoice features could not be matched.")

    if not (
        analysis["actual"].astype(int)
        == analysis["is_fraud"].astype(int)
    ).all():
        raise ValueError("Prediction labels do not match dataset labels.")

    analysis["actual"] = analysis["actual"].astype(int)
    analysis["predicted"] = analysis["predicted"].astype(int)

    false_negatives = analysis[
        (analysis["actual"] == 1)
        & (analysis["predicted"] == 0)
    ].copy()

    false_positives = analysis[
        (analysis["actual"] == 0)
        & (analysis["predicted"] == 1)
    ].copy()

    groups = {
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }

    all_summaries = []
    report_lines = [
        "LEDGERPRO - ERROR PATTERN ANALYSIS",
        "=" * 55,
        "",
        f"Total invoices analyzed: {len(analysis)}",
        f"False negatives: {len(false_negatives)}",
        f"False positives: {len(false_positives)}",
        "",
    ]

    for group_name, group_df in groups.items():
        report_lines.extend(
            [
                group_name.upper().replace("_", " "),
                "-" * 55,
                f"Invoice count: {len(group_df)}",
            ]
        )

        if group_df.empty:
            report_lines.append("No records found.")
            report_lines.append("")
            continue

        report_lines.extend(
            [
                "",
                "Numeric feature statistics:",
                group_df[NUMERIC_FEATURES]
                .describe()
                .round(2)
                .to_string(),
                "",
            ]
        )

        for feature in CATEGORICAL_FEATURES:
            summary = summarize_feature(
                group_df,
                feature,
                group_name,
            )

            if summary is None:
                continue

            all_summaries.append(summary)

            report_lines.extend(
                [
                    f"Distribution by {feature}:",
                    summary.to_string(index=False),
                    "",
                ]
            )

    # Compare errors with the complete invoice population.
    report_lines.extend(
        [
            "FULL DATASET BASELINE",
            "-" * 55,
            "Numeric feature statistics:",
            analysis[NUMERIC_FEATURES]
            .describe()
            .round(2)
            .to_string(),
            "",
        ]
    )

    # Save detailed error records.
    false_negatives.to_csv(
        REPORTS_DIR / "false_negative_patterns.csv",
        index=False,
    )

    false_positives.to_csv(
        REPORTS_DIR / "false_positive_patterns.csv",
        index=False,
    )

    if all_summaries:
        combined_summary = pd.concat(
            all_summaries,
            ignore_index=True,
        )
        combined_summary.to_csv(
            REPORTS_DIR / "error_feature_distributions.csv",
            index=False,
        )

    report_path = REPORTS_DIR / "error_pattern_analysis.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("LEDGERPRO - ERROR PATTERN ANALYSIS")
    print("=" * 60)

    print(f"Total invoices: {len(analysis)}")
    print(f"False negatives: {len(false_negatives)}")
    print(f"False positives: {len(false_positives)}")

    print("\nFALSE NEGATIVE FEATURE DISTRIBUTIONS")

    for feature in CATEGORICAL_FEATURES:
        print(f"\n{feature}:")

        if false_negatives.empty:
            print("No records.")
        else:
            print(
                false_negatives[feature]
                .value_counts(dropna=False)
                .head(10)
                .to_string()
            )

    print("\nFALSE POSITIVE FEATURE DISTRIBUTIONS")

    for feature in CATEGORICAL_FEATURES:
        print(f"\n{feature}:")

        if false_positives.empty:
            print("No records.")
        else:
            print(
                false_positives[feature]
                .value_counts(dropna=False)
                .head(10)
                .to_string()
            )

    print("\nSaved reports:")
    print("- reports/false_negative_patterns.csv")
    print("- reports/false_positive_patterns.csv")
    print("- reports/error_feature_distributions.csv")
    print("- reports/error_pattern_analysis.txt")


if __name__ == "__main__":
    main()
