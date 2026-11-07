
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "invoices_clean.csv"
REPORT_DIR = ROOT / "reports"


def analyze_features():
    df = pd.read_csv(INPUT_FILE)

    target = "is_fraud"

    # Parse date columns correctly
    for column in ["invoice_date", "due_date"]:
        df[column] = pd.to_datetime(
            df[column], errors="coerce"
        )

    # Exclude identifier and target
    features = df.drop(columns=["invoice_id", target])

    numeric_features = features.select_dtypes(
        include=["number"]
    ).columns

    categorical_features = features.select_dtypes(
        include=["object", "category"]
    ).columns

    print("\n" + "=" * 60)
    print("LEDGERPRO - FEATURE ANALYSIS")
    print("=" * 60)

    # Numeric feature correlations
    print("\nNUMERIC FEATURE CORRELATIONS")
    print("-" * 60)

    correlations = df[numeric_features].corrwith(df[target])

    print(correlations.round(4).to_string())

    # Categorical feature fraud rates
    print("\nCATEGORICAL FEATURE FRAUD RATES")
    print("-" * 60)

    for feature in categorical_features:

        rates = df.groupby(
            feature, observed=True
        )[target].agg(["count", "mean"])

        rates["fraud_rate_percent"] = rates["mean"] * 100

        print(f"\nFeature: {feature}")

        if feature == "vendor_id":
            rates = rates.sort_values(
                "count", ascending=False
            ).head(10)

            print("Top 10 vendors by invoice count:")

        print(
            rates[["count", "fraud_rate_percent"]]
            .round(2)
            .to_string()
        )

    # Monthly invoice fraud analysis
    print("\nMONTHLY FRAUD ANALYSIS")
    print("-" * 60)

    df["invoice_month"] = df["invoice_date"].dt.to_period("M")

    monthly = df.groupby(
        "invoice_month", observed=True
    )[target].agg(["count", "sum", "mean"])

    monthly["fraud_rate_percent"] = monthly["mean"] * 100

    print(
        monthly[
            ["count", "sum", "fraud_rate_percent"]
        ].round(2).to_string()
    )

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORT_DIR / "feature_analysis.txt"

    with open(report_file, "w", encoding="utf-8") as file:

        file.write("LEDGERPRO FEATURE ANALYSIS REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write("NUMERIC CORRELATIONS\n")
        file.write(correlations.round(4).to_string())

        file.write("\n\nMONTHLY FRAUD ANALYSIS\n")
        file.write(
            monthly.round(4).to_string()
        )

        file.write(
            "\n\nNote: Statistical association does not "
            "establish data leakage.\n"
        )

    print("\nAnalysis saved:", report_file)


if __name__ == "__main__":
    analyze_features()
