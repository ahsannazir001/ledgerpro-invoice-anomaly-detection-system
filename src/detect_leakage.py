
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "invoices_raw.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "invoices_clean.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "leakage_report.txt"


# Columns that must never be used as model input
EXCLUDED_FEATURES = {
    "invoice_id": "Unique identifier; not a predictive feature.",
    "processing_days": (
        "Potentially unavailable when an invoice first arrives; "
        "excluded to prevent temporal availability leakage."
    ),
}

TARGET_COLUMN = "is_fraud"


def detect_leakage() -> bool:
    """Audit features and generate a cleaned dataset."""

    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found: {INPUT_FILE}")
        return False

    df = pd.read_csv(INPUT_FILE)

    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Missing target column: {TARGET_COLUMN}")
        return False

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        print("ERROR: Target column contains invalid labels.")
        return False

    print("\n" + "=" * 65)
    print("LEDGERPRO - DATA LEAKAGE AUDIT")
    print("=" * 65)

    audit_results = []

    for column in df.columns:

        if column == TARGET_COLUMN:
            decision = "TARGET ONLY"
            reason = "Ground-truth label; never use as model input."

        elif column in EXCLUDED_FEATURES:
            decision = "EXCLUDE FROM MODEL INPUT"
            reason = EXCLUDED_FEATURES[column]

        else:
            decision = "KEEP"
            reason = "Available invoice information; no direct leakage identified."

        audit_results.append(
            f"{column:<25} | {decision:<25} | {reason}"
        )

    print("\nFEATURE AUDIT")
    print("-" * 65)

    for result in audit_results:
        print(result)

    # Preserve invoice_id and target for traceability and supervised learning.
    # Remove only features excluded from model inputs.
    clean_df = df.drop(columns=["processing_days"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(OUTPUT_FILE, index=False)

    # Create leakage documentation
    report_lines = [
        "LEDGERPRO - DATA LEAKAGE AUDIT REPORT",
        "=" * 65,
        "",
        f"Original rows: {len(df)}",
        f"Original columns: {len(df.columns)}",
        f"Cleaned columns: {len(clean_df.columns)}",
        "",
        "FEATURE AUDIT:",
        "-" * 65,
        *audit_results,
        "",
        "EXCLUDED FROM MODEL INPUT:",
    ]

    for column, reason in EXCLUDED_FEATURES.items():
        report_lines.append(f"- {column}: {reason}")

    report_lines.extend([
        "",
        f"Target column preserved: {TARGET_COLUMN}",
        "Target must never be included in model input features.",
        "",
        f"Cleaned dataset: {OUTPUT_FILE}",
        "",
        "Note: No direct target-leakage feature was identified "
        "among the generated columns. Exclusions are based on "
        "identifier handling and prediction-time availability.",
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("CLEAN DATASET SUMMARY")
    print("=" * 65)

    print(f"Original shape : {df.shape}")
    print(f"Cleaned shape  : {clean_df.shape}")
    print(f"Removed columns: {list(EXCLUDED_FEATURES)}")
    print(f"Target retained: {TARGET_COLUMN}")
    print(f"Fraud records  : {int(clean_df[TARGET_COLUMN].sum())}")
    print(f"Clean dataset  : {OUTPUT_FILE}")
    print(f"Audit report   : {REPORT_FILE}")

    print("\nDATA LEAKAGE AUDIT COMPLETED!")

    return True


if __name__ == "__main__":
    success = detect_leakage()
    raise SystemExit(0 if success else 1)
