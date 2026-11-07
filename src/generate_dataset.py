
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility
RANDOM_SEED = 42
TOTAL_RECORDS = 10_000
FRAUD_COUNT = 100

rng = np.random.default_rng(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "invoices_raw.csv"


def generate_dataset() -> pd.DataFrame:
    """Generate a realistic synthetic invoice dataset."""

    n_fraud = FRAUD_COUNT
    n_legitimate = TOTAL_RECORDS - n_fraud

    labels = np.array(
        [0] * n_legitimate + [1] * n_fraud,
        dtype=np.int8,
    )
    rng.shuffle(labels)

    n = TOTAL_RECORDS

 
    # Generate invoice categories
    categories = rng.choice(
        [
            "Office Supplies",
            "Travel",
            "Equipment",
            "Software",
            "Services",
            "Maintenance",
        ],
        size=n,
        p=[0.22, 0.15, 0.16, 0.14, 0.18, 0.15],
    )

    # Vendor identifiers
    vendor_ids = rng.choice(
        [f"VENDOR_{i:03d}" for i in range(1, 101)],
        size=n,
    )

    # Invoice amounts: fraud invoices have a different
    # but overlapping distribution.
    amounts = np.where(
        labels == 1,
        rng.lognormal(mean=10.4, sigma=0.85, size=n),
        rng.lognormal(mean=7.5, sigma=0.85, size=n),
    )
    amounts = np.clip(amounts, 50, 250_000)
    amounts = np.round(amounts, 2)

    # Invoice dates
    start_date = pd.Timestamp("2025-01-01")
    date_offsets = rng.integers(0, 365, size=n)

    invoice_dates = (
        start_date + pd.to_timedelta(date_offsets, unit="D")
    )

    # Payment terms
    payment_terms = rng.choice(
        [15, 30, 45, 60],
        size=n,
        p=[0.15, 0.55, 0.20, 0.10],
    )

    due_dates = invoice_dates + pd.to_timedelta(
        payment_terms, unit="D"
    )

    # Additional invoice characteristics
    payment_methods = rng.choice(
        ["Bank Transfer", "Credit Card", "Cheque", "Cash"],
        size=n,
        p=[0.55, 0.20, 0.20, 0.05],
    )

    countries = rng.choice(
        ["Pakistan", "USA", "UK", "UAE", "Germany"],
        size=n,
        p=[0.45, 0.20, 0.15, 0.12, 0.08],
    )

    line_item_count = rng.integers(1, 16, size=n)

    # Small natural variation in invoice processing time
    processing_days = rng.integers(0, 8, size=n)

    invoice_df = pd.DataFrame({
        "invoice_id": [
            f"INV-{i:06d}" for i in range(1, n + 1)
        ],
        "vendor_id": vendor_ids,
        "invoice_date": invoice_dates,
        "due_date": due_dates,
        "amount": amounts,
        "category": categories,
        "payment_method": payment_methods,
        "country": countries,
        "payment_terms_days": payment_terms,
        "line_item_count": line_item_count,
        "processing_days": processing_days,
        "is_fraud": labels,
    })

    return invoice_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()

    df.to_csv(OUTPUT_FILE, index=False)

    fraud_count = int(df["is_fraud"].sum())
    legitimate_count = len(df) - fraud_count

    print("\nDataset generated successfully!")
    print("-" * 45)
    print(f"Total records: {len(df)}")
    print(f"Legitimate invoices: {legitimate_count}")
    print(f"Fraudulent invoices: {fraud_count}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("-" * 45)

    print("\nDataset preview:")
    print(df.head())

    print("\nClass distribution:")
    print(df["is_fraud"].value_counts())


if __name__ == "__main__":
    main()
