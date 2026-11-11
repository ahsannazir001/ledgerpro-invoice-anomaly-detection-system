
from pathlib import Path

import numpy as np
import pandas as pd


# Independent seed: must differ from the original seed (42).
RANDOM_SEED = 20261111

TOTAL_RECORDS = 10_000
FRAUD_COUNT = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "external_test"
OUTPUT_FILE = OUTPUT_DIR / "invoices_external_test_raw.csv"


def generate_external_dataset() -> pd.DataFrame:
    """Generate an independent synthetic invoice evaluation dataset."""

    rng = np.random.default_rng(RANDOM_SEED)

    n_fraud = FRAUD_COUNT
    n_legitimate = TOTAL_RECORDS - n_fraud

    labels = np.array(
        [0] * n_legitimate + [1] * n_fraud,
        dtype=np.int8,
    )
    rng.shuffle(labels)

    n = TOTAL_RECORDS

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

    vendor_ids = rng.choice(
        [f"VENDOR_{i:03d}" for i in range(1, 101)],
        size=n,
    )

    amounts = np.where(
        labels == 1,
        rng.lognormal(mean=10.4, sigma=0.85, size=n),
        rng.lognormal(mean=7.5, sigma=0.85, size=n),
    )

    amounts = np.clip(amounts, 50, 250_000)
    amounts = np.round(amounts, 2)

    start_date = pd.Timestamp("2025-01-01")
    date_offsets = rng.integers(0, 365, size=n)

    invoice_dates = (
        start_date + pd.to_timedelta(date_offsets, unit="D")
    )

    payment_terms = rng.choice(
        [15, 30, 45, 60],
        size=n,
        p=[0.15, 0.55, 0.20, 0.10],
    )

    due_dates = invoice_dates + pd.to_timedelta(
        payment_terms,
        unit="D",
    )

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

    processing_days = rng.integers(0, 8, size=n)

    invoice_df = pd.DataFrame({
        "invoice_id": [
            f"EXT-INV-{i:06d}" for i in range(1, n + 1)
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


def validate_external_dataset(df: pd.DataFrame) -> None:
    """Validate the generated external dataset before saving."""

    if len(df) != TOTAL_RECORDS:
        raise ValueError("Unexpected number of records.")

    if len(df.columns) != 12:
        raise ValueError("Unexpected number of columns.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if df.duplicated().any():
        raise ValueError("Dataset contains duplicate rows.")

    if df["invoice_id"].duplicated().any():
        raise ValueError("Duplicate invoice IDs detected.")

    if not df["is_fraud"].isin([0, 1]).all():
        raise ValueError("Invalid target labels detected.")

    if int(df["is_fraud"].sum()) != FRAUD_COUNT:
        raise ValueError("Unexpected fraud count.")

    if not df["amount"].between(50, 250_000).all():
        raise ValueError("Invalid invoice amounts detected.")

    invoice_dates = pd.to_datetime(df["invoice_date"])
    due_dates = pd.to_datetime(df["due_date"])

    if (due_dates < invoice_dates).any():
        raise ValueError("Due date precedes invoice date.")

    actual_terms = (due_dates - invoice_dates).dt.days

    if not actual_terms.equals(
        df["payment_terms_days"].astype("int64")
    ):
        raise ValueError("Payment terms consistency check failed.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating independent external test dataset...")

    df = generate_external_dataset()

    validate_external_dataset(df)

    # Refuse to overwrite an existing external evaluation dataset.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT_FILE}"
        )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEXTERNAL TEST DATASET GENERATED")
    print("=" * 50)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Total records: {len(df)}")
    print(f"Fraud cases: {int(df['is_fraud'].sum())}")
    print(f"Legitimate cases: {int((df['is_fraud'] == 0).sum())}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")
    print(f"Columns: {len(df.columns)}")
    print(f"Validation: PASSED")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()
