
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "invoices_raw.csv"

EXPECTED_ROWS = 10_000
EXPECTED_FRAUD = 100
EXPECTED_COLUMNS = 12


def validate_dataset() -> bool:
    """Perform data quality checks on the generated invoice dataset."""

    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found: {INPUT_FILE}")
        return False

    df = pd.read_csv(INPUT_FILE)

    print("\n" + "=" * 55)
    print("LEDGERPRO - DATA QUALITY VALIDATION")
    print("=" * 55)

    checks = {}

    # 1. Dataset dimensions
    checks["Expected 10,000 rows"] = len(df) == EXPECTED_ROWS
    checks["Expected 12 columns"] = len(df.columns) == EXPECTED_COLUMNS

    # 2. Missing values
    missing_count = int(df.isnull().sum().sum())
    checks["No missing values"] = missing_count == 0

    # 3. Duplicate records
    duplicate_rows = int(df.duplicated().sum())
    checks["No duplicate rows"] = duplicate_rows == 0

    # 4. Unique invoice IDs
    duplicate_ids = int(df["invoice_id"].duplicated().sum())
    checks["Unique invoice IDs"] = duplicate_ids == 0

    # 5. Invoice amount validation
    invalid_amounts = int(
        ((df["amount"] <= 0) | (df["amount"] > 250_000)).sum()
    )
    checks["Valid invoice amounts"] = invalid_amounts == 0

    # 6. Date validation
    invoice_dates = pd.to_datetime(
        df["invoice_date"], errors="coerce"
    )
    due_dates = pd.to_datetime(
        df["due_date"], errors="coerce"
    )

    invalid_dates = int(
        (invoice_dates.isna() | due_dates.isna()).sum()
    )
    checks["Valid invoice and due dates"] = invalid_dates == 0

    # 7. Due date must not precede invoice date
    invalid_due_dates = int(
        (due_dates < invoice_dates).sum()
    )
    checks["Due date is valid"] = invalid_due_dates == 0

    # 8. Payment terms consistency
    actual_terms = (due_dates - invoice_dates).dt.days
    inconsistent_terms = int(
        (actual_terms != df["payment_terms_days"]).sum()
    )
    checks["Payment terms are consistent"] = inconsistent_terms == 0

    # 9. Target label validation
    valid_labels = set(df["is_fraud"].unique()) <= {0, 1}
    checks["Valid fraud labels (0 and 1)"] = valid_labels

    fraud_count = int((df["is_fraud"] == 1).sum())
    legitimate_count = int((df["is_fraud"] == 0).sum())

    checks["Exactly 100 fraudulent invoices"] = (
        fraud_count == EXPECTED_FRAUD
    )
    checks["Exactly 9,900 legitimate invoices"] = (
        legitimate_count == 9_900
    )

    # Display results
    print("\nVALIDATION RESULTS")
    print("-" * 55)

    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check_name}")

    print("\nDATASET SUMMARY")
    print("-" * 55)
    print(f"Total records       : {len(df)}")
    print(f"Total columns       : {len(df.columns)}")
    print(f"Missing values      : {missing_count}")
    print(f"Duplicate rows      : {duplicate_rows}")
    print(f"Duplicate IDs       : {duplicate_ids}")
    print(f"Invalid amounts     : {invalid_amounts}")
    print(f"Invalid dates       : {invalid_dates}")
    print(f"Fraudulent invoices : {fraud_count}")
    print(f"Legitimate invoices : {legitimate_count}")

    fraud_rate = fraud_count / len(df) * 100
    print(f"Fraud rate          : {fraud_rate:.2f}%")

    passed_count = sum(checks.values())
    total_checks = len(checks)

    print("\n" + "=" * 55)
    print(f"PASSED: {passed_count}/{total_checks}")

    if all(checks.values()):
        print("RESULT: ALL DATA QUALITY CHECKS PASSED!")
    else:
        print("RESULT: VALIDATION FAILED. INVESTIGATION REQUIRED.")

    print("=" * 55)

    return all(checks.values())


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
