
# LedgerPro Invoice Anomaly Detection System

## Project Overview

LedgerPro Invoice Anomaly Detection System is a Machine
Learning project designed to identify potentially fraudulent
invoices and flag them for manual review.

The project uses a synthetic dataset of 10,000 invoices,
containing 99% legitimate and 1% fraudulent records.

The objective of Task 1 is to generate realistic invoice
data, validate its quality, identify potential data leakage,
and prepare a clean dataset for future ML model training.

## Task 1: Dataset Preparation and Data Leakage Detection

### Dataset Specifications

| Property | Value |
|---|---|
| Total Records | 10,000 |
| Legitimate Invoices | 9,900 |
| Fraudulent Invoices | 100 |
| Fraud Rate | 1% |
| Original Columns | 12 |
| Cleaned Columns | 11 |
| Dataset Type | Synthetic |
| Random Seed | 42 |

### Technologies Used

- Python 3.10.2
- NumPy 1.26.4
- Pandas 2.2.3
- Matplotlib 3.9.4
- Seaborn 0.13.2

## Environment Requirements

- Python 3.10.2
- pip
- Git

### Dependency Installation

All project dependencies are pinned in requirements.txt.

Install them using:

```bash
python -m pip install -r requirements.txt
```

## Project Structure

```text
ledgerpro-invoice-anomaly-detection-system/
│
├── data/
│   ├── raw/
│   │   └── invoices_raw.csv
│   │
│   └── processed/
│       └── invoices_clean.csv
│
├── src/
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── detect_leakage.py
│   └── analyze_features.py
│
├── reports/
│   ├── leakage_report.txt
│   └── feature_analysis.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Features

| Feature | Description |
|---|---|
| invoice_id | Unique invoice identifier |
| vendor_id | Vendor identifier |
| invoice_date | Invoice creation date |
| due_date | Invoice payment due date |
| amount | Invoice amount |
| category | Invoice category |
| payment_method | Payment method |
| country | Vendor/invoice country |
| payment_terms_days | Payment terms |
| line_item_count | Number of invoice line items |
| processing_days | Invoice processing duration |
| is_fraud | Fraud target label (0 or 1) |

## Data Generation

The synthetic dataset is generated using NumPy and Pandas.

The generator creates exactly:

- 9,900 legitimate invoices.
- 100 fraudulent invoices.

A fixed random seed of 42 is used to make
dataset generation reproducible.

Fraudulent invoices are generated with a different
amount distribution to introduce a predictive signal.

## Data Quality Validation

The dataset was checked for:

| Validation Check | Result |
|---|---|
| Expected 10,000 rows | PASS |
| Expected 12 columns | PASS |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate invoice IDs | 0 |
| Invalid amounts | 0 |
| Invalid dates | 0 |
| Invalid due dates | 0 |
| Payment terms consistency | PASS |
| Valid fraud labels | PASS |
| Exactly 100 fraud records | PASS |
| Exactly 9,900 legitimate records | PASS |

**Validation Result: 12/12 checks passed.**

## Data Leakage Analysis

Data leakage occurs when a model receives
information that would not legitimately be
available at prediction time, or information
that directly reveals the target.

The following features were reviewed:

| Feature | Decision | Reason |
|---|---|---|
| invoice_id | Exclude from model inputs | Unique identifier, not predictive |
| vendor_id | Keep | Available invoice information |
| invoice_date | Keep | Available at invoice receipt |
| due_date | Keep | Available at invoice receipt |
| amount | Keep | Invoice amount |
| category | Keep | Invoice category |
| payment_method | Keep | Available invoice information |
| country | Keep | Available invoice information |
| payment_terms_days | Keep | Available at invoice receipt |
| line_item_count | Keep | Invoice characteristic |
| processing_days | Exclude | May be unavailable at initial prediction time |
| is_fraud | Target only | Ground-truth label; never an input feature |

### Leakage Decisions

1. `invoice_id` is retained in the cleaned dataset
   for traceability but excluded from model inputs.

2. `processing_days` is excluded because it may not
   be available when an invoice first arrives.

3. `is_fraud` is preserved as the supervised learning
   target and must never be included in input features.

4. No direct target-leakage feature was identified
   among the reviewed generated columns.

## Statistical Feature Analysis

Numeric feature correlations with the fraud label:

| Feature | Correlation |
|---|---:|
| amount | 0.6798 |
| payment_terms_days | -0.0221 |
| line_item_count | -0.0117 |

The amount feature has a positive statistical
association with the fraud label.

This is intentional because the synthetic data
generator uses different amount distributions
for fraudulent and legitimate invoices.

Correlation does not establish data leakage.

Categorical features were also analyzed using
fraud rates by vendor, category, payment method,
and country.

Monthly fraud rates were calculated to inspect
temporal variation.

## Generated Files

| File | Description |
|---|---|
| invoices_raw.csv | Original synthetic dataset |
| invoices_clean.csv | Cleaned dataset |
| leakage_report.txt | Feature leakage audit |
| feature_analysis.txt | Statistical feature analysis |

## How to Run

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Generate dataset

```bash
python src/generate_dataset.py
```

### 5. Validate dataset

```bash
python src/validate_dataset.py
```

### 6. Detect data leakage

```bash
python src/detect_leakage.py
```

### 7. Analyze features

```bash
python src/analyze_features.py
```

## Limitations

- The dataset is synthetic, not real financial data.
- Fraud patterns are artificially generated.
- The amount feature has an intentionally different
  distribution for fraudulent invoices.
- Statistical associations do not prove causation.
- The dataset does not represent all real-world
  invoice fraud scenarios.
- Real-world performance must be evaluated using
  representative, independently collected data.

## Current Project Status

Task 1: Prepare Invoice Dataset and Detect Data Leakage.

- [x] Synthetic dataset generation
- [x] Data quality validation
- [x] Feature leakage audit
- [x] Clean dataset preparation
- [x] Statistical feature analysis
- [x] README documentation

## Next Task

Task 2: Train Baseline Model with Imbalance Handling.

The next phase will focus on training and evaluating
a baseline fraud detection model using the cleaned
invoice dataset.

---

**Project:** LedgerPro Invoice Anomaly Detection System

**Client:** LedgerPro Accountants

**Technology:** Machine Learning

**Dataset:** 10,000 Synthetic Invoices
