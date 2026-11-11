# LedgerPro Invoice Anomaly Detection System

## Project Overview

LedgerPro Invoice Anomaly Detection System is a Machine Learning project designed to identify potentially fraudulent invoices and flag them for manual review.

The project uses synthetic invoice datasets containing legitimate and fraudulent records.

The project is divided into multiple tasks, covering dataset preparation, data leakage detection, baseline model training, imbalance handling, model evaluation, and cost-sensitive threshold optimization.

The objective is to build a reliable invoice fraud detection workflow while maintaining proper data validation and evaluation practices.

---

# Task 1: Dataset Preparation and Data Leakage Detection

## Dataset Specifications

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

## Technologies Used

- Python 3.10.2
- NumPy 1.26.4
- Pandas 2.2.3
- Matplotlib 3.9.4
- Seaborn 0.13.2
- Scikit-learn 1.5.2
- imbalanced-learn 0.12.4
- Joblib 1.4.2

## Environment Requirements

- Python 3.10.2
- pip
- Git

### Dependency Installation

All project dependencies are pinned in `requirements.txt`.

Install them using:

```bash
python -m pip install -r requirements.txt
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

A fixed random seed of 42 is used to make dataset generation reproducible.

Fraudulent invoices are generated with a different amount distribution to introduce a predictive signal.

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

Data leakage occurs when a model receives information that would not legitimately be available at prediction time, or information that directly reveals the target.

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

1. `invoice_id` is retained in the cleaned dataset for traceability but excluded from model inputs.
2. `processing_days` is excluded because it may not be available when an invoice first arrives.
3. `is_fraud` is preserved as the supervised learning target and must never be included in input features.
4. No direct target-leakage feature was identified among the reviewed generated columns.

## Statistical Feature Analysis

Numeric feature correlations with the fraud label:

| Feature | Correlation |
|---|---:|
| amount | 0.6798 |
| payment_terms_days | -0.0221 |
| line_item_count | -0.0117 |

The amount feature has a positive statistical association with the fraud label.

This is intentional because the synthetic data generator uses different amount distributions for fraudulent and legitimate invoices.

Correlation does not establish data leakage.

Categorical features were also analyzed using fraud rates by vendor, category, payment method, and country.

Monthly fraud rates were calculated to inspect temporal variation.

---

# Task 2: Train Baseline Model with Imbalance Handling

## Objective

The objective of Task 2 was to train and evaluate machine learning models for invoice fraud detection while addressing the severe class imbalance in the dataset.

The original dataset contains:

- 9,900 legitimate invoices.
- 100 fraudulent invoices.
- Fraud rate: 1%.

Accuracy alone is not sufficient for evaluating this problem because a model can achieve high accuracy while failing to detect fraudulent invoices.

Therefore, precision, recall, F1-score, PR-AUC, and ROC-AUC were considered.

## Data Preparation

The cleaned dataset was used for model development.

The following features were excluded from model inputs:

- `invoice_id`: unique identifier used for traceability.
- `processing_days`: potentially unavailable at the initial invoice prediction stage.
- `is_fraud`: target label.

Date features were derived from invoice and due dates:

- `invoice_year`
- `invoice_month`
- `invoice_day_of_week`
- `due_days_after_invoice`

The selected final feature group contains:

- `amount`
- `payment_terms_days`
- `line_item_count`
- `invoice_year`
- `invoice_month`
- `invoice_day_of_week`
- `due_days_after_invoice`

## Train-Test Split

A stratified 80/20 split was used for the final training experiment.

| Dataset | Legitimate | Fraudulent | Total |
|---|---:|---:|---:|
| Training | 7,920 | 80 | 8,000 |
| Test | 1,980 | 20 | 2,000 |

Random state: 42.

Stratification maintained the class distribution across both sets.

## Imbalance Handling

SMOTE (Synthetic Minority Over-sampling Technique) was used to address class imbalance.

Configuration:

| Parameter | Value |
|---|---|
| Sampling method | SMOTE |
| Sampling strategy | 0.5 |
| k_neighbors | 5 |
| Random state | 42 |

SMOTE was applied inside the training pipeline so that validation and test records were not used to generate synthetic training samples.

## Models and Experiments

The following experiments were conducted:

1. Random Forest with balanced class weights.
2. Random Forest with SMOTE.
3. Gradient Boosting with SMOTE.
4. Gradient Boosting with sample weights.
5. SMOTENC-based experiment.
6. Feature importance analysis.
7. Out-of-fold evaluation.
8. Feature group comparison.
9. Repeated stratified cross-validation.
10. Gradient Boosting hyperparameter comparison.

### Baseline Random Forest

The initial Random Forest model used:

- 300 estimators.
- `class_weight="balanced"`.
- Stratified 80/20 train-test split.

Initial held-out test results:

| Metric | Result |
|---|---:|
| Precision | 0.7500 |
| Recall | 0.1500 |
| F1-score | 0.2500 |
| ROC-AUC | 0.9611 |
| PR-AUC | 0.6822 |

The baseline model achieved high accuracy but had poor fraud recall.

### Feature Selection

Four feature groups were compared using stratified cross-validation:

- Amount only.
- Amount plus categorical features.
- Amount plus other numeric features.
- All features.

The amount plus other numeric feature group achieved a pooled F1-score of 0.6190 in the corresponding five-fold experiment.

Repeated five-fold cross-validation was subsequently performed.

| Metric | Amount Only | Amount + Other Numeric |
|---|---:|---:|
| Mean Fold F1 | 0.4215 | 0.6144 |
| Pooled Precision | 0.2694 | 0.5167 |
| Pooled Recall | 0.9040 | 0.7440 |
| Pooled F1 | 0.4151 | 0.6098 |
| PR-AUC | 0.7518 | 0.6870 |
| ROC-AUC | 0.9795 | 0.9891 |

The amount plus other numeric feature group was selected for further model experiments because it provided a better precision-recall balance at the evaluated threshold.

### Hyperparameter Comparison

Sixteen Gradient Boosting configurations were compared using five-fold stratified cross-validation.

The selected configuration was:

| Parameter | Selected Value |
|---|---:|
| n_estimators | 200 |
| learning_rate | 0.10 |
| max_depth | 3 |
| min_samples_leaf | 5 |
| SMOTE sampling strategy | 0.5 |
| Classification threshold | 0.50 |

Best configuration results:

| Metric | Result |
|---|---:|
| Mean Fold F1 | 0.6570 |
| Pooled F1 | 0.6514 |
| Precision | 0.6017 |
| Recall | 0.7100 |
| PR-AUC | 0.7077 |
| ROC-AUC | 0.9820 |

## Final Model

The selected Gradient Boosting configuration was trained using the stratified training split.

SMOTE was applied only to the training data.

The final model was evaluated on the held-out test split.

### Held-Out Test Results

| Metric | Result |
|---|---:|
| Accuracy | 0.9935 |
| Precision | 0.6842 |
| Recall | 0.6500 |
| F1-score | 0.6667 |
| PR-AUC | 0.7341 |
| ROC-AUC | 0.9802 |

Confusion matrix:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 1974 | 6 |
| Actual Fraud | 7 | 13 |

## Independent Synthetic Test Evaluation

A separate synthetic dataset was generated using the same data-generation distributions and a different random seed.

| Property | Value |
|---|---:|
| Records | 10,000 |
| Legitimate Invoices | 9,900 |
| Fraudulent Invoices | 100 |
| Random Seed | 20261111 |
| Fraud Rate | 1% |

The external test dataset was not used to train the final model or select its hyperparameters.

The previously trained pipeline was loaded and used only for prediction.

### Independent Test Results

| Metric | Result |
|---|---:|
| Accuracy | 0.9933 |
| Precision | 0.6737 |
| Recall | 0.6400 |
| F1-score | 0.6564 |
| PR-AUC | 0.6964 |
| ROC-AUC | 0.9764 |

Confusion matrix:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 9869 | 31 |
| Actual Fraud | 36 | 64 |

The model showed similar performance on the original held-out split and the independently generated synthetic evaluation dataset.

**Important:** This independent evaluation measures generalization to another dataset generated from the same synthetic process. It does not establish performance on real-world financial invoices.

## Task 2 Model Artifacts

The following model artifacts were generated:

| File | Description |
|---|---|
| `models/baseline_model.joblib` | Baseline Random Forest classifier |
| `models/baseline_pipeline.joblib` | Baseline complete pipeline |
| `models/preprocessing_pipeline.joblib` | Baseline preprocessing pipeline |
| `models/final_model.joblib` | Final Gradient Boosting classifier |
| `models/final_pipeline.joblib` | Complete final inference pipeline |
| `models/final_preprocessing_pipeline.joblib` | Fitted final preprocessing pipeline |

The complete `final_pipeline.joblib` should be used for inference on raw feature data because it includes preprocessing and classification.

Only load trusted model artifacts. Joblib files use Python serialization and should not be loaded from untrusted sources.

## Task 2 Reports

Important generated reports include:

| File | Description |
|---|---|
| `reports/baseline_evaluation.txt` | Baseline Random Forest evaluation |
| `reports/model_comparison.csv` | Initial model comparison |
| `reports/gb_weighted_evaluation.csv` | Gradient Boosting sample-weight experiment |
| `reports/smotenc_evaluation.csv` | SMOTENC experiment |
| `reports/feature_importance.csv` | Feature importance analysis |
| `reports/oof_evaluation.txt` | Out-of-fold evaluation |
| `reports/oof_fold_metrics.csv` | Fold-level evaluation metrics |
| `reports/oof_error_analysis.txt` | Out-of-fold error analysis |
| `reports/error_pattern_analysis.txt` | Error pattern analysis |
| `reports/error_rate_comparison.txt` | Error rate comparisons |
| `reports/feature_group_comparison.txt` | Feature group comparison |
| `reports/repeated_cv_comparison.txt` | Repeated cross-validation results |
| `reports/feature_subset_comparison.txt` | Feature subset comparison |
| `reports/gb_hyperparameter_comparison.txt` | Hyperparameter comparison |
| `reports/final_model_evaluation.txt` | Final held-out test evaluation |
| `reports/external_test_evaluation.txt` | Independent synthetic test evaluation |

Additional CSV and JSON files contain detailed predictions, metrics, and analysis results.

## Task 2 Acceptance Status

The implementation, model training, imbalance handling, and independent synthetic evaluation have been completed.

However, the mentor's target F1-score is greater than 0.70.

The independent synthetic test F1-score is 0.6564.

**Therefore, the target has not yet been achieved, and mentor acceptance is not claimed.**

The threshold remains 0.50 in Task 2. Cost-sensitive threshold optimization is reserved for Task 3.

---

# Project Structure

```text
ledgerpro-invoice-anomaly-detection-system/
│
├── data/
│   ├── raw/
│   │   └── invoices_raw.csv
│   ├── processed/
│   │   └── invoices_clean.csv
│   └── external_test/
│       └── invoices_external_test_raw.csv
│
├── models/
│   ├── baseline_model.joblib
│   ├── baseline_pipeline.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── final_model.joblib
│   ├── final_pipeline.joblib
│   └── final_preprocessing_pipeline.joblib
│
├── reports/
│
├── src/
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── detect_leakage.py
│   ├── analyze_features.py
│   ├── train_baseline.py
│   ├── compare.py
│   ├── train_smotenc.py
│   ├── train_gb_weighted.py
│   ├── analyze_model_features.py
│   ├── evaluate_oof.py
│   ├── analyze_oof_errors.py
│   ├── analyze_error_patterns.py
│   ├── compare_error_rates.py
│   ├── compare_feature_groups.py
│   ├── compare_feature_subsets.py
│   ├── validate_feature_selection.py
│   ├── tune_gradient_boosting.py
│   ├── train_final_model.py
│   ├── generate_external_test.py
│   └── evaluate_external_test.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# How to Run

## 1. Create virtual environment

```bash
python -m venv .venv
```

## 2. Activate environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Task 1: Generate and validate dataset

```bash
python src/generate_dataset.py
python src/validate_dataset.py
python src/detect_leakage.py
python src/analyze_features.py
```

## 5. Task 2: Train and evaluate models

```bash
python src/train_baseline.py
python src/compare.py
python src/train_smotenc.py
python src/train_gb_weighted.py
python src/evaluate_oof.py
python src/train_final_model.py
```

## 6. Generate and evaluate independent synthetic test data

```bash
python src/generate_external_test.py
python src/evaluate_external_test.py
```

---

# Limitations

- The datasets are synthetic, not real financial data.
- Fraud patterns are artificially generated.
- The amount feature has an intentionally different distribution for fraudulent and legitimate invoices.
- Statistical associations do not prove causation.
- The dataset does not represent all real-world invoice fraud scenarios.
- The independent test dataset uses the same synthetic generation process with a different random seed.
- The mentor's target F1-score greater than 0.70 has not yet been achieved.
- Real-world performance must be evaluated using representative, independently collected data.

---

# Current Project Status

## Task 1: Dataset Preparation and Data Leakage Detection

- [x] Synthetic dataset generation
- [x] Data quality validation
- [x] Feature leakage audit
- [x] Clean dataset preparation
- [x] Statistical feature analysis
- [x] README documentation

## Task 2: Train Baseline Model with Imbalance Handling

- [x] Baseline Random Forest training
- [x] Class imbalance experiments
- [x] SMOTE implementation
- [x] Gradient Boosting experiments
- [x] Feature group comparison
- [x] Repeated stratified cross-validation
- [x] Hyperparameter comparison
- [x] Final model training
- [x] Held-out test evaluation
- [x] Independent synthetic test evaluation
- [x] Model and report generation
- [x] README documentation
- [ ] Mentor F1 target greater than 0.70
- [ ] Mentor acceptance
