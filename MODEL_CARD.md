# LedgerPro Invoice Anomaly Detection — Model Card

## Model Overview

| Item | Details |
|---|---|
| Model name | LedgerPro threshold-optimized invoice anomaly detector |
| Model version | `1.0.0` |
| Task | Binary fraud/anomaly classification (`is_fraud`) |
| Positive class | `1` (Fraudulent) |
| Serving threshold | `0.10` |
| Decision rule | `fraud_probability >= 0.10` |
| Classifier | `sklearn.ensemble.GradientBoostingClassifier` |
| Model artifact | `models/threshold_optimized_pipeline.joblib` |
| Training date | **Unavailable** — the repository has no verified original training date |

> **Important:** This model was developed and evaluated using synthetic invoice data. Its reported metrics demonstrate behavior on the project datasets only and do **not** prove real-world fraud-detection performance.

## Intended Use

### Purpose

The model estimates the probability that an invoice is fraudulent or anomalous. The live API returns that probability and applies the frozen threshold of `0.10` to create a manual-review flag.

### Intended users

- LedgerPro invoice-processing staff
- Finance or compliance reviewers
- Technical staff operating the API

### Intended workflow

The model is intended to prioritize invoices for human review. It must not automatically reject payments or make a final fraud determination.

## Out of Scope and Unsupported Uses

- Conclusive fraud or criminal wrongdoing determination
- Automatic invoice rejection or account suspension
- Real-world credit, legal, employment, or regulatory decisions
- Processing invoices outside the API validation schema
- Replacing independent vendor, bank, accounting, or sanctions checks
- Use on materially different real-world populations without retraining and validation

## Dataset

The development dataset is synthetic and was created for this project.

- Total records: 10,000
- Legitimate records: 9,900
- Fraudulent records: 100
- Fraud rate: 1%
- Random seed for the original Task 2 external test: `20261111` (for the independently generated synthetic test)
- Stratified final training split: 80% training / 20% testing
- Training records: 8,000 (7,920 legitimate, 80 fraudulent)
- Held-out test records: 2,000 (1,980 legitimate, 20 fraudulent)

The cleaned modeling process excludes `invoice_id`, `processing_days`, and the target `is_fraud` from model inputs. `invoice_id` is used only for traceability, while `processing_days` may not be available at the initial prediction point.

## Preprocessing and Features

The complete pipeline is used for inference; preprocessing is stored in the artifact and is applied automatically.

### Required API inputs

- `amount`
- `payment_terms_days`
- `line_item_count`
- `invoice_date`
- `due_date`

### Engineered features

- `amount`
- `payment_terms_days`
- `line_item_count`
- `invoice_year`
- `invoice_month`
- `invoice_day_of_week`
- `due_days_after_invoice`

API validation restricts `amount` to 50–250,000, `line_item_count` to 1–15, payment terms to 15, 30, 45, or 60 days, and requires dates to be consistent with the payment terms.

## Model Architecture

The classifier is a Gradient Boosting model with:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `learning_rate` | 0.10 |
| `max_depth` | 3 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |

Because the data is severely imbalanced, SMOTE is applied inside the pipeline workflow:

| Parameter | Value |
|---|---:|
| Method | SMOTE |
| Sampling strategy | 0.5 |
| `k_neighbors` | 5 |
| Random state | 42 |

No retraining occurs in the API. The service loads the trusted saved pipeline once at startup.

## Evaluation Metrics

Metrics below are taken from existing project reports and configuration. The values describe synthetic test datasets and must not be interpreted as evidence of real-world performance.

### Final held-out synthetic test

Source: `reports/final_model_metrics.json` and `reports/final_model_evaluation.txt`

| Metric | Result |
|---|---:|
| Accuracy | 0.9935 |
| Precision | 0.6842 |
| Recall | 0.6500 |
| F1-score | 0.6667 |
| PR-AUC | 0.7341 |
| ROC-AUC | 0.9802 |

Confusion matrix, with actual rows and predicted columns:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 1,974 | 6 |
| Actual Fraud | 7 | 13 |

### Independently generated synthetic test

Source: `reports/external_test_metrics.json`, generated with seed `20261111`

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
| Actual Legitimate | 9,869 | 31 |
| Actual Fraud | 36 | 64 |

### Historical Task 2 threshold-optimization synthetic evaluation

Source: `reports/threshold_optimized_metrics.json`, dataset seed `20261112`

This evaluation used the historical maximum-F1 threshold of `0.58`, not the current cost-sensitive serving threshold.

| Metric | Result |
|---|---:|
| Accuracy | 0.9952 |
| Precision | 0.7826 |
| Recall | 0.7200 |
| F1-score | 0.7500 |
| PR-AUC | 0.8066 |
| ROC-AUC | 0.9802 |

## Cost-Sensitive Threshold Selection

Task 3 used 5×5 repeated-stratified out-of-fold predictions averaged to one probability per of 10,000 invoices.

Business costs:

- False positive (legitimate invoice incorrectly flagged): **$5**
- False negative (fraudulent invoice missed): **$500**
- Cost formula: `FP × $5 + FN × $500`

Threshold candidates from `0.1` through `0.9` were evaluated. The lowest validation cost was selected with ties resolved by higher recall, then higher precision, then higher threshold.

### Selected threshold

| Item | Result |
|---|---:|
| Selected threshold | `0.10` |
| Validation precision | 0.3812 |
| Validation recall | 0.8500 |
| Validation F1 | 0.5263 |
| Validation false positives | 138 |
| Validation false negatives | 15 |
| Validation total cost | $8,190 |

Source: `reports/cost_sensitive_threshold_decision.json` and `reports/cost_sensitive_threshold_comparison.csv`.

### Separate fresh cost-sensitive synthetic evaluation

Source: `reports/cost_sensitive_threshold_decision.json`, seed `20261117`

| Metric | Result |
|---|---:|
| Records | 10,000 |
| Fraud cases | 100 |
| Legitimate cases | 9,900 |
| False positives | 146 |
| False negatives | 13 |
| Precision | 0.3734 |
| Recall | 0.8700 |
| F1-score | 0.5225 |
| Total cost | $7,230 |

Confusion matrix:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 9,754 | 146 |
| Actual Fraud | 13 | 87 |

This separate evaluation was not used to select the threshold.

## Live API Verification

Both live endpoints were verified on the Render service.

- Health: `GET https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com/health`
- Prediction: `POST https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com/predict`

The verified sample request produced:

- Fraud probability: `0.00021785372778649`
- Threshold: `0.10`
- Fraud flag: `false`
- Prediction: `Legitimate`
- Model version: `1.0.0`

## Limitations, Risks, and Responsible Use

### Data limitations

- The dataset is synthetic, not a collection of LedgerPro's real invoices.
- The data-generation process is not evidence of real-world fraud patterns.
- No verified real-world external validation is available.
- The available training date is unavailable and must not be inferred from file timestamps or Git history.

### Operational limitations

- A lower threshold increases fraud recall but also increases false-positive review workload.
- A false positive means a legitimate invoice may receive unnecessary review.
- A false negative means a fraudulent or anomalous invoice may be missed.
- The probability is a model score, not a calibrated statement of confirmed fraud likelihood.
- The API does not authenticate users and does not currently provide persistent monitoring or audit storage.

### Human-review requirements

- Every flagged invoice requires qualified human review.
- Reviewers should validate vendor identity, invoice amount, supporting records, payment terms, and relevant internal controls.
- A flag does not prove fraud.
- A non-flagged invoice is not guaranteed legitimate and may still require normal review procedures.

### Responsible use

LedgerPro should monitor false positives, false negatives, drift, vendor changes, and shifts in invoice volume or distribution. Real-world deployment should include representative retrospective validation, independent approval procedures, audit logs, access controls, rate limiting, and retraining governance.
