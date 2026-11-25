# LedgerPro API — Client Handover Guide

## Live Service

- Base URL: `https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com`
- Interactive API documentation: `https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com/docs`
- Health endpoint: `GET /health`
- Prediction endpoint: `POST /predict`
- Current model version: `1.0.0`
- Fraud decision threshold: `0.10`
- Training date: **Unavailable** — the repository does not contain verified training-date metadata

The live service was verified on October 5, 2023.

## Check Service Health

Use a browser or API tool to call:

```text
GET https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com/health
```

PowerShell:

```powershell
Invoke-RestMethod -Uri "https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com/health" -Method Get
```

Verified response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "last_training_date": "unknown",
  "threshold": 0.1
}
```

Interpretation:

- `status: "healthy"` means the API is ready.
- `model_loaded: true` means the trusted pipeline is loaded.
- `threshold: 0.1` means a probability of 0.10 or higher creates a fraud flag.
- `last_training_date: "unknown"` is intentional because no verified training date exists.

If the status is unhealthy or a 503 response is returned, do not submit invoices again repeatedly. Record the error and contact technical support.

## Send an Invoice for Prediction

Send a JSON `POST` request to:

```text
POST https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com/predict
```

The request must contain these fields:

| Field | Description | Example / rule |
|---|---|---|
| `amount` | Invoice amount | Numeric, 50.0 to 250000.0 |
| `payment_terms_days` | Payment terms | Integer: 15, 30, 45, or 60 |
| `line_item_count` | Number of invoice lines | Integer 1 to 15 |
| `invoice_date` | Invoice date | ISO date during 2025 |
| `due_date` | Due date | ISO date, after or equal to invoice date |

The due date must equal `invoice_date + payment_terms_days`.

### Verified example request

```json
{
  "amount": 1000.0,
  "payment_terms_days": 30,
  "line_item_count": 2,
  "invoice_date": "2025-01-01",
  "due_date": "2025-01-31"
}
```

PowerShell:

```powershell
$body = @{ 
  amount = 1000.0
  payment_terms_days = 30
  line_item_count = 2
  invoice_date = "2025-01-01"
  due_date = "2025-01-31"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://ledgerpro-invoice-anomaly-detection-8l6u.onrender.com/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Verified example response

```json
{
  "fraud_probability": 0.00021785372778649,
  "threshold": 0.1,
  "fraud_flag": false,
  "prediction": "Legitimate",
  "model_version": "1.0.0"
}
```

This exact response was obtained from the live Render service for the example request above.

## Fraud Probability and Threshold

The `fraud_probability` is a number from `0.0` to `1.0`:

- `0.10` means 10% model score.
- `0.70` means 70% model score.

The API sets:

```text
fraud_flag = fraud_probability >= 0.10
```

- Probability below `0.10`: `fraud_flag: false`, prediction `Legitimate`.
- Probability from `0.10` upward: `fraud_flag: true`, prediction `Fraudulent`.

A `Fraudulent` response means the invoice is marked for manual review. It does **not** conclusively prove that fraud occurred. Conversely, `Legitimate` does not guarantee that the invoice is fraud-free.

Staff must not:

- Automatically reject or delay a payment based only on the API.
- Conclude that an invoice is criminally fraudulent.
- Ignore normal invoice-control and approval procedures.

## Staff Workflow for a Flagged Invoice

1. **Do not make an automatic final decision.** Treat `fraud_flag: true` as a review alert.
2. **Record the response.** Preserve the invoice identifier, amount, probability, threshold, and model version in the normal review log.
3. **Verify the invoice.** Check vendor identity, tax information, banking details, purchase order, goods or services received, authorization, and supporting documents.
4. **Investigate warning signs.** Compare duplicates, unusual terms, changed payment details, unexpected vendors, and policy exceptions.
5. **Use independent approval.** Require review by authorized finance or compliance staff according to LedgerPro procedures.
6. **Document the outcome.** Record whether the invoice was approved, held, corrected, escalated, or rejected.
7. **Do not feed outcome labels back automatically.** Retraining or threshold changes require controlled project governance.

## Human Review Requirements

Human review is required for every flagged invoice. The model is trained on synthetic data and is not a substitute for real-world due diligence. Staff should assess supporting evidence and follow applicable policy, segregation-of-duties, and fraud-investigation procedures.

A flagged invoice may be a false positive. A legitimate invoice may be flagged because of unusual but valid characteristics.

## Error Handling

### HTTP 422 — Invalid request

Example causes:

- Missing field
- Extra field not allowed by the API
- Invalid amount or line-item count
- Unsupported payment terms
- Invalid date format or inconsistent dates

The response contains `detail: "Invalid request"` and a list of validation errors.

Correct the request and retry. Do not bypass validation or send extra fields.

### HTTP 413 — Request too large

The request body exceeds 64 KB.

Send one invoice per request and do not include large attachments or unrelated data.

### HTTP 503 — Service unavailable

Possible causes:

- Model artifact could not be loaded
- Model prediction failed
- Invalid probability returned by the model
- Live service is unavailable or restarting

The response may contain:

```json
{
  "detail": "Prediction service is unavailable"
}
```

Do not infer a legitimate or fraudulent result from a 503. Preserve the invoice in the normal pending queue and retry according to operational procedure.

### Timeout or connection failure

Possible causes:

- Render service is starting or sleeping
- Network connectivity is unavailable
- The service is temporarily unavailable

Check `/health` first. If it remains unavailable, contact technical support and do not replace the response with an assumed model result.

## Troubleshooting Checklist

1. Confirm that the URL is exact and uses `https`.
2. Confirm that the request is `GET` for `/health` or `POST` for `/predict`.
3. Confirm that the request body is valid JSON.
4. Remove extra fields.
5. Check that dates use `YYYY-MM-DD`.
6. Confirm that the due date is exactly the invoice date plus the payment terms.
7. Verify that amount, terms, and line-count values meet the documented ranges.
8. Check `/health` before sending a prediction.
9. Save the HTTP status and response for support investigations.

## Limitations

- The model is based on synthetic training and test data.
- There is no verified real-world performance measurement in the repository.
- The API accepts limited invoice fields.
- The model can produce false positives and false negatives.
- The training date is unknown.
- The public service has no API authentication or persistent audit monitoring in the current implementation.

These limitations must be considered by LedgerPro staff and system administrators.

## Support Information

For application-level data or review issues, use LedgerPro's internal fraud-review and finance escalation process. For API outages or code errors, provide:

- Request time
- Invoice identifier
- HTTP status code
- Sanitized request fields
- Response body
- Model version, if returned

Do not include unnecessary sensitive account or identity information in support messages.
