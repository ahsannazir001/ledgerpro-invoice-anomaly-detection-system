"""Select and evaluate a fraud threshold using business costs.

Task 3 consumes Task 2's per-invoice mean OOF fraud probabilities. It does not
fit or replace the Task 2 model and never uses the external evaluation labels
to choose the threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import generate_external_test as synthetic_generator
from train_final_model import FEATURES, TARGET


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
EXTERNAL_DATA_DIR = ROOT / "data" / "external_test"
OOF_PATH = REPORT_DIR / "threshold_optimization_oof_predictions.csv"
PIPELINE_PATH = MODEL_DIR / "threshold_optimized_pipeline.joblib"
CONFIG_PATH = MODEL_DIR / "threshold_optimized_config.json"

FALSE_POSITIVE_COST = 5.0
FALSE_NEGATIVE_COST = 500.0
THRESHOLD_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FRESH_EVALUATION_SEED = 20261117


def load_oof_predictions(path: Path = OOF_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one aggregated OOF probability per original invoice."""
    if not path.is_file():
        raise FileNotFoundError(f"OOF prediction file not found: {path}")
    frame = pd.read_csv(path)
    required = {"row_index", "actual", "mean_oof_probability", "repeat_predictions_averaged"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")
    if frame.empty or frame["row_index"].isna().any() or frame["row_index"].duplicated().any():
        raise ValueError("OOF predictions must contain one unique row per original invoice.")
    labels = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    probabilities = pd.to_numeric(frame["mean_oof_probability"], errors="coerce").to_numpy(dtype=float)
    repeats = pd.to_numeric(frame["repeat_predictions_averaged"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("OOF actual labels must be finite binary values.")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("OOF fraud probabilities must be finite values between 0 and 1.")
    if not np.isfinite(repeats).all() or not np.equal(repeats, 5).all():
        raise ValueError("Expected five repeated-fold predictions averaged per invoice.")
    labels = labels.astype(int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("OOF labels must contain both legitimate and fraud classes.")
    return labels, probabilities


def evaluate_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Calculate confusion counts, quality metrics, and business cost."""
    rows = []
    for threshold in THRESHOLD_CANDIDATES:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        fp_cost = int(fp) * FALSE_POSITIVE_COST
        fn_cost = int(fn) * FALSE_NEGATIVE_COST
        rows.append({
            "threshold": threshold,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "fp_cost": fp_cost, "fn_cost": fn_cost,
            "total_cost": fp_cost + fn_cost,
        })
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> pd.Series:
    """Minimize cost; ties prefer recall, then precision, then higher threshold."""
    return table.sort_values(
        ["total_cost", "recall", "precision", "threshold"],
        ascending=[True, False, False, False], kind="stable",
    ).iloc[0]


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    data = frame.copy()
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="raise")
    data["due_date"] = pd.to_datetime(data["due_date"], errors="raise")
    data["invoice_year"] = data["invoice_date"].dt.year
    data["invoice_month"] = data["invoice_date"].dt.month
    data["invoice_day_of_week"] = data["invoice_date"].dt.dayofweek
    data["due_days_after_invoice"] = (data["due_date"] - data["invoice_date"]).dt.days
    if TARGET not in data:
        raise ValueError(f"Evaluation data is missing target column {TARGET!r}.")
    return data[FEATURES].copy(), data[TARGET].astype(int).to_numpy()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    y_true, probabilities = load_oof_predictions()
    table = evaluate_thresholds(y_true, probabilities)
    selected = select_threshold(table)
    threshold = float(selected["threshold"])
    table.to_csv(REPORT_DIR / "cost_sensitive_threshold_comparison.csv", index=False)

    decision = {
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "threshold_candidates": THRESHOLD_CANDIDATES,
        "selected_threshold": threshold,
        "minimum_total_cost": float(selected["total_cost"]),
        "selected_fp_count": int(selected["fp"]),
        "selected_fn_count": int(selected["fn"]),
        "selected_precision": float(selected["precision"]),
        "selected_recall": float(selected["recall"]),
        "selected_f1": float(selected["f1"]),
        "validation_dataset_size": int(len(y_true)),
        "validation_fraud_count": int(y_true.sum()),
        "validation_legitimate_count": int((y_true == 0).sum()),
        "source_prediction_file": str(OOF_PATH.relative_to(ROOT)).replace("\\", "/"),
        "tie_breaking_policy": "Lowest total cost; ties prefer higher recall, then higher precision, then higher threshold.",
        "decision_rule": "fraud_probability >= threshold",
    }
    (REPORT_DIR / "cost_sensitive_threshold_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    # Apply the selected threshold to the existing Task 2 fitted pipeline on
    # a new generated evaluation set. The labels here never affect selection.
    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(f"Saved Task 2 pipeline not found: {PIPELINE_PATH}")
    synthetic_generator.RANDOM_SEED = FRESH_EVALUATION_SEED
    evaluation = synthetic_generator.generate_external_dataset()
    synthetic_generator.validate_external_dataset(evaluation)
    evaluation_path = EXTERNAL_DATA_DIR / f"cost_sensitive_evaluation_seed_{FRESH_EVALUATION_SEED}.csv"
    evaluation.to_csv(evaluation_path, index=False)
    X_eval, y_eval = prepare_features(evaluation)
    pipeline = joblib.load(PIPELINE_PATH)
    classes = np.asarray(getattr(pipeline, "classes_", []))
    if classes.tolist() != [0, 1]:
        raise ValueError("Saved pipeline classes_ must be exactly [0, 1].")
    probs_eval = np.asarray(pipeline.predict_proba(X_eval))[:, 1]
    if not np.isfinite(probs_eval).all() or not ((probs_eval >= 0) & (probs_eval <= 1)).all():
        raise ValueError("Evaluation fraud probabilities are invalid.")
    pred_eval = (probs_eval >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_eval, pred_eval, labels=[0, 1]).ravel()
    evaluation_cost = int(fp) * FALSE_POSITIVE_COST + int(fn) * FALSE_NEGATIVE_COST
    evaluation_summary = {
        "seed": FRESH_EVALUATION_SEED,
        "records": int(len(y_eval)),
        "fraud_count": int(y_eval.sum()),
        "legitimate_count": int((y_eval == 0).sum()),
        "false_positives": int(fp), "false_negatives": int(fn),
        "total_cost": float(evaluation_cost),
        "precision": float(precision_score(y_eval, pred_eval, zero_division=0)),
        "recall": float(recall_score(y_eval, pred_eval, zero_division=0)),
        "f1": float(f1_score(y_eval, pred_eval, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "dataset": str(evaluation_path.relative_to(ROOT)).replace("\\", "/"),
    }
    decision["fresh_evaluation"] = evaluation_summary
    # Keep the authoritative decision JSON complete with the independent evaluation.
    (REPORT_DIR / "cost_sensitive_threshold_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    # Update only inference decision metadata; historical Task 2 metric reports
    # and model binaries are not changed.
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["threshold"] = threshold
    config["threshold_selection"] = {
        "method": "Minimum expected business cost across the Task 3 threshold grid",
        "validation": "Per-invoice mean fraud probabilities from Task 2 RepeatedStratifiedKFold 5x5 OOF predictions",
        "selected_threshold": threshold,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "decision_report": "reports/cost_sensitive_threshold_decision.json",
        "tie_breaking_policy": decision["tie_breaking_policy"],
    }
    config["cost_sensitive_evaluation"] = evaluation_summary
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    report = [
        "LEDGERPRO TASK 3: COST-SENSITIVE THRESHOLD DECISION",
        "=" * 62, "", "BUSINESS ASSUMPTIONS",
        f"False positive: ${FALSE_POSITIVE_COST:.2f} per legitimate invoice reviewed.",
        f"False negative: ${FALSE_NEGATIVE_COST:.2f} per missed fraudulent invoice.",
        "Total cost = FP * false-positive cost + FN * false-negative cost.", "",
        "VALIDATION AND DECISION",
        f"Source: {decision['source_prediction_file']}",
        f"Rows counted once: {len(y_true)}; fraud: {int(y_true.sum())}; legitimate: {int((y_true == 0).sum())}.",
        "OOF predictions are already averaged across five repeats and have unique row_index values.",
        "Tie-break: minimum cost, then higher recall, higher precision, then higher threshold.",
        f"Selected threshold: {threshold:.1f}",
        f"Validation FP: {int(selected['fp'])}; FN: {int(selected['fn'])}; total cost: ${selected['total_cost']:.2f}.",
        f"Precision: {selected['precision']:.6f}; recall: {selected['recall']:.6f}; F1: {selected['f1']:.6f}.", "",
        "THRESHOLD COMPARISON",
        table.to_string(index=False, float_format=lambda value: f"{value:.4f}"), "",
        "FRESH SYNTHETIC EVALUATION (NOT USED FOR THRESHOLD SELECTION)",
        f"Dataset: {evaluation_summary['dataset']} (seed {FRESH_EVALUATION_SEED})",
        f"Threshold: {threshold:.1f}; FP: {fp}; FN: {fn}; total cost: ${evaluation_cost:.2f}.",
        f"Precision: {evaluation_summary['precision']:.6f}; recall: {evaluation_summary['recall']:.6f}; F1: {evaluation_summary['f1']:.6f}.",
        "Confusion matrix (actual rows [legitimate, fraud], predicted columns [legitimate, fraud]):",
        str(evaluation_summary["confusion_matrix"]), "",
        "The saved Task 2 pipeline was loaded unchanged; only the inference threshold configuration was updated.",
    ]
    (REPORT_DIR / "cost_sensitive_threshold_decision.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:18]))
    print("\nWrote Task 3 reports and updated inference threshold configuration.")


if __name__ == "__main__":
    main()
