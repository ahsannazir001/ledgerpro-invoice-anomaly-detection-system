"""Select a fraud threshold from repeated out-of-fold predictions.

The original final-model scripts and their artifacts are left untouched.
This experiment uses repeated stratified CV on the processed training dataset,
then evaluates the frozen threshold once on a new synthetic dataset.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

import generate_external_test as synthetic_generator
from train_final_model import (
    FEATURES,
    RANDOM_STATE,
    TARGET,
    build_model_pipeline,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
EXTERNAL_DATA_DIR = ROOT / "data" / "external_test"

N_SPLITS = 5
N_REPEATS = 5
EXTERNAL_RANDOM_SEED = 20261112
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the same feature engineering as the final training script."""
    frame = df.copy()
    frame["invoice_date"] = pd.to_datetime(
        frame["invoice_date"], errors="raise"
    )
    frame["due_date"] = pd.to_datetime(
        frame["due_date"], errors="raise"
    )
    frame["invoice_year"] = frame["invoice_date"].dt.year
    frame["invoice_month"] = frame["invoice_date"].dt.month
    frame["invoice_day_of_week"] = frame["invoice_date"].dt.dayofweek
    frame["due_days_after_invoice"] = (
        frame["due_date"] - frame["invoice_date"]
    ).dt.days
    return frame[FEATURES].copy(), frame[TARGET].astype(int).copy()


def generate_fresh_evaluation_data() -> pd.DataFrame:
    """Generate data in memory without touching the prior external dataset."""
    synthetic_generator.RANDOM_SEED = EXTERNAL_RANDOM_SEED
    fresh = synthetic_generator.generate_external_dataset()
    synthetic_generator.validate_external_dataset(fresh)
    return fresh


def evaluate_thresholds(
    y_true: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(y_true, predictions, zero_division=0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed dataset for repeated out-of-fold validation...")
    training_df, X, y = load_dataset()
    if y.value_counts().min() < N_SPLITS:
        raise ValueError("Not enough minority samples for repeated stratified CV.")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    probability_sum = np.zeros(len(X), dtype=float)
    prediction_count = np.zeros(len(X), dtype=int)

    total_fits = N_SPLITS * N_REPEATS
    print(f"Dataset rows: {len(X)}; fraud cases: {int(y.sum())}")
    print(f"Generating OOF predictions: {N_SPLITS} folds x {N_REPEATS} repeats")

    for split_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        model = build_model_pipeline()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
        probability_sum[valid_idx] += fold_probabilities
        prediction_count[valid_idx] += 1
        if split_number % N_SPLITS == 0:
            print(f"Completed {split_number}/{total_fits} fold fits")

    if not np.all(prediction_count == N_REPEATS):
        raise RuntimeError("Each sample must receive one OOF prediction per repeat.")

    mean_oof_probabilities = probability_sum / prediction_count
    threshold_table = evaluate_thresholds(
        y.to_numpy(), mean_oof_probabilities
    )
    # Deterministic tie handling: maximize F1, then precision, then choose
    # the highest threshold among remaining ties.
    selected = threshold_table.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    frozen_threshold = float(selected["threshold"])

    threshold_csv = REPORT_DIR / "threshold_optimization_validation.csv"
    threshold_table.to_csv(threshold_csv, index=False)
    oof_predictions = (mean_oof_probabilities >= frozen_threshold).astype(int)
    oof_predictions_path = REPORT_DIR / "threshold_optimization_oof_predictions.csv"
    pd.DataFrame({
        "row_index": training_df.index,
        "actual": y.to_numpy(),
        "mean_oof_probability": mean_oof_probabilities,
        "prediction_at_frozen_threshold": oof_predictions,
        "repeat_predictions_averaged": prediction_count,
    }).to_csv(oof_predictions_path, index=False)

    print("\nThreshold selected from validation predictions only:")
    print(f"Threshold: {frozen_threshold:.2f}")
    print(
        f"OOF precision={selected['precision']:.4f}, "
        f"recall={selected['recall']:.4f}, F1={selected['f1']:.4f}"
    )

    # Train the exact configured final pipeline on all processed records.
    print("\nTraining the configured final pipeline on the processed dataset...")
    final_pipeline = build_model_pipeline()
    final_pipeline.fit(X, y)

    # Generate a new seed-controlled synthetic test set in memory. The old
    # external dataset and its reports are never opened or modified here.
    print(f"Generating fresh synthetic evaluation data (seed {EXTERNAL_RANDOM_SEED})...")
    fresh_df = generate_fresh_evaluation_data()
    fresh_X, fresh_y = prepare_features(fresh_df)
    fresh_data_path = (
        EXTERNAL_DATA_DIR
        / f"threshold_optimized_external_seed_{EXTERNAL_RANDOM_SEED}.csv"
    )
    fresh_df.to_csv(fresh_data_path, index=False)

    print("Evaluating frozen threshold once on fresh synthetic data...")
    probabilities = final_pipeline.predict_proba(fresh_X)[:, 1]
    predictions = (probabilities >= frozen_threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(fresh_y, predictions)),
        "precision": float(precision_score(fresh_y, predictions, zero_division=0)),
        "recall": float(recall_score(fresh_y, predictions, zero_division=0)),
        "f1": float(f1_score(fresh_y, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(fresh_y, probabilities)),
        "roc_auc": float(roc_auc_score(fresh_y, probabilities)),
    }
    matrix = confusion_matrix(fresh_y, predictions, labels=[0, 1])
    report = classification_report(
        fresh_y,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )

    # New artifact names preserve all existing Task 2 artifacts.
    pipeline_path = MODEL_DIR / "threshold_optimized_pipeline.joblib"
    classifier_path = MODEL_DIR / "threshold_optimized_model.joblib"
    preprocessing_path = MODEL_DIR / "threshold_optimized_preprocessing_pipeline.joblib"
    joblib.dump(final_pipeline, pipeline_path)
    joblib.dump(final_pipeline.named_steps["classifier"], classifier_path)
    joblib.dump(final_pipeline.named_steps["preprocessing"], preprocessing_path)

    metrics["selected_threshold"] = frozen_threshold
    metrics["threshold_selection_method"] = (
        "Maximum F1 on mean per-sample probabilities from 5x5 repeated stratified OOF predictions"
    )
    metrics["external_random_seed"] = EXTERNAL_RANDOM_SEED
    metrics["external_records"] = int(len(fresh_df))
    metrics["external_fraud_cases"] = int(fresh_y.sum())
    metrics["cv_folds"] = N_SPLITS
    metrics["cv_repeats"] = N_REPEATS

    metrics_path = REPORT_DIR / "threshold_optimized_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prediction_path = REPORT_DIR / "threshold_optimized_external_predictions.csv"
    pd.DataFrame({
        "invoice_id": fresh_df["invoice_id"],
        "actual": fresh_y,
        "predicted": predictions,
        "fraud_probability": probabilities,
    }).to_csv(prediction_path, index=False)

    report_path = REPORT_DIR / "threshold_optimized_evaluation.txt"
    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("LEDGERPRO VALIDATION-SELECTED THRESHOLD EXPERIMENT\n")
        report_file.write("=" * 68 + "\n\n")
        report_file.write("VALIDATION METHODOLOGY\n")
        report_file.write(
            f"RepeatedStratifiedKFold: {N_SPLITS} folds x {N_REPEATS} repeats\n"
        )
        report_file.write("Preprocessing and SMOTE fitted independently in each training fold.\n")
        report_file.write("Per-sample probabilities averaged across the 5 validation appearances.\n")
        report_file.write("Threshold selected by maximum validation OOF F1; tie-break: precision, then higher threshold.\n")
        report_file.write("Previously evaluated held-out and external datasets were not used for selection.\n")
        report_file.write(f"Frozen threshold: {frozen_threshold:.2f}\n\n")
        report_file.write("VALIDATION OOF METRICS AT FROZEN THRESHOLD\n")
        report_file.write(
            f"Precision: {selected['precision']:.6f}\n"
            f"Recall: {selected['recall']:.6f}\n"
            f"F1: {selected['f1']:.6f}\n\n"
        )
        report_file.write("FRESH SYNTHETIC EVALUATION\n")
        report_file.write(f"Random seed: {EXTERNAL_RANDOM_SEED}\n")
        report_file.write(f"Records: {len(fresh_df)}\n")
        report_file.write(f"Fraud cases: {int(fresh_y.sum())}\n")
        report_file.write("Model: configured final GradientBoostingClassifier with SMOTE pipeline, refit on all processed records\n")
        report_file.write("Metrics\n")
        for name in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"):
            report_file.write(f"{name}: {metrics[name]:.6f}\n")
        report_file.write("\nConfusion matrix (actual rows, predicted columns)\n")
        report_file.write(f"{matrix}\n\nClassification report\n{report}\n")
        report_file.write("\nTHRESHOLD VALIDATION CURVE\n")
        report_file.write(threshold_table.to_string(index=False))
        report_file.write("\n")

    print("\nFRESH SYNTHETIC EVALUATION RESULTS")
    print("=" * 55)
    print(f"Frozen threshold: {frozen_threshold:.2f}")
    for name in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"):
        print(f"{name.upper():<12}: {metrics[name]:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print("\nNew outputs:")
    for path in (
        threshold_csv,
        oof_predictions_path,
        fresh_data_path,
        pipeline_path,
        classifier_path,
        preprocessing_path,
        metrics_path,
        prediction_path,
        report_path,
    ):
        print(path)
    print(
        "\nF1 TARGET "
        + ("ACHIEVED" if metrics["f1"] > 0.70 else "NOT ACHIEVED")
        + " on the fresh synthetic evaluation dataset."
    )


if __name__ == "__main__":
    main()
