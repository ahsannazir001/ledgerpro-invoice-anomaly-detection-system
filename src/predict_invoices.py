"""Run inference with the frozen threshold-optimized invoice model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = (ROOT / "models").resolve()
EXPECTED_ARTIFACT = "models/threshold_optimized_pipeline.joblib"
DEFAULT_CONFIG_PATH = MODELS_DIR / "threshold_optimized_config.json"
FROZEN_THRESHOLD = 0.58
REQUIRED_COLUMNS = {
    "amount",
    "payment_terms_days",
    "line_item_count",
    "invoice_date",
    "due_date",
}
OPTIONAL_COLUMNS = {
    "invoice_id",
    "vendor_id",
    "category",
    "payment_method",
    "country",
    "processing_days",
    "is_fraud",
}
ENGINEERED_FEATURES = [
    "amount",
    "payment_terms_days",
    "line_item_count",
    "invoice_year",
    "invoice_month",
    "invoice_day_of_week",
    "due_days_after_invoice",
]


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the frozen inference configuration."""
    try:
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError("Inference config is not valid JSON.") from exc
    if not isinstance(config, dict):
        raise ValueError("Inference config must be a JSON object.")

    required_keys = {
        "artifact", "threshold", "decision_rule", "target", "positive_class",
        "input_columns", "engineered_features", "classifier",
        "imbalance_handling", "preprocessing", "threshold_selection",
        "evaluation", "limitation",
    }
    missing_keys = required_keys - config.keys()
    if missing_keys:
        raise ValueError(f"Inference config is missing keys: {sorted(missing_keys)}")

    if config["artifact"] != EXPECTED_ARTIFACT:
        raise ValueError("Config artifact must name the approved threshold-optimized pipeline.")
    if not isinstance(config["threshold"], (int, float)) or isinstance(config["threshold"], bool):
        raise ValueError("Inference config threshold must be numeric.")
    if float(config["threshold"]) != FROZEN_THRESHOLD:
        raise ValueError(f"Inference threshold is frozen at {FROZEN_THRESHOLD:.2f}.")
    if config["decision_rule"] != "predict_proba(X)[:, 1] >= threshold":
        raise ValueError("Unsupported inference decision rule in config.")
    if config["target"] != "is_fraud":
        raise ValueError("Inference config target must be 'is_fraud'.")
    if type(config["positive_class"]) is not int or config["positive_class"] != 1:
        raise ValueError("Inference config positive_class must be integer 1.")
    if config["input_columns"] != [
        "amount", "payment_terms_days", "line_item_count", "invoice_date", "due_date"
    ]:
        raise ValueError("Inference config input_columns do not match the supported schema.")
    if config["engineered_features"] != ENGINEERED_FEATURES:
        raise ValueError("Inference config engineered_features do not match the pipeline schema.")
    if not isinstance(config["preprocessing"], str) or not config["preprocessing"]:
        raise ValueError("Inference config preprocessing description must be a non-empty string.")
    if not isinstance(config["limitation"], str) or not config["limitation"]:
        raise ValueError("Inference config limitation must be a non-empty string.")

    classifier = config["classifier"]
    if not isinstance(classifier, dict) or classifier.get("class") != (
        "sklearn.ensemble.GradientBoostingClassifier"
    ):
        raise ValueError("Inference config classifier metadata is invalid.")
    expected_parameters = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "random_state": 42,
    }
    parameters = classifier.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Inference config classifier parameters must be an object.")
    for key in ("n_estimators", "max_depth", "min_samples_leaf", "random_state"):
        if type(parameters.get(key)) is not int:
            raise ValueError(f"Inference config classifier {key} must be an integer.")
    if isinstance(parameters.get("learning_rate"), bool) or not isinstance(
        parameters.get("learning_rate"), (int, float)
    ):
        raise ValueError("Inference config classifier learning_rate must be numeric.")
    if parameters != expected_parameters:
        raise ValueError("Inference config classifier parameters do not match the saved model.")

    imbalance = config["imbalance_handling"]
    if imbalance != {
        "method": "SMOTE", "sampling_strategy": 0.5,
        "k_neighbors": 5, "random_state": 42,
    }:
        raise ValueError("Inference config imbalance metadata is invalid.")

    selection = config["threshold_selection"]
    if not isinstance(selection, dict) or selection.get("selected_threshold") != FROZEN_THRESHOLD:
        raise ValueError("Inference config threshold-selection metadata is invalid.")
    if not isinstance(selection.get("method"), str) or not isinstance(selection.get("validation"), str):
        raise ValueError("Inference config threshold-selection metadata is incomplete.")

    evaluation = config["evaluation"]
    evaluation_keys = {
        "dataset", "random_seed", "metrics_report", "predictions", "records",
        "fraud_cases", "precision", "recall", "f1", "pr_auc", "roc_auc",
    }
    if not isinstance(evaluation, dict) or evaluation_keys - evaluation.keys():
        raise ValueError("Inference config evaluation metadata is incomplete.")
    if not all(isinstance(evaluation[key], str) and evaluation[key] for key in (
        "dataset", "metrics_report", "predictions"
    )):
        raise ValueError("Inference config evaluation paths must be non-empty strings.")
    for key in ("random_seed", "records", "fraud_cases"):
        if type(evaluation[key]) is not int:
            raise ValueError(f"Inference config evaluation {key} must be an integer.")
    for key in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
        if not isinstance(evaluation[key], (int, float)) or isinstance(evaluation[key], bool):
            raise ValueError(f"Inference config evaluation {key} must be numeric.")
        if not np.isfinite(float(evaluation[key])):
            raise ValueError(f"Inference config evaluation {key} must be finite.")
        if not 0.0 <= float(evaluation[key]) <= 1.0:
            raise ValueError(f"Inference config evaluation {key} must be between 0 and 1.")
    return config


def resolve_model_artifact(config: dict[str, Any]) -> Path:
    """Resolve only the approved, trusted model artifact inside models/."""
    artifact_value = config.get("artifact")
    if not isinstance(artifact_value, str):
        raise ValueError("Configured model artifact must be a relative path string.")
    artifact_relative = Path(artifact_value)
    if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
        raise ValueError("Configured model artifact cannot be absolute or traverse directories.")
    if artifact_value != EXPECTED_ARTIFACT:
        raise ValueError("Configured model artifact is not the approved pipeline artifact.")
    repository_root = ROOT.resolve(strict=True)
    models_directory = (ROOT / "models").resolve(strict=True)
    try:
        models_directory.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("Repository models directory resolves outside the repository.") from exc
    unresolved_candidate = ROOT / artifact_relative
    if unresolved_candidate.is_symlink():
        raise ValueError("Configured model artifact cannot be a symbolic link.")
    candidate = unresolved_candidate.resolve(strict=True)
    try:
        candidate.relative_to(models_directory)
    except ValueError as exc:
        raise ValueError("Configured model artifact must resolve inside the approved models directory.") from exc
    if candidate.suffix.lower() != ".joblib" or not candidate.is_file():
        raise FileNotFoundError("Approved model artifact is missing or is not a .joblib file.")
    return candidate


def prepare_inference_features(records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate raw invoice records and derive the model's expected features."""
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")
    if records.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names.")
    unexpected_columns = set(records.columns) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS
    if unexpected_columns:
        raise ValueError(f"Unexpected input columns: {sorted(unexpected_columns)}")
    if records.isna().any().any():
        raise ValueError("Input contains missing values.")

    missing_columns = REQUIRED_COLUMNS - set(records.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required input columns: {sorted(missing_columns)}"
        )

    frame = records.copy()
    if "invoice_id" in frame.columns:
        if frame["invoice_id"].isna().any():
            raise ValueError("invoice_id contains missing values.")
        if frame["invoice_id"].duplicated().any():
            raise ValueError("invoice_id values must be unique within the input.")
        identifiers = frame["invoice_id"].astype(str).tolist()
    else:
        identifiers = [f"row-{index + 1:06d}" for index in range(len(frame))]

    for column in ("amount", "payment_terms_days", "line_item_count"):
        if frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{column} must contain numeric values, not booleans.")
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} must contain numeric, non-missing values.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values.")
        frame[column] = values

    if not frame["amount"].between(50, 250_000, inclusive="both").all():
        raise ValueError("amount must be between 50 and 250000.")
    if not frame["payment_terms_days"].isin([15, 30, 45, 60]).all():
        raise ValueError("payment_terms_days must be one of 15, 30, 45, or 60.")
    if not frame["line_item_count"].between(1, 15).all():
        raise ValueError("line_item_count must be an integer from 1 through 15.")
    if not np.equal(
        frame["line_item_count"].to_numpy(dtype=float),
        np.floor(frame["line_item_count"].to_numpy(dtype=float)),
    ).all():
        raise ValueError("line_item_count values must be whole numbers.")

    for column in ("invoice_date", "due_date"):
        try:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            timezone = parsed.dt.tz
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"{column} must contain valid timezone-naive dates.") from exc
        if parsed.isna().any():
            raise ValueError(f"{column} must contain valid, non-missing dates.")
        if timezone is not None:
            raise ValueError(f"{column} must contain timezone-naive dates.")
        frame[column] = parsed

    invoice_lower = pd.Timestamp("2025-01-01")
    invoice_upper = pd.Timestamp("2025-12-31")
    due_upper = pd.Timestamp("2026-03-01")
    if not frame["invoice_date"].between(invoice_lower, invoice_upper).all():
        raise ValueError("invoice_date must fall within the generator's 2025 date range.")
    if (frame["due_date"] > due_upper).any():
        raise ValueError("due_date exceeds the generator's maximum date of 2026-03-01.")

    if (frame["due_date"] < frame["invoice_date"]).any():
        raise ValueError("due_date cannot be earlier than invoice_date.")
    actual_terms = (frame["due_date"] - frame["invoice_date"]).dt.days
    if not actual_terms.eq(frame["payment_terms_days"]).all():
        raise ValueError(
            "payment_terms_days must match the difference between due_date and invoice_date."
        )

    features = pd.DataFrame(index=frame.index)
    features["amount"] = frame["amount"].astype(float)
    features["payment_terms_days"] = frame["payment_terms_days"].astype(int)
    features["line_item_count"] = frame["line_item_count"].astype(int)
    features["invoice_year"] = frame["invoice_date"].dt.year
    features["invoice_month"] = frame["invoice_date"].dt.month
    features["invoice_day_of_week"] = frame["invoice_date"].dt.dayofweek
    features["due_days_after_invoice"] = actual_terms
    features = features[ENGINEERED_FEATURES]
    if features.isna().any().any():
        raise ValueError("Derived model features contain missing values.")
    return features, identifiers


def predict_with_pipeline(
    records: pd.DataFrame,
    pipeline: Any,
    threshold: float,
) -> pd.DataFrame:
    """Predict via the saved pipeline's predict_proba; never fit on input."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.number))
        or float(threshold) != FROZEN_THRESHOLD
    ):
        raise ValueError(f"Inference threshold is frozen at {FROZEN_THRESHOLD:.2f}.")
    features, identifiers = prepare_inference_features(records)
    classes = getattr(pipeline, "classes_", None)
    class_values = None if classes is None else np.asarray(classes).tolist()
    if (
        not isinstance(class_values, list)
        or len(class_values) != 2
        or any(type(value) is not int for value in class_values)
        or class_values != [0, 1]
    ):
        raise ValueError("Pipeline classes_ must be exactly [0, 1] in that order.")
    probabilities = np.asarray(pipeline.predict_proba(features))
    if probabilities.shape != (len(features), 2):
        raise ValueError("Pipeline predict_proba returned an unexpected shape.")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError("Pipeline returned invalid class probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("Pipeline class probabilities do not sum to 1.")
    fraud_probabilities = probabilities[:, 1]
    predictions = (fraud_probabilities >= float(threshold)).astype(int)
    return pd.DataFrame({
        "invoice_id": identifiers,
        "fraud_probability": fraud_probabilities,
        "threshold": float(threshold),
        "is_fraud_prediction": predictions,
    })


def predict_records(
    records: pd.DataFrame,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    config = load_config(config_path)
    model_path = resolve_model_artifact(config)
    # Load only this repository's trusted model artifact. Inference calls
    # predict_proba only; the fitted preprocessing inside the pipeline runs
    # its transform path and is never fit on these records.
    # The configured artifact is fixed to this project-owned model. Never
    # deserialize model paths supplied by inference callers.
    pipeline = joblib.load(model_path)
    return predict_with_pipeline(records, pipeline, float(config["threshold"]))


def validate_output_path(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Reject input collisions and require explicit opt-in for CSV replacement."""
    resolved_input = Path(input_path).resolve(strict=True)
    if Path(output_path).is_symlink():
        raise ValueError("Output path cannot be a symbolic link.")
    resolved_output = Path(output_path).resolve()
    same_file = resolved_input == resolved_output
    if not same_file and resolved_output.exists():
        try:
            same_file = resolved_input.samefile(resolved_output)
        except OSError:
            same_file = False
    if same_file:
        raise ValueError("Output path cannot overwrite the input CSV.")
    if resolved_output.suffix.lower() != ".csv":
        raise ValueError("Inference output path must end in .csv.")
    if resolved_output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists; pass --overwrite to replace it: {resolved_output}"
            )
        if resolved_output.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {resolved_output}")
    return resolved_output


def write_predictions_atomically(
    predictions: pd.DataFrame,
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Flush a same-directory temporary CSV, then publish it atomically."""
    destination = validate_output_path(input_path, output_path, overwrite)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Could not create inference output directory: {destination.parent}"
        ) from exc

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            predictions.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # The closed temporary file is in the destination directory, hence
        # both publishing operations stay on the same filesystem. os.link
        # atomically refuses an existing destination when overwrite is off;
        # os.replace atomically replaces it only after explicit opt-in.
        if overwrite:
            os.replace(temporary_path, destination)
            temporary_path = None
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()
            temporary_path = None
        return destination
    except FileExistsError as exc:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {destination}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Could not safely publish inference output: {destination}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                # Preserve the original write/publish error. Cleanup was
                # attempted; the filesystem may itself be unavailable.
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict invoice fraud using the saved threshold-optimized pipeline."
    )
    parser.add_argument("input_csv", type=Path, help="CSV containing raw invoice records")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV (default: <input stem>_predictions.csv)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Inference configuration JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing output CSV (never the input CSV).",
    )
    args = parser.parse_args()

    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    requested_output = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}_predictions.csv"
    )
    output_path = validate_output_path(args.input_csv, requested_output, args.overwrite)
    records = pd.read_csv(args.input_csv)
    results = predict_records(records, args.config)
    output_path = write_predictions_atomically(
        results, args.input_csv, output_path, overwrite=args.overwrite
    )
    print(f"Predicted {len(results)} invoice records.")
    print(f"Frozen threshold: {FROZEN_THRESHOLD:.2f}")
    print(f"Saved predictions: {output_path.resolve()}")


if __name__ == "__main__":
    main()
