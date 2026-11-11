
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50

FEATURES = [
    "amount",
    "payment_terms_days",
    "line_item_count",
    "invoice_year",
    "invoice_month",
    "invoice_day_of_week",
    "due_days_after_invoice",
]

TARGET = "is_fraud"

BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.10,
    "max_depth": 3,
    "min_samples_leaf": 5,
}


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        TARGET,
        "invoice_date",
        "due_date",
        *FEATURES[:3],
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        errors="raise",
    )

    df["due_date"] = pd.to_datetime(
        df["due_date"],
        errors="raise",
    )

    # Date feature engineering
    df["invoice_year"] = df["invoice_date"].dt.year
    df["invoice_month"] = df["invoice_date"].dt.month

    df["invoice_day_of_week"] = (
        df["invoice_date"].dt.dayofweek
    )

    df["due_days_after_invoice"] = (
        df["due_date"] - df["invoice_date"]
    ).dt.days

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if X.isna().all().any():
        raise ValueError(
            "At least one feature contains only missing values."
        )

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "Target must contain exactly 0 and 1 classes."
        )

    y = y.astype(int)

    return df, X, y


# --------------------------------------------------
# 3. Build preprocessing pipeline
# --------------------------------------------------

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                FEATURES,
            ),
        ],
        remainder="drop",
    )


# --------------------------------------------------
# 4. Build complete training pipeline
# --------------------------------------------------

def build_model_pipeline():
    return ImbPipeline([
        (
            "preprocessing",
            build_preprocessor(),
        ),
        (
            "smote",
            SMOTE(
                sampling_strategy=0.5,
                k_neighbors=5,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=BEST_PARAMS["n_estimators"],
                learning_rate=BEST_PARAMS["learning_rate"],
                max_depth=BEST_PARAMS["max_depth"],
                min_samples_leaf=BEST_PARAMS["min_samples_leaf"],
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# --------------------------------------------------
# 5. Save evaluation report
# --------------------------------------------------

def save_report(
    metrics,
    report_text,
    matrix,
    train_distribution,
    test_distribution,
):
    report_path = REPORT_DIR / "final_model_evaluation.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("LEDGERPRO FINAL MODEL EVALUATION\n")
        file.write("=" * 65 + "\n\n")

        file.write("MODEL CONFIGURATION\n")
        file.write(json.dumps(BEST_PARAMS, indent=4))
        file.write("\n\n")

        file.write(f"Test threshold: {THRESHOLD}\n")
        file.write(f"Test size: {TEST_SIZE}\n")
        file.write("Sampling: SMOTE on training data only\n\n")

        file.write("TRAIN CLASS DISTRIBUTION\n")
        file.write(train_distribution.to_string())
        file.write("\n\n")

        file.write("TEST CLASS DISTRIBUTION\n")
        file.write(test_distribution.to_string())
        file.write("\n\n")

        file.write("TEST METRICS\n")
        for name, value in metrics.items():
            file.write(f"{name}: {value:.6f}\n")

        file.write("\nCONFUSION MATRIX\n")
        file.write(
            "Rows = actual classes; columns = predicted classes\n"
        )
        file.write(str(matrix))
        file.write("\n\nCLASSIFICATION REPORT\n")
        file.write(report_text)
        file.write("\n")

    return report_path


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    df, X, y = load_dataset()

    print(f"Dataset shape: {df.shape}")
    print("\nFull class distribution:")
    print(y.value_counts().sort_index())

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting distribution:")
    print(y_test.value_counts().sort_index())

    # Build and train model
    print("\nBuilding final model...")

    pipeline = build_model_pipeline()

    print("Training final Gradient Boosting model...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Predict on untouched test data
    print("\nEvaluating on held-out test set...")

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )

    # Save predictions
    predictions_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "fraud_probability": probabilities,
    }, index=y_test.index)

    predictions_path = (
        REPORT_DIR / "final_test_predictions.csv"
    )

    predictions_df.to_csv(predictions_path, index=True)

    # Save artifacts
    print("\nSaving model artifacts...")

    model_path = MODEL_DIR / "final_model.joblib"
    pipeline_path = MODEL_DIR / "final_pipeline.joblib"
    preprocessing_path = (
        MODEL_DIR / "final_preprocessing_pipeline.joblib"
    )

    # Full pipeline is the recommended inference artifact.
    joblib.dump(pipeline, pipeline_path)

    # Separate classifier and fitted preprocessing artifact.
    joblib.dump(
        pipeline.named_steps["classifier"],
        model_path,
    )

    joblib.dump(
        pipeline.named_steps["preprocessing"],
        preprocessing_path,
    )

    # Save metrics as JSON
    metrics_path = REPORT_DIR / "final_model_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    report_path = save_report(
        metrics=metrics,
        report_text=report_text,
        matrix=matrix,
        train_distribution=y_train.value_counts().sort_index(),
        test_distribution=y_test.value_counts().sort_index(),
    )

    # Final output
    print("\n" + "=" * 65)
    print("FINAL MODEL TEST RESULTS")
    print("=" * 65)

    for name, value in metrics.items():
        print(f"{name.upper():<15}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report_text)

    print("\nSaved artifacts:")
    print(pipeline_path)
    print(model_path)
    print(preprocessing_path)

    print("\nSaved reports:")
    print(report_path)
    print(metrics_path)
    print(predictions_path)

    if metrics["f1"] > 0.70:
        print(
            "\nF1 TARGET ACHIEVED on this held-out test split."
        )
    else:
        print(
            "\nF1 TARGET NOT ACHIEVED on this held-out test split."
        )

    print("\nFinal model training and evaluation completed.")


if __name__ == "__main__":
    main()
