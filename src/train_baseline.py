
"""Train and evaluate the LedgerPro baseline fraud detection model."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. Project paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "invoices_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "is_fraud"
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. Load and validate dataset
# ---------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "invoice_id",
        "vendor_id",
        "invoice_date",
        "due_date",
        "amount",
        "category",
        "payment_method",
        "country",
        "payment_terms_days",
        "line_item_count",
        TARGET,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes: 0 and 1.")

    if df[TARGET].value_counts().min() < 2:
        raise ValueError("Not enough samples in the minority class.")

    print(f"Dataset loaded: {df.shape}")
    print("\nClass distribution:")
    print(df[TARGET].value_counts())

    return df


# ---------------------------------------------------------
# 3. Prepare model features
# ---------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[TARGET, "invoice_id"]).copy()

    # Parse dates safely.
    invoice_dates = pd.to_datetime(
        features.pop("invoice_date"),
        errors="coerce",
    )

    due_dates = pd.to_datetime(
        features.pop("due_date"),
        errors="coerce",
    )

    # Extract useful numerical date features.
    features["invoice_year"] = invoice_dates.dt.year
    features["invoice_month"] = invoice_dates.dt.month
    features["invoice_day_of_week"] = invoice_dates.dt.dayofweek

    features["due_days_after_invoice"] = (
        due_dates - invoice_dates
    ).dt.days

    # Treat categorical identifiers as categorical values.
    categorical_columns = [
        "vendor_id",
        "category",
        "payment_method",
        "country",
    ]

    for column in categorical_columns:
        features[column] = features[column].astype("string")

    return features


# ---------------------------------------------------------
# 4. Build preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------
# 5. Train and evaluate baseline model
# ---------------------------------------------------------

def train_baseline() -> None:

    df = load_dataset()

    X = prepare_features(df)
    y = df[TARGET].astype("int64")

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    if not numeric_columns or not categorical_columns:
        raise ValueError(
            "Expected both numerical and categorical features."
        )

    print("\nNumerical features:")
    print(numeric_columns)

    print("\nCategorical features:")
    print(categorical_columns)

    # Stratified split preserves the original class ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Fit preprocessing only on training data.
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("\nTraining Random Forest baseline model...")

    model_pipeline.fit(X_train, y_train)

    # Evaluate on untouched test data.
    y_pred = model_pipeline.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )

    print("\n========== BASELINE MODEL RESULTS ==========")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # -----------------------------------------------------
    # 6. Save model and preprocessing artifacts
    # -----------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model_pipeline,
        MODEL_DIR / "baseline_pipeline.joblib",
    )

    joblib.dump(
        model_pipeline.named_steps["classifier"],
        MODEL_DIR / "baseline_model.joblib",
    )

    joblib.dump(
        model_pipeline.named_steps["preprocessor"],
        MODEL_DIR / "preprocessing_pipeline.joblib",
    )

    # Save evaluation results.
    report_path = REPORT_DIR / "baseline_evaluation.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("LedgerPro Baseline Model Evaluation\n")
        file.write("=" * 45 + "\n\n")
        file.write(f"Model: RandomForestClassifier\n")
        file.write(f"Class imbalance: class_weight='balanced'\n")
        file.write(f"Test size: {TEST_SIZE}\n")
        file.write(f"Random state: {RANDOM_STATE}\n\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n\n")
        file.write(f"Confusion Matrix:\n{cm}\n\n")
        file.write(f"Classification Report:\n{report}")

    print("\nSaved artifacts:")
    print(MODEL_DIR / "baseline_pipeline.joblib")
    print(MODEL_DIR / "baseline_model.joblib")
    print(MODEL_DIR / "preprocessing_pipeline.joblib")
    print(report_path)


if __name__ == "__main__":
    train_baseline()
