
"""Analyze feature importance for the Gradient Boosting + SMOTE model."""

from pathlib import Path

import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "invoices_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"

TARGET = "is_fraud"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Permutation importance will use only the training partition.
PERMUTATION_REPEATS = 5


# ---------------------------------------------------------
# 2. Load and prepare data
# ---------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty or TARGET not in df.columns:
        raise ValueError("Dataset is empty or target column is missing.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both classes: 0 and 1.")

    X = df.drop(columns=[TARGET, "invoice_id"]).copy()

    invoice_dates = pd.to_datetime(
        X.pop("invoice_date"),
        errors="coerce",
    )

    due_dates = pd.to_datetime(
        X.pop("due_date"),
        errors="coerce",
    )

    X["invoice_year"] = invoice_dates.dt.year
    X["invoice_month"] = invoice_dates.dt.month
    X["invoice_day_of_week"] = invoice_dates.dt.dayofweek

    X["due_days_after_invoice"] = (
        due_dates - invoice_dates
    ).dt.days

    categorical_columns = [
        "vendor_id",
        "category",
        "payment_method",
        "country",
    ]

    for column in categorical_columns:
        X[column] = X[column].astype("string")

    y = df[TARGET].astype(int)

    return X, y


# ---------------------------------------------------------
# 3. Build model pipeline
# ---------------------------------------------------------

def build_model(X):

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

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

    model = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(
                    sampling_strategy=0.5,
                    random_state=RANDOM_STATE,
                    k_neighbors=5,
                ),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


# ---------------------------------------------------------
# 4. Analyze feature importance
# ---------------------------------------------------------

def analyze_features():

    X, y = load_data()

    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())

    # Split before fitting any preprocessing or resampling.
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model(X)

    print("\nTraining Gradient Boosting + SMOTE...")

    model.fit(X_train, y_train)

    print("\nCalculating permutation importance on training data...")

    # Use original, non-resampled training data for importance.
    # The scoring metric is average precision.
    importance = permutation_importance(
        model,
        X_train,
        y_train,
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    print("\n========== FEATURE IMPORTANCE ==========")
    print(importance_df.to_string(index=False))

    print("\n========== TOP 15 FEATURES ==========")
    print(importance_df.head(15).to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature importance saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    analyze_features()
