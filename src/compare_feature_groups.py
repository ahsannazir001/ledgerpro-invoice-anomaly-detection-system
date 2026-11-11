
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5

TARGET = "is_fraud"
ID_COLUMN = "invoice_id"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        ID_COLUMN,
        TARGET,
        "invoice_date",
        "due_date",
        "amount",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[ID_COLUMN].isna().any():
        raise ValueError("Invoice IDs contain missing values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate invoice IDs found.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    y = df[TARGET].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes.")

    if y.value_counts().min() < N_SPLITS:
        raise ValueError("Not enough minority samples for CV.")

    X = df.drop(columns=[TARGET, ID_COLUMN]).copy()

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

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in categorical_columns:
        X[column] = X[column].astype("string")

    if "amount" not in numeric_columns:
        raise ValueError("Amount must be a numeric feature.")

    return X, y, numeric_columns, categorical_columns


def build_model(numeric_columns, categorical_columns):
    transformers = []

    if numeric_columns:
        numeric_pipeline = SklearnPipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = SklearnPipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return ImbPipeline(
        steps=[
            ("preprocessing", preprocessing),
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
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_feature_group(name, X, y, cv):
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    if X.shape[1] == 0:
        raise ValueError(f"{name} has no features.")

    oof_probabilities = [None] * len(X)
    fold_metrics = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = build_model(
            numeric_columns,
            categorical_columns,
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_valid)[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        for index, probability in zip(
            valid_idx, probabilities
        ):
            if oof_probabilities[index] is not None:
                raise RuntimeError(
                    f"Duplicate OOF prediction for row {index}"
                )

            oof_probabilities[index] = float(probability)

        fold_metrics.append(
            {
                "feature_group": name,
                "fold": fold,
                "precision": precision_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),
            }
        )

        print(
            f"{name} | Fold {fold}: "
            f"Precision={fold_metrics[-1]['precision']:.4f}, "
            f"Recall={fold_metrics[-1]['recall']:.4f}, "
            f"F1={fold_metrics[-1]['f1_score']:.4f}"
        )

    if any(value is None for value in oof_probabilities):
        raise RuntimeError("Missing out-of-fold predictions.")

    probabilities = pd.Series(
        oof_probabilities,
        dtype="float64",
    ).to_numpy()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    pooled_metrics = {
        "feature_group": name,
        "feature_count": X.shape[1],
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }

    return pooled_metrics, fold_metrics


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, numeric_columns, categorical_columns = load_data()

    feature_groups = {
        "amount_only": ["amount"],
        "without_amount": [
            column for column in X.columns
            if column != "amount"
        ],
        "all_features": X.columns.tolist(),
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("=" * 65)
    print("LEDGERPRO - FEATURE GROUP COMPARISON")
    print("=" * 65)
    print(f"Dataset rows: {len(X)}")
    print(f"Fraud invoices: {int(y.sum())}")
    print(f"Legitimate invoices: {int((y == 0).sum())}")
    print(f"CV folds: {N_SPLITS}")
    print(f"Threshold: {THRESHOLD}")
    print()

    results = []
    all_fold_results = []

    for group_name, columns in feature_groups.items():
        print(f"\nEvaluating: {group_name}")
        print(f"Features: {columns}")

        X_group = X[columns].copy()

        result, fold_results = evaluate_feature_group(
            group_name,
            X_group,
            y,
            cv,
        )

        results.append(result)
        all_fold_results.extend(fold_results)

    results_df = pd.DataFrame(results)

    fold_df = pd.DataFrame(all_fold_results)

    results_path = REPORTS_DIR / "feature_group_comparison.csv"
    folds_path = REPORTS_DIR / "feature_group_fold_metrics.csv"
    report_path = REPORTS_DIR / "feature_group_comparison.txt"

    results_df.to_csv(results_path, index=False)
    fold_df.to_csv(folds_path, index=False)

    report = [
        "LEDGERPRO - FEATURE GROUP COMPARISON",
        "=" * 55,
        f"Rows: {len(X)}",
        f"Folds: {N_SPLITS}",
        f"Threshold: {THRESHOLD}",
        "Model: GradientBoostingClassifier",
        "Imbalance handling: SMOTE (sampling_strategy=0.5)",
        "",
        "POOLED OUT-OF-FOLD RESULTS",
        "-" * 55,
        results_df.to_string(index=False),
        "",
        "FOLD-LEVEL RESULTS",
        "-" * 55,
        fold_df.to_string(index=False),
        "",
        "NOTE: All preprocessing and SMOTE fitting occur inside each",
        "training fold. Validation folds are not resampled.",
        "This experiment does not tune the classification threshold.",
    ]

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("POOLED FEATURE GROUP RESULTS")
    print("=" * 65)
    print(results_df.round(4).to_string(index=False))

    print("\nSaved:")
    print(results_path)
    print(folds_path)
    print(report_path)


if __name__ == "__main__":
    main()
