
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
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.50

CATEGORICAL_FEATURES = [
    "vendor_id",
    "category",
    "payment_method",
    "country",
]

NUMERIC_FEATURES = [
    "amount",
    "payment_terms_days",
    "line_item_count",
    "invoice_year",
    "invoice_month",
    "invoice_day_of_week",
    "due_days_after_invoice",
]

FEATURE_GROUPS = {
    "amount_only": ["amount"],
    "amount_plus_categorical": [
        "amount",
        *CATEGORICAL_FEATURES,
    ],
    "amount_plus_other_numeric": NUMERIC_FEATURES,
    "all_features": [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ],
}


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {
        "invoice_id",
        "invoice_date",
        "due_date",
        "is_fraud",
        *CATEGORICAL_FEATURES,
        "amount",
        "payment_terms_days",
        "line_item_count",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], errors="raise"
    )
    df["due_date"] = pd.to_datetime(
        df["due_date"], errors="raise"
    )

    df["invoice_year"] = df["invoice_date"].dt.year
    df["invoice_month"] = df["invoice_date"].dt.month
    df["invoice_day_of_week"] = (
        df["invoice_date"].dt.dayofweek
    )
    df["due_days_after_invoice"] = (
        df["due_date"] - df["invoice_date"]
    ).dt.days

    return df


def build_pipeline(feature_columns):
    categorical = [
        col for col in CATEGORICAL_FEATURES
        if col in feature_columns
    ]
    numeric = [
        col for col in feature_columns
        if col not in categorical
    ]

    if categorical:
        sampler = SMOTENC(
            categorical_features=categorical,
            sampling_strategy=0.5,
            k_neighbors=5,
            random_state=RANDOM_STATE,
        )
    else:
        sampler = SMOTE(
            sampling_strategy=0.5,
            k_neighbors=5,
            random_state=RANDOM_STATE,
        )

    numeric_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    imputer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    categorical_encoder = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ],
        remainder="drop",
    )

    return ImbPipeline([
        ("imputer", imputer),
        ("sampler", sampler),
        ("encoder", categorical_encoder),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def evaluate_group(df, feature_columns, group_name):
    X = df[feature_columns].copy()
    y = df["is_fraud"].astype(int)

    if X.isna().any().any():
        print(
            f"Note: {group_name} contains missing values; "
            "they will be imputed inside each training fold."
        )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = pd.Series(index=df.index, dtype=float)
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        pipeline = build_pipeline(feature_columns)

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        oof_probabilities.iloc[valid_idx] = probabilities

        fold_results.append({
            "feature_group": group_name,
            "fold": fold,
            "precision": precision_score(
                y_valid, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_valid, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_valid, predictions, zero_division=0
            ),
            "pr_auc": average_precision_score(
                y_valid, probabilities
            ),
            "roc_auc": roc_auc_score(
                y_valid, probabilities
            ),
        })

        print(
            f"{group_name} | Fold {fold} | "
            f"F1: {fold_results[-1]['f1']:.4f}"
        )

    valid_mask = oof_probabilities.notna()
    y_true = y.loc[valid_mask]
    oof_probabilities = oof_probabilities.loc[valid_mask]

    oof_predictions = (
        oof_probabilities >= THRESHOLD
    ).astype(int)

    summary = {
        "feature_group": group_name,
        "feature_count": len(feature_columns),
        "precision": precision_score(
            y_true, oof_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, oof_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, oof_predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, oof_probabilities
        ),
        "roc_auc": roc_auc_score(
            y_true, oof_probabilities
        ),
    }

    return summary, fold_results


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    if df["is_fraud"].nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")

    if df["is_fraud"].isna().any():
        raise ValueError("Target contains missing values.")

    if (df["is_fraud"].value_counts() < N_SPLITS).any():
        raise ValueError(
            "Not enough samples in a class for the selected CV folds."
        )

    summaries = []
    all_fold_results = []

    print(f"Dataset shape: {df.shape}")
    print("Fraud distribution:")
    print(df["is_fraud"].value_counts().to_string())
    print("\nStarting feature subset comparison...\n")

    for group_name, features in FEATURE_GROUPS.items():
        print(f"\nEvaluating: {group_name}")

        summary, fold_results = evaluate_group(
            df,
            features,
            group_name,
        )

        summaries.append(summary)
        all_fold_results.extend(fold_results)

    summary_df = pd.DataFrame(summaries)
    fold_df = pd.DataFrame(all_fold_results)

    summary_path = REPORT_DIR / "feature_subset_comparison.csv"
    fold_path = REPORT_DIR / "feature_subset_fold_metrics.csv"
    text_path = REPORT_DIR / "feature_subset_comparison.txt"

    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    with text_path.open("w", encoding="utf-8") as file:
        file.write("FEATURE SUBSET COMPARISON\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Threshold: {THRESHOLD}\n")
        file.write(f"CV folds: {N_SPLITS}\n")
        file.write("Sampling: SMOTE / SMOTENC, training folds only\n")
        file.write("Classifier: GradientBoostingClassifier\n\n")
        file.write(summary_df.to_string(index=False))
        file.write("\n\nFOLD-LEVEL METRICS\n")
        file.write("=" * 60 + "\n")
        file.write(fold_df.to_string(index=False))
        file.write("\n")

    print("\nFINAL COMPARISON")
    print(summary_df.to_string(index=False))

    print("\nReports saved:")
    print(summary_path)
    print(fold_path)
    print(text_path)


if __name__ == "__main__":
    main()
