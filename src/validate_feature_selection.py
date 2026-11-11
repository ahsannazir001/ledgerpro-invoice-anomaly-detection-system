
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
THRESHOLD = 0.50

FEATURE_GROUPS = {
    "amount_only": [
        "amount",
    ],
    "amount_plus_other_numeric": [
        "amount",
        "payment_terms_days",
        "line_item_count",
        "invoice_year",
        "invoice_month",
        "invoice_day_of_week",
        "due_days_after_invoice",
    ],
}


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {
        "amount",
        "payment_terms_days",
        "line_item_count",
        "invoice_date",
        "due_date",
        "is_fraud",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

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
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                feature_columns,
            ),
        ],
        remainder="drop",
    )

    return ImbPipeline([
        ("preprocessing", preprocessing),
        (
            "sampler",
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
    ])


def evaluate_group(X, y, group_name, cv):
    fold_results = []
    all_actual = []
    all_predictions = []
    all_probabilities = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        pipeline = build_pipeline(list(X.columns))
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        precision = precision_score(
            y_valid, predictions, zero_division=0
        )
        recall = recall_score(
            y_valid, predictions, zero_division=0
        )
        f1 = f1_score(
            y_valid, predictions, zero_division=0
        )

        fold_results.append({
            "feature_group": group_name,
            "fold": fold_number,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": average_precision_score(
                y_valid, probabilities
            ),
            "roc_auc": roc_auc_score(
                y_valid, probabilities
            ),
        })

        all_actual.extend(y_valid.tolist())
        all_predictions.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())

        print(
            f"{group_name} | Fold {fold_number:02d} "
            f"| F1: {f1:.4f} "
            f"| Precision: {precision:.4f} "
            f"| Recall: {recall:.4f}"
        )

    # Pooled metrics across all validation predictions.
    # Each sample is evaluated once per repeat.
    summary = {
        "feature_group": group_name,
        "feature_count": X.shape[1],
        "total_folds": len(fold_results),
        "mean_fold_f1": pd.DataFrame(fold_results)["f1"].mean(),
        "std_fold_f1": pd.DataFrame(fold_results)["f1"].std(),
        "mean_fold_precision": pd.DataFrame(
            fold_results
        )["precision"].mean(),
        "mean_fold_recall": pd.DataFrame(
            fold_results
        )["recall"].mean(),
        "pooled_precision": precision_score(
            all_actual, all_predictions, zero_division=0
        ),
        "pooled_recall": recall_score(
            all_actual, all_predictions, zero_division=0
        ),
        "pooled_f1": f1_score(
            all_actual, all_predictions, zero_division=0
        ),
        "pooled_pr_auc": average_precision_score(
            all_actual, all_probabilities
        ),
        "pooled_roc_auc": roc_auc_score(
            all_actual, all_probabilities
        ),
    }

    return summary, fold_results


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    if df["is_fraud"].isna().any():
        raise ValueError("Target contains missing values.")

    y = df["is_fraud"].astype(int)

    if y.nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")

    if (y.value_counts() < N_SPLITS).any():
        raise ValueError("Insufficient samples for stratified CV.")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    summaries = []
    all_fold_results = []

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution:\n{y.value_counts()}")
    print(f"\nCV: {N_SPLITS} folds x {N_REPEATS} repeats")
    print(f"Threshold: {THRESHOLD}")
    print("Starting repeated cross-validation...\n")

    for group_name, features in FEATURE_GROUPS.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {group_name}")
        print("=" * 60)

        X = df[features].copy()

        summary, fold_results = evaluate_group(
            X,
            y,
            group_name,
            cv,
        )

        summaries.append(summary)
        all_fold_results.extend(fold_results)

    summary_df = pd.DataFrame(summaries)
    fold_df = pd.DataFrame(all_fold_results)

    summary_path = REPORT_DIR / "repeated_cv_comparison.csv"
    fold_path = REPORT_DIR / "repeated_cv_fold_metrics.csv"
    text_path = REPORT_DIR / "repeated_cv_comparison.txt"

    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    with text_path.open("w", encoding="utf-8") as file:
        file.write("REPEATED STRATIFIED CROSS-VALIDATION\n")
        file.write("=" * 65 + "\n\n")
        file.write(f"Folds: {N_SPLITS}\n")
        file.write(f"Repeats: {N_REPEATS}\n")
        file.write(f"Threshold: {THRESHOLD}\n")
        file.write("Sampling: SMOTE inside training folds only\n")
        file.write(
            "Classifier: GradientBoostingClassifier\n\n"
        )
        file.write("SUMMARY\n")
        file.write(summary_df.to_string(index=False))
        file.write("\n\nFOLD METRICS\n")
        file.write(fold_df.to_string(index=False))
        file.write("\n")

    print("\n\nFINAL REPEATED CV COMPARISON")
    print(summary_df.to_string(index=False))

    print("\nReports saved:")
    print(summary_path)
    print(fold_path)
    print(text_path)


if __name__ == "__main__":
    main()
