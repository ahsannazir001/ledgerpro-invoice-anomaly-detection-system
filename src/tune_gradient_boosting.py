
from pathlib import Path
from itertools import product

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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "invoices_clean.csv"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_SPLITS = 5
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

# 16 controlled configurations
PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
    "min_samples_leaf": [2, 5],
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

    X = df[FEATURES].copy()
    y = df["is_fraud"].astype(int)

    if y.isna().any() or X.isna().all().any():
        raise ValueError("Invalid missing values detected.")

    if y.nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")

    return X, y


def build_pipeline(params):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_STATE,
    )

    return ImbPipeline([
        ("preprocessing", preprocessing),
        (
            "smote",
            SMOTE(
                sampling_strategy=0.5,
                k_neighbors=5,
                random_state=RANDOM_STATE,
            ),
        ),
        ("classifier", classifier),
    ])


def evaluate_configuration(X, y, params, cv):
    fold_results = []

    all_actual = []
    all_predictions = []
    all_probabilities = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        pipeline = build_pipeline(params)
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        fold_results.append({
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

        all_actual.extend(y_valid.tolist())
        all_predictions.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())

    fold_df = pd.DataFrame(fold_results)

    return {
        **params,
        "mean_fold_f1": fold_df["f1"].mean(),
        "std_fold_f1": fold_df["f1"].std(),
        "mean_fold_precision": fold_df["precision"].mean(),
        "mean_fold_recall": fold_df["recall"].mean(),
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


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    if (y.value_counts() < N_SPLITS).any():
        raise ValueError("Not enough samples for stratified CV.")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    combinations = list(product(
        PARAM_GRID["n_estimators"],
        PARAM_GRID["learning_rate"],
        PARAM_GRID["max_depth"],
        PARAM_GRID["min_samples_leaf"],
    ))

    results = []

    print(f"Dataset shape: {X.shape}")
    print(f"Fraud distribution:\n{y.value_counts()}")
    print(f"\nConfigurations: {len(combinations)}")
    print(f"Total model fits: {len(combinations) * N_SPLITS}")
    print(f"Threshold: {THRESHOLD}")
    print("\nStarting hyperparameter comparison...\n")

    for index, values in enumerate(combinations, start=1):
        params = {
            "n_estimators": values[0],
            "learning_rate": values[1],
            "max_depth": values[2],
            "min_samples_leaf": values[3],
        }

        print(
            f"\n[{index}/{len(combinations)}] "
            f"Testing: {params}"
        )

        result = evaluate_configuration(
            X, y, params, cv
        )
        results.append(result)

        print(
            f"Mean F1: {result['mean_fold_f1']:.4f} | "
            f"Pooled F1: {result['pooled_f1']:.4f} | "
            f"Precision: {result['pooled_precision']:.4f} | "
            f"Recall: {result['pooled_recall']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["mean_fold_f1", "pooled_f1"],
        ascending=False,
    ).reset_index(drop=True)

    results_path = REPORT_DIR / "gb_hyperparameter_comparison.csv"
    text_path = REPORT_DIR / "gb_hyperparameter_comparison.txt"

    results_df.to_csv(results_path, index=False)

    with text_path.open("w", encoding="utf-8") as file:
        file.write("GRADIENT BOOSTING HYPERPARAMETER COMPARISON\n")
        file.write("=" * 65 + "\n\n")
        file.write(f"CV folds: {N_SPLITS}\n")
        file.write(f"Threshold: {THRESHOLD}\n")
        file.write("Sampling: SMOTE inside training folds only\n")
        file.write(f"Configurations: {len(combinations)}\n\n")
        file.write(results_df.to_string(index=False))
        file.write("\n")

    print("\n\nFINAL RANKING")
    print(results_df.to_string(index=False))

    print("\nReports saved:")
    print(results_path)
    print(text_path)

    print("\nBest configuration based on mean CV F1:")
    print(results_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
