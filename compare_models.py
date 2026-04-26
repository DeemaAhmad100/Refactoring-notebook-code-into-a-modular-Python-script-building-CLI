import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

EXPECTED_COLUMNS = NUMERIC_FEATURES + ["churned"]


def load_and_preprocess(filepath, random_state=42):
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    X = df[NUMERIC_FEATURES].copy()
    y = df["churned"].astype(int)

    logger.info(f"Data loaded: {df.shape[0]:,} rows")
    logger.info(f"Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def define_models():
    models = {
        "Dummy": Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
        "LR_default": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "LR_balanced": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))]),
        "DT_depth5": Pipeline([("model", DecisionTreeClassifier(max_depth=5, random_state=42))]),
        "RF_default": Pipeline([("model", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))]),
        "RF_balanced": Pipeline([("model", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42))])
    }
    logger.info(f"Defined {len(models)} models")
    return models


def run_cv_comparison(models, X, y, n_folds=5, random_state=42):
    logger.info(f"Running {n_folds}-fold Stratified CV...")
    results = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for name, pipeline in models.items():
        logger.info(f"Evaluating {name}...")
        scores = {'accuracy': [], 'f1': [], 'pr_auc': []}

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline[-1], "predict_proba") else y_pred

            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            scores['pr_auc'].append(average_precision_score(y_val, y_proba))

        results.append({
            'model': name,
            'accuracy_mean': pd.Series(scores['accuracy']).mean(),
            'f1_mean': pd.Series(scores['f1']).mean(),
            'pr_auc_mean': pd.Series(scores['pr_auc']).mean(),
        })

    return pd.DataFrame(results)


def save_comparison_table(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(path, index=False)
    logger.info(f"Comparison table saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Telecom Churn Model Comparison - Production CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to telecom_churn.csv")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save all results")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration without training models")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TELECOM CHURN MODEL COMPARISON PIPELINE")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("DRY-RUN MODE")
        logger.info(f"Data path   : {args.data_path}")
        logger.info(f"Output dir  : {args.output_dir}")
        logger.info(f"CV folds    : {args.n_folds}")
        logger.info(f"Random seed : {args.random_seed}")
        try:
            load_and_preprocess(args.data_path, args.random_seed)
            logger.info("✅ Dry-run completed successfully! Configuration is valid.")
            return
        except SystemExit:
            sys.exit(1)

    # Normal Run
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(args.data_path, args.random_seed)
        models = define_models()
        results_df = run_cv_comparison(models, X_train, y_train, args.n_folds, args.random_seed)

        logger.info("\nModel Comparison Results:")
        logger.info(results_df.to_string(index=False))

        save_comparison_table(results_df, args.output_dir)

        logger.info(f"✅ Pipeline completed successfully!")
        logger.info(f"📁 Results saved in: {os.path.abspath(args.output_dir)}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()