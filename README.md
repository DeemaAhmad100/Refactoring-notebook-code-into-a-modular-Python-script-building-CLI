# Telecom Churn Model Comparison - Production CLI

A clean, production-ready command-line tool that compares 6 machine learning models on the Telecom Churn dataset using stratified 5-fold cross-validation.

## Features
- Full support for `argparse` CLI with clear help messages
- Structured logging using Python's `logging` module
- `--dry-run` mode for safe configuration validation
- Saves all results (comparison table, best model, plots) to the output directory
- Compares: Dummy Classifier, Logistic Regression (default + balanced), Decision Tree, Random Forest (default + balanced)

## Installation

```bash
pip install -r requirements.txt
Usage Examples
1. Dry-run (recommended before full run)
Bashpython compare_models.py --data-path data/telecom_churn.csv --dry-run
2. Full Model Comparison
Bashpython compare_models.py --data-path data/telecom_churn.csv
3. Full run with custom options
Bashpython compare_models.py \
  --data-path data/telecom_churn.csv \
  --output-dir ./my_results \
  --n-folds 10 \
  --random-seed 123
Command Line Arguments









































ArgumentRequiredDefaultDescription--data-pathYes-Path to the telecom_churn.csv file--output-dirNo./outputDirectory where results will be saved--n-foldsNo5Number of cross-validation folds--random-seedNo42Random seed for reproducibility--dry-runNoFalseValidate data and config without training models
Output Files (saved in --output-dir)

model_comparison.csv → Full comparison table with mean scores
best_model.joblib     → The best performing model (by PR-AUC)
pr_curves.png         → Precision-Recall curves for top 3 models

Project Structure
text.
├── compare_models.py
├── requirements.txt
├── README.md
├── data/
│   └── telecom_churn.csv
└── output/
    ├── model_comparison.csv
    ├── best_model.joblib
    └── pr_curves.png
Requirements
See requirements.txt