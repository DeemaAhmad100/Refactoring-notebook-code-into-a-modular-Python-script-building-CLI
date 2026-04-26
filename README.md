# Refactoring-notebook-code-into-a-modular-Python-script-building-CLI

# Telecom Churn Model Comparison - Production CLI

A production-ready command-line tool that compares 6 machine learning models on the Telecom Churn dataset using stratified cross-validation.

## Features
- Supports `--dry-run` for safe configuration validation
- Uses `argparse` for clean CLI interface
- Structured logging instead of print statements
- Saves all results to the specified output directory
- Compares: Dummy, Logistic Regression (default + balanced), Decision Tree, Random Forest (default + balanced)

## Installation

```bash
pip install -r requirements.txt



Usage
Dry-run (recommended first)
Bashpython compare_models.py --data-path data/telecom_churn.csv --dry-run
Full run
Bashpython compare_models.py --data-path data/telecom_churn.csv
With custom options
Bashpython compare_models.py \
  --data-path data/telecom_churn.csv \
  --output-dir ./my_results \
  --n-folds 10 \
  --random-seed 123
Arguments









































ArgumentRequiredDefaultDescription--data-pathYes-Path to telecom_churn.csv--output-dirNo./outputDirectory to save results--n-foldsNo5Number of CV folds--random-seedNo42Random seed for reproducibility--dry-runNoFalseValidate without training models
Output
The script creates the following in the output directory:

model_comparison.csv - Full comparison table with mean scores


# requirements.txt
pandas
numpy
scikit-learn
matplotlib
joblib
