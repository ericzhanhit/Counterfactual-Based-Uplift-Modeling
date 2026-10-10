# Counterfactual Uplift Modeling

This repository provides a research implementation for training and evaluating
uplift models with counterfactual treatment generation. It supports standard
uplift learners, counterfactual-data variants, cross-fitted nuisance models,
synthetic data generation, and CSV-based experiment outputs.

## Supported models

The full synthetic benchmark combines five base learners with five uplift
learners.

Base learners:

- Decision Tree (`DT`)
- Random Forest (`RF`)
- Gradient Boosting Decision Tree (`GBDT`)
- CatBoost (`CatBoost`)
- XGBoost (`XGBoost`)

Uplift learners:

- S-Learner (`SL`)
- T-Learner (`TL`)
- X-Learner (`XL`)
- Double Machine Learning (`DML`)
- Doubly Robust Learner (`DRL`)

Each combination is evaluated in two forms:

- The standard benchmark model, such as `RF__TL`
- The counterfactual-training model, such as `RF__CF-TL`

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- SciPy
- scikit-learn
- CatBoost
- XGBoost
- Matplotlib
- tqdm (recommended for the progress bar)
- pytest (for the unit-test suite)

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn catboost xgboost matplotlib tqdm pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the full synthetic experiment

From the repository root, run:

```bash
python tests/test.py
```

The script generates a synthetic dataset containing:

- 10,000 observations
- 10 features
- One binary treatment variable
- One continuous outcome variable

It then trains all 25 base-learner/uplift-learner combinations using both the
standard and counterfactual variants, producing 50 fitted uplift models in
total. The experiment uses a 70% training and 30% test split, 10-fold
counterfactual cross-fitting, and 10-fold nuisance-model cross-fitting.

The model parameters and random seed are defined explicitly in
[`tests/test.py`](tests/test.py), so they can be adjusted before running the
experiment.

> **Runtime:** This is a full integration benchmark rather than a small unit
> test. Runtime depends on the computer, but the experiment may take several
> minutes because DML and DRL repeatedly fit nuisance models.

## Experiment outputs

The synthetic experiment writes its results to:

```text
outputs/synthetic_all_learners/
```

Important result files include:

```text
outputs/synthetic_all_learners/
├── metrics/
│   ├── test_metrics.csv
│   ├── summary.csv
│   ├── counterfactual_metrics.csv
│   └── counterfactual_summary.csv
├── counterfactuals/
│   └── training.csv
├── predictions/
│   └── test.csv
├── curves/
│   └── test_{model}.csv
└── ranking_tables/
    └── test_{model}.csv
```

`test_metrics.csv` reports the following test-set metrics:

- `auuc`
- `qini_coefficient`
- `auuc_normalized`
- `qini_normalized`

`counterfactual_summary.csv` reports:

- Counterfactual learner
- Number of generated and valid counterfactuals
- Validity
- Proximity
- Implausibility

The complete metric and counterfactual summaries are also printed to the
terminal after a successful run. Running the experiment again with the same
output directory replaces CSV files with results from the new run.

## Run the unit tests

The regular unit-test suite is separate from the full benchmark. Run it from
the repository root with:

```bash
PYTHONPATH=src python -m pytest -q
```

Windows PowerShell users can run:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

## Repository structure

```text
cf_uplift/
├── src/cf_uplift/
│   ├── counterfactual/   # Counterfactual treatment generation
│   ├── data/             # Dataset containers, loaders, and synthetic data
│   ├── evaluation/       # Experiment orchestration and cross-fitting
│   ├── experiments/      # Binary, multi-treatment, and continuous runners
│   ├── learners/         # Uplift and response-model implementations
│   ├── metrics/          # Uplift and counterfactual metrics
│   └── visualization/    # Plotting utilities
├── tests/
│   ├── test.py           # Full 10,000-sample integration benchmark
│   └── test_*.py         # Unit and integration tests
└── outputs/              # Generated experiment results
```

## Reproducibility

The synthetic benchmark uses a fixed random seed (`42`). Training/test
splitting, cross-fitting, counterfactual generation, and supported base learners
are seeded to make repeated runs reproducible under the same software and
hardware environment.
