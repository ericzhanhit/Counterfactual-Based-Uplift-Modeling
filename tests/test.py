"""Run the full synthetic-data benchmark and write its result CSV files.

This is an executable integration benchmark rather than a pytest unit test:

    python tests/test.py

The default run fits every combination of five base learners, five uplift
learners, and the benchmark/counterfactual training variants.
"""
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_uplift.data.synthetic import generate_synthetic_data
from cf_uplift.evaluation.cross_validation import run_experiment
from cf_uplift.evaluation.pipeline import ExperimentConfig


N_SAMPLES = 10_000
N_FEATURES = 10
SEED = 42

# Parameters are deliberately explicit so that the benchmark is reproducible.
# The ensemble sizes are large enough for a substantive integration run while
# remaining practical across all 50 base/uplift/variant combinations.
BASE_LEARNER_PARAMETERS = {
    "DT": {
        "max_depth": 8,
        "min_samples_leaf": 20,
    },
    "RF": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "n_jobs": -1,
    },
    "GBDT": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 10,
    },
    "CatBoost": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.05,
        "thread_count": -1,
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
    },
}


def main():
    data = generate_synthetic_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        seed=SEED,
    )
    assert data.X.shape == (N_SAMPLES, N_FEATURES)
    assert data.treatment.shape == (N_SAMPLES,)
    assert data.outcome.shape == (N_SAMPLES,)

    output_dir = PROJECT_ROOT / "outputs" / "synthetic_all_learners"
    config = ExperimentConfig(
        outcome_type="continuous",
        learners=["DT", "RF", "GBDT", "CatBoost", "XGBoost"],
        estimators=["SL", "TL", "XL", "DML", "DRL"],
        variants=["benchmark", "cf"],
        learner_params=BASE_LEARNER_PARAMETERS,
        counterfactual_folds=10,
        nuisance_folds=10,
        propensity_clip=0.10,
        test_size=0.30,
        seed=SEED,
        output_dir=str(output_dir),
        make_figures=False,
        show_progress=True,
        cf={
            "method": "manuscript_random_search",
            "n_candidates": 100,
            "lambda_gain": 1.0,
            "exclude_original": True,
            "target": "predicted",
            "training_strategy": "augment",
        },
    )

    result = run_experiment(data, config)

    print(f"\nResults written to: {output_dir}")
    print("\nTest metrics:")
    print(result["metrics"].to_string(index=False))
    print("\nSummary:")
    print(result["summary"].to_string())
    print("\nCounterfactual summary:")
    print((output_dir / "metrics" / "counterfactual_summary.csv").read_text())


if __name__ == "__main__":
    main()
