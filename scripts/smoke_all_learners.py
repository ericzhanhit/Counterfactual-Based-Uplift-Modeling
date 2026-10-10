"""Small binary-outcome integration run, all 50 learner/estimator/variant pairs.

Small RF/CatBoost/XGBoost models keep this a functionality check, not a
manuscript-quality model-selection experiment.
"""
import numpy as np
import pandas as pd
from cf_uplift.data.base import DatasetBundle
from cf_uplift.evaluation.pipeline import ExperimentConfig
from cf_uplift.evaluation.cross_validation import run_experiment

if __name__ == '__main__':
    rng = np.random.default_rng(24)
    X = pd.DataFrame({'x': rng.normal(size=300), 'category': rng.choice(['A', 'B'], size=300)})
    t = rng.binomial(1, .5, len(X))
    y = rng.binomial(1, 1/(1+np.exp(-(X.x+t))))
    data = DatasetBundle(X, t, y, metadata={'dataset': 'binary_outcome_toy', 'seed': 24})
    cfg = ExperimentConfig(outcome_type='binary', learners=['DT', 'RF', 'GBDT', 'CatBoost', 'XGBoost'],
                           counterfactual_folds=3, output_dir='outputs/all_learners_toy', make_figures=False,
                           learner_params={'RF': {'n_estimators': 10}, 'GBDT': {'n_estimators': 10},
                                           'CatBoost': {'iterations': 10},
                                           'XGBoost': {'n_estimators': 10, 'max_depth': 3}})
    run_experiment(data, cfg)
