"""Persist clearly labeled toy validation runs for advanced treatment modes."""
from pathlib import Path
import numpy as np
import pandas as pd
from cf_uplift.data.base import DatasetBundle
from cf_uplift.evaluation.pipeline import ExperimentConfig
from cf_uplift.counterfactual.optimizer import CounterfactualConfig
from cf_uplift.experiments.multi_treatment import run_multi
from cf_uplift.experiments.continuous_treatment import run_continuous

if __name__ == '__main__':
    X = pd.DataFrame({'x': np.repeat(np.linspace(-2, 2, 120), 3)})
    t = np.tile([0, 1, 2], 120)
    y = np.where(t == 0, 0, np.where(t == 1, 3*X.x, -3*X.x)).astype(float)
    multi = DatasetBundle(X, t, y, 'multi', {'dataset': 'balanced_multi_toy', 'not_hillstrom': True})
    config = ExperimentConfig(counterfactual_folds=3, output_dir='outputs/multi_toy', cf=CounterfactualConfig(lambda_gain=2))
    run_multi(multi, config)
    config.output_dir = 'outputs/multi_reproduction_toy'
    run_multi(multi, config, 'multi_reproduction')
    rng = np.random.default_rng(2)
    X = pd.DataFrame({'x': rng.normal(size=120)})
    treatment = rng.uniform(.2, 1.8, len(X))
    continuous = DatasetBundle(X, treatment, X.x+2*treatment, 'continuous', {'dataset': 'continuous_toy', 'not_dominicks': True})
    config = ExperimentConfig(estimators=['SL'], counterfactual_folds=3, output_dir='outputs/continuous_toy',
                              cf=CounterfactualConfig(lower_bound=0, upper_bound=2, n_candidates=20))
    run_continuous(continuous, config, 'continuous_general', {'dose_grid': [0, .5, 1, 1.5, 2], 'reference_dose': 1})
    config.output_dir = 'outputs/continuous_reproduction_toy'
    config.estimators = ['SL', 'TL', 'XL', 'DML', 'DRL']
    run_continuous(continuous, config, 'continuous_reproduction', {'binarization': 'train_median'})
