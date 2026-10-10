"""Eqs. (19)–(21); unspecified covariance/noise/propensity recovered from old code."""
import numpy as np
import pandas as pd
from .base import DatasetBundle


def generate_synthetic_data(
    n_samples: int = 10000, seed: int = 42, n_features: int = 10, noise_std: float = 1.0
) -> DatasetBundle:
    if n_samples < 2 or n_features < 3 or noise_std < 0:
        raise ValueError("Need n_samples>=2, n_features>=3, noise_std>=0")
    rng = np.random.default_rng(seed)
    beta = rng.uniform(-3, 3, n_features)
    X = rng.multivariate_normal(np.zeros(n_features), np.eye(n_features), n_samples)
    propensity = np.where((X[:, 2] > -0.5) & (X[:, 2] < 0.5), 0.8, 0.2)
    treatment = rng.binomial(1, propensity)
    tau = 8.0 * (X[:, 1] > 0.1)
    y = X @ beta + rng.normal(0, noise_std, n_samples) + treatment * tau
    return DatasetBundle(
        pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)]),
        treatment,
        y,
        metadata={
            "dataset": "synthetic",
            "seed": seed,
            "beta": beta.tolist(),
            "covariance": "identity",
            "noise_std": noise_std,
            "true_tau": tau.tolist(),
            "propensity": propensity.tolist(),
        },
    )
