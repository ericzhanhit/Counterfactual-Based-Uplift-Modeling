"""Common estimator interface and fold-local nuisance helpers."""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ..data.base import frame, vectors
from .factory import LearnerConfig, ResponseModel

# Appending T to X is isolated from user columns; reserved name validated below.
T_COLUMN = "_cf_model_treatment"


def with_treatment(X, treatment):
    result = frame(X)
    if T_COLUMN in result:
        raise ValueError(f"{T_COLUMN} is reserved")
    result[T_COLUMN] = treatment
    return result


class BaseUpliftEstimator(ABC):
    def __init__(
        self,
        config: LearnerConfig | None = None,
        outcome_type: str = "continuous",
        nuisance_folds: int = 2,
        propensity_clip: float = 0.01,
    ):
        self.config = config or LearnerConfig()
        self.outcome_type, self.nuisance_folds = outcome_type, nuisance_folds
        if not 0 < propensity_clip < 0.5:
            raise ValueError("propensity_clip must be in (0, .5)")
        self.propensity_clip = propensity_clip

    def _data(self, X, treatment, y):
        X = frame(X)
        t, y = vectors(treatment, y, len(X))
        if len(np.unique(t)) != 2:
            raise ValueError(
                "Both treatment arms are required; CF assignment may have collapsed an arm"
            )
        if self.outcome_type == "binary" and not np.isin(y, [0, 1]).all():
            raise ValueError("Binary outcomes must be 0/1")
        return X, t.astype(int), y

    def _model(self, kind=None, offset=0):
        return ResponseModel(self.config, kind or self.outcome_type, offset)

    def _folds(self, t):
        n = min(self.nuisance_folds, int(np.bincount(t).min()))
        if n < 2:
            raise ValueError(
                "Cross-fitting requires at least two observations in each arm"
            )
        return list(
            StratifiedKFold(n, shuffle=True, random_state=self.config.seed).split(t, t)
        )

    @abstractmethod
    def fit(self, X, treatment, y):
        ...

    @abstractmethod
    def predict_uplift(self, X) -> np.ndarray:
        ...
