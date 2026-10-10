"""Explicit dose-response interface for future continuous causal estimators."""
from typing import Protocol
import numpy as np
from .factory import ResponseModel
from .base import with_treatment


class DoseResponseEstimator(Protocol):
    def fit(self, X, treatment, y):
        ...

    def predict_response(self, X, dose) -> np.ndarray:
        ...

    def predict_effect(self, X, dose, reference) -> np.ndarray:
        ...


class ContinuousSLearner:
    """Outcome-regression dose-response extension, not a binary uplift estimator.

    Causal interpretation additionally requires continuous-treatment identification
    assumptions; factual prediction error does not validate those assumptions.
    """

    def __init__(self, config):
        self.config = config

    def fit(self, X, treatment, y):
        t = np.asarray(treatment, dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all() or len(np.unique(t)) < 2:
            raise ValueError("Need finite varying continuous treatments")
        self.range_ = (t.min(), t.max())
        self.model_ = ResponseModel(self.config).fit(with_treatment(X, t), y)
        return self

    def predict_response(self, X, dose):
        return self.model_.predict(with_treatment(X, dose))

    def predict_effect(self, X, dose, reference):
        return self.predict_response(X, dose) - self.predict_response(X, reference)
