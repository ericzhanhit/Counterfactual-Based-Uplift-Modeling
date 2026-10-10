"""Separate outcome models; Eq. (7)."""
import numpy as np
from .base import BaseUpliftEstimator


class TLearner(BaseUpliftEstimator):
    def fit(self, X, treatment, y):
        X, t, y = self._data(X, treatment, y)
        self.models_ = [self._model(offset=a).fit(X[t == a], y[t == a]) for a in (0, 1)]
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.models_[1].predict(X) - self.models_[0].predict(X)
