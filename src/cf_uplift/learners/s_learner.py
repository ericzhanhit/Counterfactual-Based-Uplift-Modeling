"""Single-model learner; Eq. (6)."""
import numpy as np
from .base import BaseUpliftEstimator, with_treatment


class SLearner(BaseUpliftEstimator):
    def fit(self, X, treatment, y):
        X, t, y = self._data(X, treatment, y)
        self.model_ = self._model().fit(with_treatment(X, t), y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.model_.predict(with_treatment(X, 1)) - self.model_.predict(
            with_treatment(X, 0)
        )
