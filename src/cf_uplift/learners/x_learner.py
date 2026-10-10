"""Imputed-effect regression and propensity aggregation; Eq. (8)."""
import numpy as np
from .base import BaseUpliftEstimator


class XLearner(BaseUpliftEstimator):
    def fit(self, X, treatment, y):
        X, t, y = self._data(X, treatment, y)
        m0 = self._model().fit(X[t == 0], y[t == 0])
        m1 = self._model(offset=1).fit(X[t == 1], y[t == 1])
        self.d0_ = self._model("continuous", 2).fit(
            X[t == 0], m1.predict(X[t == 0]) - y[t == 0]
        )
        self.d1_ = self._model("continuous", 3).fit(
            X[t == 1], y[t == 1] - m0.predict(X[t == 1])
        )
        self.propensity_ = self._model("binary", 4).fit(X, t)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        p = np.clip(
            self.propensity_.predict(X), self.propensity_clip, 1 - self.propensity_clip
        )
        return p * self.d0_.predict(X) + (1 - p) * self.d1_.predict(X)
