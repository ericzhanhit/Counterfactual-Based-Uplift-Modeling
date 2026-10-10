"""Cross-fitted augmented inverse-propensity outcome regression; Eq. (10)."""
import numpy as np
from .base import BaseUpliftEstimator, with_treatment


class DRLearner(BaseUpliftEstimator):
    def fit(self, X, treatment, y):
        X, t, y = self._data(X, treatment, y)
        m0, m1, p = (np.empty(len(y)) for _ in range(3))
        self.nuisance_splits_ = self._folds(t)
        for fold, (tr, va) in enumerate(self.nuisance_splits_):
            m = self._model(offset=10 * fold).fit(
                with_treatment(X.iloc[tr], t[tr]), y[tr]
            )
            m0[va] = m.predict(with_treatment(X.iloc[va], 0))
            m1[va] = m.predict(with_treatment(X.iloc[va], 1))
            p[va] = (
                self._model("binary", 10 * fold + 1)
                .fit(X.iloc[tr], t[tr])
                .predict(X.iloc[va])
            )
        self.propensity_ = np.clip(p, self.propensity_clip, 1 - self.propensity_clip)
        self.mu0_, self.mu1_ = m0, m1
        self.pseudo_outcome_ = (
            m1
            - m0
            + t * (y - m1) / self.propensity_
            - (1 - t) * (y - m0) / (1 - self.propensity_)
        )
        self.final_ = self._model("continuous", 100).fit(X, self.pseudo_outcome_)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.final_.predict(X)
