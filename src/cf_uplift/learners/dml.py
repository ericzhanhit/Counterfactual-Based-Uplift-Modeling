"""Eq. (9) via cross-fitted residuals and an equivalent weighted squared loss."""
import numpy as np
from .base import BaseUpliftEstimator


class DML(BaseUpliftEstimator):
    def fit(self, X, treatment, y):
        X, t, y = self._data(X, treatment, y)
        m, p = np.empty(len(y)), np.empty(len(y))
        self.nuisance_splits_ = self._folds(t)
        for fold, (tr, va) in enumerate(self.nuisance_splits_):
            m[va] = (
                self._model(offset=10 * fold).fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
            )
            p[va] = (
                self._model("binary", 10 * fold + 1)
                .fit(X.iloc[tr], t[tr])
                .predict(X.iloc[va])
            )
        p = np.clip(p, self.propensity_clip, 1 - self.propensity_clip)
        self.outcome_residual_ = y - m
        self.treatment_residual_ = t - p
        self.residual_tolerance_ = 1e-8
        keep = np.abs(self.treatment_residual_) > self.residual_tolerance_
        if not keep.any():
            raise ValueError("DML has no nonzero treatment residuals")
        self.final_ = self._model("continuous", 100).fit(
            X[keep],
            self.outcome_residual_[keep] / self.treatment_residual_[keep],
            sample_weight=self.treatment_residual_[keep] ** 2,
        )
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.final_.predict(X)
