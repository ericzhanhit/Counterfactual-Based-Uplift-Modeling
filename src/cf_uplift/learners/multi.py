"""Native multi-arm extensions; all treatment categories remain in the model.

These are explicitly general extensions, not the manuscript's one-vs-rest analysis.
SL/TL use categorical outcome models. XL uses arm-specific imputation with a joint
multinomial propensity. DRL uses joint cross-fitted propensity augmentation. DML
uses a joint residual treatment matrix and linear heterogeneity basis; it does not
fit independent binary DML models. Base learners govern the nuisance models.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from ..data.base import frame, vectors
from ..data.preprocessing import make_preprocessor
from .factory import ResponseModel, get_base_learner
from .base import with_treatment, T_COLUMN


def categorical_treatment(X, t):
    result = with_treatment(X, t)
    result[T_COLUMN] = result[T_COLUMN].astype(float).astype(str)
    return result


class MultiTreatmentEstimator:
    def __init__(
        self,
        estimator,
        config,
        outcome_type="continuous",
        control=0,
        nuisance_folds=2,
        propensity_clip=0.01,
    ):
        if estimator not in ("SL", "TL", "XL", "DML", "DRL"):
            raise ValueError("Unknown multi-treatment estimator")
        self.estimator, self.config, self.outcome_type = estimator, config, outcome_type
        self.control, self.nuisance_folds, self.clip = (
            control,
            nuisance_folds,
            propensity_clip,
        )

    def _model(self, kind=None, offset=0):
        return ResponseModel(self.config, kind or self.outcome_type, offset)

    def _propensity(self, X, t, seed_offset=0):
        model = get_base_learner(
            self.config.name,
            "classification",
            self.config.params | self.config.classification_params,
            self.config.seed + seed_offset,
        )
        return Pipeline([("preprocess", make_preprocessor()), ("model", model)]).fit(
            X, t
        )

    def _probabilities(self, model, X):
        p = model.predict_proba(X)
        labels = model.named_steps["model"].classes_
        if set(labels) != set(self.arms_):
            raise ValueError("A nuisance training fold is missing a treatment arm")
        p = p[:, [list(labels).index(a) for a in self.arms_]]
        p = np.clip(p, self.clip, 1 - self.clip)
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, X, treatment, y):
        X = frame(X)
        t, y = vectors(treatment, y, len(X), binary=False)
        self.arms_ = np.unique(t)
        if self.control not in self.arms_ or len(self.arms_) < 2:
            raise ValueError(
                "Multi-treatment training requires control and active arms"
            )
        self.active_ = self.arms_[self.arms_ != self.control]
        self.control_index_ = list(self.arms_).index(self.control)
        if self.estimator == "SL":
            self.single_ = self._model().fit(categorical_treatment(X, t), y)
        elif self.estimator in ("TL", "XL"):
            self.outcomes_ = {
                a: self._model(offset=i).fit(X[t == a], y[t == a])
                for i, a in enumerate(self.arms_)
            }
            if self.estimator == "XL":
                self.propensity_ = self._propensity(X, t)
                self.effects_ = {}
                for i, arm in enumerate(self.active_):
                    ctrl, active = t == self.control, t == arm
                    d0 = self.outcomes_[arm].predict(X[ctrl]) - y[ctrl]
                    d1 = y[active] - self.outcomes_[self.control].predict(X[active])
                    self.effects_[arm] = (
                        self._model("continuous", i + 10).fit(X[ctrl], d0),
                        self._model("continuous", i + 20).fit(X[active], d1),
                    )
        else:
            nfold = min(self.nuisance_folds, np.unique(t, return_counts=True)[1].min())
            if nfold < 2:
                raise ValueError(
                    "Multi nuisance cross-fitting needs >=2 rows in every arm"
                )
            folds = StratifiedKFold(nfold, shuffle=True, random_state=self.config.seed)
            p = np.empty((len(y), len(self.arms_)))
            mu = np.empty_like(p)
            marginal = np.empty(len(y))
            for fold, (tr, va) in enumerate(folds.split(X, t)):
                propensity = self._propensity(X.iloc[tr], t[tr], fold)
                p[va] = self._probabilities(propensity, X.iloc[va])
                if self.estimator == "DRL":
                    response = self._model(offset=fold).fit(
                        categorical_treatment(X.iloc[tr], t[tr]), y[tr]
                    )
                    mu[va] = np.column_stack(
                        [
                            response.predict(categorical_treatment(X.iloc[va], a))
                            for a in self.arms_
                        ]
                    )
                else:
                    marginal[va] = (
                        self._model(offset=fold)
                        .fit(X.iloc[tr], y[tr])
                        .predict(X.iloc[va])
                    )
            if self.estimator == "DRL":
                pseudo = (
                    mu
                    + np.column_stack([t == a for a in self.arms_])
                    * (y[:, None] - mu)
                    / p
                )
                self.effects_ = {
                    a: self._model("continuous", i + 100).fit(
                        X, pseudo[:, i] - pseudo[:, self.control_index_]
                    )
                    for i, a in enumerate(self.arms_)
                    if a != self.control
                }
            else:
                self.preprocessor_ = make_preprocessor().fit(X)
                basis = np.column_stack(
                    [np.ones(len(X)), self.preprocessor_.transform(X)]
                )
                active_idx = [list(self.arms_).index(a) for a in self.active_]
                residual = (
                    np.column_stack([t == a for a in self.active_]) - p[:, active_idx]
                )
                design = (residual[:, :, None] * basis[:, None, :]).reshape(len(X), -1)
                self.coefficients_ = np.linalg.lstsq(design, y - marginal, rcond=None)[
                    0
                ].reshape(len(self.active_), -1)
        return self

    def predict_uplift(self, X):
        X = frame(X)
        if self.estimator == "SL":
            base = self.single_.predict(categorical_treatment(X, self.control))
            return np.column_stack(
                [
                    self.single_.predict(categorical_treatment(X, a)) - base
                    for a in self.active_
                ]
            )
        if self.estimator == "TL":
            base = self.outcomes_[self.control].predict(X)
            return np.column_stack(
                [self.outcomes_[a].predict(X) - base for a in self.active_]
            )
        if self.estimator == "XL":
            p = self._probabilities(self.propensity_, X)
            result = []
            for a in self.active_:
                j = list(self.arms_).index(a)
                weight = p[:, j] / (p[:, j] + p[:, self.control_index_])
                d0, d1 = self.effects_[a]
                result.append(weight * d0.predict(X) + (1 - weight) * d1.predict(X))
            return np.column_stack(result)
        if self.estimator == "DRL":
            return np.column_stack([self.effects_[a].predict(X) for a in self.active_])
        basis = np.column_stack([np.ones(len(X)), self.preprocessor_.transform(X)])
        return basis @ self.coefficients_.T

    def recommend(self, X):
        effects = np.column_stack([np.zeros(len(X)), self.predict_uplift(X)])
        return np.r_[self.control, self.active_][effects.argmax(axis=1)]
