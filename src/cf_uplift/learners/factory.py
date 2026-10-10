"""Seeded base learners, lazy optional dependencies, and probability-safe response models."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.utils.validation import has_fit_parameter
from ..data.base import frame
from ..data.preprocessing import make_preprocessor


@dataclass
class LearnerConfig:
    name: str = "DT"
    params: dict[str, Any] = field(default_factory=dict)
    classification_params: dict[str, Any] = field(default_factory=dict)
    regression_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42


def get_base_learner(name: str, task: str, config: dict | None = None, seed: int = 42):
    if task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    name = name.upper()
    choices = {
        "DT": (DecisionTreeClassifier, DecisionTreeRegressor),
        "RF": (RandomForestClassifier, RandomForestRegressor),
        "GBDT": (GradientBoostingClassifier, GradientBoostingRegressor),
    }
    if name == "CATBOOST":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "CatBoost is optional. Install: pip install '.[catboost]'"
            ) from exc
        choices[name] = (CatBoostClassifier, CatBoostRegressor)
    if name == "XGBOOST":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is optional. Install: pip install '.[xgboost]'"
            ) from exc
        choices[name] = (XGBClassifier, XGBRegressor)
    if name not in choices:
        raise ValueError(
            f"Unknown base learner {name}; choose DT/RF/GBDT/CatBoost/XGBoost"
        )
    defaults = {
        "DT": {
            "max_depth": 8,
            "min_samples_leaf": 20,
        },
        "RF": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 10,
            "n_jobs": -1,
        },
        "GBDT": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 10,
        },
        "CATBOOST": {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.05,
            "thread_count": -1,
        },
        "XGBOOST": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
        },
    }
    params = defaults[name] | (config or {})
    params["random_seed" if name == "CATBOOST" else "random_state"] = seed
    if name == "XGBOOST":
        params.setdefault(
            "objective",
            "binary:logistic" if task == "classification" else "reg:squarederror",
        )
        if task == "classification":
            params.setdefault("eval_metric", "logloss")
    return choices[name][task == "regression"](**params)


class ResponseModel:
    """Raw-DataFrame model with training-only preprocessing and soft predictions."""

    def __init__(
        self,
        config: LearnerConfig,
        outcome_type: str = "continuous",
        seed_offset: int = 0,
    ):
        if outcome_type not in ("binary", "continuous"):
            raise ValueError("outcome_type must be binary or continuous")
        self.config, self.outcome_type, self.seed_offset = (
            config,
            outcome_type,
            seed_offset,
        )

    def fit(self, X, y, sample_weight=None):
        X, y = frame(X), np.asarray(y, dtype=float)
        # Internal treatment feature is carried under a normal explicit name.
        if len(X) != len(y) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Invalid response data")
        classification = self.outcome_type == "binary"
        if classification and not np.isin(y, [0, 1]).all():
            raise ValueError(
                "Binary outcomes must be 0/1; predicted CF probabilities require continuous regression"
            )
        task = "classification" if classification else "regression"
        params = self.config.params | (
            self.config.classification_params
            if classification
            else self.config.regression_params
        )
        model = get_base_learner(
            self.config.name, task, params, self.config.seed + self.seed_offset
        )
        if classification and len(np.unique(y)) == 1:
            model = DummyClassifier(strategy="constant", constant=int(y[0]))
        self.pipeline_ = Pipeline(
            [("preprocess", make_preprocessor()), ("model", model)]
        )
        # Learners without native sample_weight support fall back to an explicit
        # deterministic weighted resample for DML's weighted final regression.
        kwargs = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if (
                weights.shape != y.shape
                or not np.isfinite(weights).all()
                or (weights < 0).any()
            ):
                raise ValueError(
                    "sample_weight must be finite, nonnegative, and match y"
                )
            if has_fit_parameter(model, "sample_weight"):
                kwargs["model__sample_weight"] = weights
            else:
                total = weights.sum()
                if total <= 0:
                    raise ValueError("sample_weight must have positive total weight")
                rng = np.random.default_rng(self.config.seed + self.seed_offset)
                chosen = rng.choice(
                    len(y), size=len(y), replace=True, p=weights / total
                )
                X, y = X.iloc[chosen].reset_index(drop=True), y[chosen]
        self.pipeline_.fit(X, y, **kwargs)
        self.columns_ = list(X.columns)
        return self

    def predict(self, X) -> np.ndarray:
        X = frame(X)
        if list(X.columns) != self.columns_:
            raise ValueError("Prediction feature names/order differ from training")
        if self.outcome_type == "continuous":
            return np.asarray(self.pipeline_.predict(X), dtype=float)
        model = self.pipeline_.named_steps["model"]
        probabilities = self.pipeline_.predict_proba(X)
        positive = np.flatnonzero(model.classes_ == 1)
        return probabilities[:, positive[0]] if len(positive) else np.zeros(len(X))
