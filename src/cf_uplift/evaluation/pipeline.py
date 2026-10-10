"""Fit observed and counterfactual estimators using training data only."""
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from ..data.base import DatasetBundle
from ..learners import LearnerConfig, make_estimator
from ..learners.factory import ResponseModel
from ..learners.base import with_treatment
from ..counterfactual.optimizer import (
    CounterfactualConfig,
    CounterfactualTreatmentOptimizer,
)
from ..metrics.counterfactual import (
    validity_conditioned_implausibility,
)


@dataclass
class ExperimentConfig:
    outcome_type: str = "continuous"
    learners: list[str] = field(default_factory=lambda: ["DT"])
    estimators: list[str] = field(
        default_factory=lambda: ["SL", "TL", "XL", "DML", "DRL"]
    )
    variants: list[str] = field(default_factory=lambda: ["benchmark", "cf"])
    learner_params: dict = field(default_factory=dict)
    counterfactual_folds: int = 10
    nuisance_folds: int = 2
    test_size: float = 0.3
    shuffle: bool = True
    seed: int = 42
    n_bins: int = 10
    propensity_clip: float = 0.01
    undefined_prefix: str = "interpolate"
    output_dir: str = "outputs/binary"
    save_counterfactuals: bool = True
    save_candidates: bool = False
    make_figures: bool = True
    show_progress: bool = True
    progress_description: str = "Training uplift models"
    cf: CounterfactualConfig = field(default_factory=CounterfactualConfig)

    def __post_init__(self):
        if isinstance(self.cf, dict):
            self.cf = CounterfactualConfig(**self.cf)
        if (
            self.outcome_type not in ("binary", "continuous")
            or self.counterfactual_folds < 2
            or self.nuisance_folds < 2
        ):
            raise ValueError("Invalid outcome type or fold counts")
        if not 0 < self.test_size < 1 or self.n_bins < 1:
            raise ValueError("Invalid test_size or n_bins")
        if not self.variants or not set(self.variants) <= {"benchmark", "cf"}:
            raise ValueError("variants must contain benchmark and/or cf")
        if not self.learners or not self.estimators:
            raise ValueError("Select at least one learner and estimator")
        if len(set(self.learners)) != len(self.learners) or len(
            set(self.estimators)
        ) != len(self.estimators):
            raise ValueError("Duplicate learners/estimators")


def counterfactual_training_sample(X, treatment, outcome, result, config):
    """Construct Stage-2 data and A_i consistently across experiment runners."""
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    if config.exclude_original:
        selected_t = result.treatment
        selected_y = result.outcome
        selected_objective = result.objective
    else:
        selected_t = result.relaxed_treatment
        selected_y = result.relaxed_outcome
        selected_objective = result.relaxed_objective
    accepted = config.acceptance_mask(
        selected_objective, treatment, result.factual_outcome, selected_y
    )
    if config.training_strategy == "replace":
        fit_X = X
        fit_t = selected_t
        fit_y = outcome if config.target == "observed" else selected_y
    elif config.training_strategy == "selective":
        fit_X, fit_t, fit_y = X, treatment.copy(), outcome.copy()
        fit_t[accepted] = selected_t[accepted]
        fit_y[accepted] = selected_y[accepted]
    else:  # factual-plus-accepted-counterfactual augmentation
        fit_X = pd.concat([X, X.iloc[np.flatnonzero(accepted)]], ignore_index=True)
        fit_t = np.concatenate([treatment, selected_t[accepted]])
        fit_y = np.concatenate([outcome, selected_y[accepted]])
    return fit_X, fit_t, fit_y, accepted


def counterfactual_splits(treatment, outcome, n_folds, seed):
    """Training-only folds for Algorithm 1 counterfactual cross-fitting."""
    treatment, outcome = np.asarray(treatment), np.asarray(outcome)
    n_folds = min(n_folds, len(treatment))
    if n_folds < 2:
        raise ValueError("Counterfactual cross-fitting needs at least two rows")
    joint = np.array([f"{a}_{b}" for a, b in zip(treatment, outcome)])
    treatment_counts = np.unique(treatment, return_counts=True)[1]
    joint_counts = np.unique(joint, return_counts=True)[1]
    if joint_counts.min() >= n_folds:
        splitter, strata = (
            StratifiedKFold(n_folds, shuffle=True, random_state=seed),
            joint,
        )
    elif treatment_counts.min() >= n_folds:
        splitter, strata = (
            StratifiedKFold(n_folds, shuffle=True, random_state=seed),
            treatment,
        )
    else:
        splitter, strata = KFold(n_folds, shuffle=True, random_state=seed), None
    return list(splitter.split(np.zeros(len(treatment)), strata))


class FittedFold:
    """Reusable training-set model. predict accepts X only, never test Y/T."""

    def __init__(
        self, config: ExperimentConfig, seed: int | None = None, progress_callback=None
    ):
        self.config, self.seed = config, config.seed if seed is None else seed
        self.progress_callback = progress_callback

    def fit(self, training: DatasetBundle):
        if training.treatment_type != "binary":
            raise ValueError(
                "FittedFold requires binary data; use advanced experiment entry points"
            )
        cfg = self.config
        self.training_ids_ = training.row_ids.copy()
        self.models_, self.cf_tables_, self.cf_metrics_, self.candidate_tables_ = (
            {},
            [],
            [],
            [],
        )
        X, t, y = training.X, training.treatment, training.outcome
        for base in cfg.learners:
            learner_cfg = LearnerConfig(
                base, params=cfg.learner_params.get(base, {}), seed=self.seed
            )
            cf_result = None
            if "cf" in cfg.variants:
                teacher_name = cfg.cf.teacher_learner or base
                teacher_params = (
                    cfg.cf.teacher_params
                    if cfg.cf.teacher_learner
                    else cfg.learner_params.get(base, {})
                )
                teacher_cfg = LearnerConfig(
                    teacher_name, params=teacher_params, seed=self.seed
                )
                cf_result = self._generate_oof_counterfactuals(training, teacher_cfg)
                table, metrics = describe_counterfactuals(
                    training, cf_result, cfg, base
                )
                cf_X, cf_t, cf_y, accepted = counterfactual_training_sample(
                    X, t, y, cf_result, cfg.cf
                )
                self.cf_tables_.append(table)
                self.cf_metrics_.append(metrics)
                if cfg.save_candidates:
                    for mode, candidates_by_row, outcomes_by_row in (
                        (
                            "forced",
                            cf_result.candidates,
                            cf_result.candidate_outcomes,
                        ),
                        (
                            "relaxed",
                            cf_result.relaxed_candidates,
                            cf_result.relaxed_candidate_outcomes,
                        ),
                    ):
                        for row_id, candidates, outcomes in zip(
                            training.row_ids, candidates_by_row, outcomes_by_row
                        ):
                            self.candidate_tables_.append(
                                pd.DataFrame(
                                    {
                                        "row_id": row_id,
                                        "learner": base,
                                        "candidate_mode": mode,
                                        "treatment_cf": candidates,
                                        "predicted_outcome": outcomes,
                                    }
                                )
                            )
            for variant in cfg.variants:
                fit_X = X
                if variant == "benchmark":
                    fit_t, fit_y = t, y
                else:
                    fit_X, fit_t, fit_y = cf_X, cf_t, cf_y
                kind = (
                    cfg.outcome_type
                    if variant == "benchmark" or cfg.cf.target == "observed"
                    else "continuous"
                )
                for name in cfg.estimators:
                    key = f"{base}__{'CF-' if variant == 'cf' else ''}{name}"
                    self.models_[key] = make_estimator(
                        name,
                        config=learner_cfg,
                        outcome_type=kind,
                        nuisance_folds=cfg.nuisance_folds,
                        propensity_clip=cfg.propensity_clip,
                    ).fit(fit_X, fit_t, fit_y)
                    if self.progress_callback is not None:
                        self.progress_callback(key)
        return self

    def _generate_oof_counterfactuals(self, training, learner_cfg):
        """Implements Algorithm 1 Steps 3-17 with strict OOF generation."""
        X, t, y = training.X, training.treatment, training.outcome
        splits = counterfactual_splits(
            t, y, self.config.counterfactual_folds, self.seed
        )
        chosen_t = np.empty(len(X))
        chosen_y = np.empty(len(X))
        factual_y = np.empty(len(X))
        objective = np.empty(len(X))
        valid = np.empty(len(X), dtype=bool)
        candidates, candidate_outcomes = [None] * len(X), [None] * len(X)
        relaxed_t = np.empty(len(X))
        relaxed_y = np.empty(len(X))
        relaxed_objective = np.empty(len(X))
        relaxed_candidates, relaxed_candidate_outcomes = [None] * len(X), [None] * len(
            X
        )
        self.cf_generation_splits_ = getattr(self, "cf_generation_splits_", {})
        split_records = []
        for fold, (fit_idx, held_idx) in enumerate(splits):
            if set(fit_idx) & set(held_idx):
                raise AssertionError(
                    "Algorithm 1 OOF fitting and held-out indices overlap"
                )
            response = ResponseModel(
                learner_cfg, self.config.outcome_type, seed_offset=100 + fold
            ).fit(with_treatment(X.iloc[fit_idx], t[fit_idx]), y[fit_idx])
            optimizer = CounterfactualTreatmentOptimizer(
                self.config.cf, self.seed + fold
            )
            result = optimizer.generate(
                X.iloc[held_idx],
                t[held_idx],
                y[held_idx],
                lambda features, tx: response.predict(with_treatment(features, tx)),
                outcome_type=self.config.outcome_type,
            )
            chosen_t[held_idx], chosen_y[held_idx] = result.treatment, result.outcome
            factual_y[held_idx] = result.factual_outcome
            objective[held_idx], valid[held_idx] = result.objective, result.valid
            relaxed_t[held_idx] = result.relaxed_treatment
            relaxed_y[held_idx] = result.relaxed_outcome
            relaxed_objective[held_idx] = result.relaxed_objective
            for local, original in enumerate(held_idx):
                candidates[original] = result.candidates[local]
                candidate_outcomes[original] = result.candidate_outcomes[local]
                relaxed_candidates[original] = result.relaxed_candidates[local]
                relaxed_candidate_outcomes[
                    original
                ] = result.relaxed_candidate_outcomes[local]
            split_records.append(
                {
                    "fold": fold,
                    "fit_ids": training.row_ids[fit_idx].copy(),
                    "held_out_ids": training.row_ids[held_idx].copy(),
                }
            )
        if any(x is None for x in candidates) or not np.isfinite(chosen_y).all():
            raise AssertionError(
                "Every training row must receive exactly one OOF counterfactual"
            )
        self.cf_generation_splits_[learner_cfg.name] = split_records
        from ..counterfactual.optimizer import CounterfactualResult

        return CounterfactualResult(
            chosen_t,
            chosen_y,
            factual_y,
            objective,
            valid,
            candidates,
            candidate_outcomes,
            relaxed_t,
            relaxed_y,
            relaxed_objective,
            relaxed_candidates,
            relaxed_candidate_outcomes,
        )

    def predict(self, X) -> dict[str, np.ndarray]:
        return {name: model.predict_uplift(X) for name, model in self.models_.items()}


def describe_counterfactuals(training, result, cfg, base):
    t = training.treatment.astype(float)
    implausibility_value, implausibility_distances = validity_conditioned_implausibility(
        training.X,
        result.treatment,
        result.valid,
        training.X,
        training.treatment,
        return_distances=True,
    )
    if training.treatment_type == "binary":
        recommended = np.where(
            result.outcome > result.factual_outcome, result.treatment, t
        )
        proximity_values = np.abs(recommended - t)
        proximity_value = float(proximity_values.mean())
    else:
        proximity_value = np.nan
        proximity_values = np.full(len(t), np.nan)
    directional_uplift = (
        np.where(
            t == 0,
            result.outcome - result.factual_outcome,
            result.factual_outcome - result.outcome,
        )
        if training.treatment_type == "binary"
        else np.full(len(t), np.nan)
    )
    table = pd.DataFrame(
        {
            "row_id": training.row_ids,
            "learner": base,
            "treatment_observed": t,
            "outcome_observed": training.outcome,
            "treatment_cf": result.treatment,
            "outcome_factual_predicted": result.factual_outcome,
            "outcome_cf_predicted": result.outcome,
            "directional_uplift_predicted": directional_uplift,
            "objective": result.objective,
            "valid": result.valid,
            "proximity": proximity_values,
            "implausibility": implausibility_distances,
        }
    )
    metrics = {
        "learner": base,
        "n_generation": len(t),
        "n_valid": int(result.valid.sum()),
        "validity": float(result.valid.mean()),
        "proximity": proximity_value,
        "implausibility": implausibility_value,
    }
    return table, metrics
