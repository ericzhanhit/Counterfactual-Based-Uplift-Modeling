"""Leakage-free comparison and selection of counterfactual response teachers."""
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..counterfactual.optimizer import (
    CounterfactualConfig,
    CounterfactualTreatmentOptimizer,
)
from ..data.base import DatasetBundle
from ..learners.base import with_treatment
from ..learners.factory import LearnerConfig, ResponseModel
from ..metrics.counterfactual import (
    binary_treatment_distance,
    binary_treatment_proximity,
    implausibility,
    policy_proximity,
    validity_conditioned_implausibility,
)
from .cross_validation import holdout_indices, write_json
from .pipeline import counterfactual_splits
from .progress import ExperimentProgress


@dataclass
class CFComparisonConfig:
    """Configuration for teacher selection without fitting uplift estimators."""

    teachers: list[str] = field(
        default_factory=lambda: ["DT", "RF", "GBDT", "CATBOOST", "XGBOOST"]
    )
    teacher_params: dict = field(default_factory=dict)
    outcome_type: str = "continuous"
    counterfactual_folds: int = 10
    test_size: float = 0.30
    shuffle: bool = True
    seed: int = 0
    output_dir: str = "outputs/cf_teacher_comparison"
    show_progress: bool = True
    save_oof_counterfactuals: bool = True
    cf: CounterfactualConfig = field(default_factory=CounterfactualConfig)

    def __post_init__(self):
        if isinstance(self.cf, dict):
            self.cf = CounterfactualConfig(**self.cf)
        self.teachers = [name.upper() for name in self.teachers]
        normalized_params = {}
        for name, params in self.teacher_params.items():
            key = name.upper()
            if key in normalized_params:
                raise ValueError("Duplicate case-insensitive teacher_params keys")
            if not isinstance(params, dict):
                raise ValueError("Each teacher_params value must be a dictionary")
            normalized_params[key] = params
        self.teacher_params = normalized_params
        if not self.teachers or len(set(self.teachers)) != len(self.teachers):
            raise ValueError("teachers must be a nonempty list without duplicates")
        allowed = {"DT", "RF", "GBDT", "CATBOOST", "XGBOOST"}
        if not set(self.teachers) <= allowed:
            raise ValueError("teachers must be chosen from DT/RF/GBDT/CATBOOST/XGBOOST")
        if self.outcome_type not in ("binary", "continuous"):
            raise ValueError("outcome_type must be binary or continuous")
        if self.counterfactual_folds < 2 or not 0 < self.test_size < 1:
            raise ValueError("Invalid counterfactual_folds or test_size")
        if self.cf.target != "predicted":
            raise ValueError(
                "Teacher comparison requires predicted counterfactual outcomes"
            )


def _fold_metrics(train, validation, result, teacher, fold, runtime):
    binary_distance = binary_treatment_distance(validation.treatment, result.treatment)
    binary_proximity = binary_treatment_proximity(
        validation.treatment, result.treatment
    )
    relaxed_distance = binary_treatment_distance(
        validation.treatment, result.relaxed_treatment
    )
    relaxed_proximity = binary_treatment_proximity(
        validation.treatment, result.relaxed_treatment
    )
    relaxed_changes = np.abs(result.relaxed_treatment - validation.treatment)
    relaxed_changed = relaxed_changes > 0
    (
        policy_value,
        policy_change_rate,
        policy_treatment,
        policy_changes,
    ) = policy_proximity(
        validation.treatment,
        result.treatment,
        result.factual_outcome,
        result.outcome,
        return_details=True,
    )
    (
        conditioned_implausibility,
        conditioned_distances,
    ) = validity_conditioned_implausibility(
        validation.X,
        result.treatment,
        result.valid,
        train.X,
        train.treatment,
        return_distances=True,
    )
    implausibility_distances = np.full(len(validation.X), np.nan)
    if relaxed_changed.any():
        implausibility_value, changed_implausibility = implausibility(
            validation.X.iloc[np.flatnonzero(relaxed_changed)],
            result.relaxed_treatment[relaxed_changed],
            train.X,
            train.treatment,
            return_distances=True,
        )
        implausibility_distances[relaxed_changed] = changed_implausibility
    else:
        implausibility_value = np.nan
    return (
        {
            "fold": fold,
            "teacher": teacher,
            "n_explanations": len(validation.X),
            "n_valid": int(result.valid.sum()),
            "validity": float(result.valid.mean()),
            "binary_treatment_distance": binary_distance,
            "binary_treatment_proximity": binary_proximity,
            "n_relaxed_changed": int(relaxed_changed.sum()),
            "relaxed_treatment_change_rate": relaxed_distance,
            "relaxed_categorical_proximity": relaxed_proximity,
            "policy_change_rate": policy_change_rate,
            "policy_proximity": policy_value,
            "validity_conditioned_implausibility": conditioned_implausibility,
            "implausibility": implausibility_value,
            "factual_rmse": float(
                np.sqrt(np.mean((result.factual_outcome - validation.outcome) ** 2))
            ),
            "mean_predicted_cf_outcome": float(result.outcome.mean()),
            "mean_predicted_directional_uplift": float(
                np.where(
                    validation.treatment == 0,
                    result.outcome - result.factual_outcome,
                    result.factual_outcome - result.outcome,
                ).mean()
            ),
            "runtime_seconds": runtime,
        },
        policy_treatment,
        policy_changes,
        conditioned_distances,
        relaxed_changes,
        implausibility_distances,
    )


def _summarize(fold_metrics):
    numeric = [
        "n_valid",
        "n_relaxed_changed",
        "validity",
        "binary_treatment_distance",
        "binary_treatment_proximity",
        "relaxed_treatment_change_rate",
        "relaxed_categorical_proximity",
        "policy_change_rate",
        "policy_proximity",
        "validity_conditioned_implausibility",
        "implausibility",
        "factual_rmse",
        "mean_predicted_cf_outcome",
        "mean_predicted_directional_uplift",
        "runtime_seconds",
    ]
    summary = fold_metrics.groupby("teacher", sort=False)[numeric].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()

    # Pool row-level rates over all OOF observations, rather than giving a
    # slightly smaller last fold the same weight as every other fold.
    grouped = fold_metrics.groupby("teacher", sort=False)
    summary.insert(
        1,
        "n_explanations_total",
        summary.teacher.map(grouped.n_explanations.sum()),
    )
    summary.insert(2, "n_valid_total", summary.teacher.map(grouped.n_valid.sum()))
    summary.insert(
        3,
        "n_relaxed_changed_total",
        summary.teacher.map(grouped.n_relaxed_changed.sum()),
    )
    pooled_by_explanations = [
        "validity",
        "binary_treatment_distance",
        "binary_treatment_proximity",
        "relaxed_treatment_change_rate",
        "relaxed_categorical_proximity",
        "policy_change_rate",
        "policy_proximity",
        "mean_predicted_cf_outcome",
        "mean_predicted_directional_uplift",
    ]
    for row_index, teacher in enumerate(summary.teacher):
        rows = grouped.get_group(teacher)
        weights = rows.n_explanations.to_numpy(dtype=float)
        for metric in pooled_by_explanations:
            summary.loc[row_index, f"{metric}_mean"] = np.average(
                rows[metric].to_numpy(dtype=float), weights=weights
            )
        finite = np.isfinite(rows.validity_conditioned_implausibility)
        valid_weights = rows.n_valid.to_numpy(dtype=float)
        usable = finite & (valid_weights > 0)
        summary.loc[row_index, "validity_conditioned_implausibility_mean"] = (
            np.average(
                rows.loc[usable, "validity_conditioned_implausibility"],
                weights=valid_weights[usable],
            )
            if usable.any()
            else np.nan
        )
        finite = np.isfinite(rows.implausibility)
        changed_weights = rows.n_relaxed_changed.to_numpy(dtype=float)
        usable = finite & (changed_weights > 0)
        summary.loc[row_index, "implausibility_mean"] = (
            np.average(
                rows.loc[usable, "implausibility"],
                weights=changed_weights[usable],
            )
            if usable.any()
            else np.nan
        )
    return summary


def _rank_teachers(summary, teacher_order):
    """Rank binary-treatment teachers using validity and factual OOF fidelity.

    Treatment-distance, policy, and plausibility metrics remain reported diagnostics.
    They are not teacher-selection criteria because structural treatment constraints
    or teacher-specific valid subsets make them unsuitable for a fair ranking.
    """
    directions = {"validity": "max", "factual_rmse": "min"}
    ranking = summary[["teacher"]].copy()
    statuses, used = [], []
    for metric, direction in directions.items():
        values = summary[f"{metric}_mean"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.all():
            statuses.append(
                {
                    "metric": metric,
                    "used_for_selection": False,
                    "reason": "undefined for at least one teacher",
                }
            )
            ranking[f"{metric}_score"] = np.nan
            continue
        span = float(values.max() - values.min())
        if span <= 1e-12:
            statuses.append(
                {
                    "metric": metric,
                    "used_for_selection": False,
                    "reason": "no between-teacher variation",
                }
            )
            ranking[f"{metric}_score"] = np.nan
            continue
        score = (values - values.min()) / span
        if direction == "min":
            score = 1 - score
        ranking[f"{metric}_score"] = score
        statuses.append(
            {"metric": metric, "used_for_selection": True, "reason": "informative"}
        )
        used.append(f"{metric}_score")
    diagnostic_reasons = {
        "binary_treatment_distance": (
            "diagnostic only; the primary forced binary result always changes treatment"
        ),
        "binary_treatment_proximity": (
            "diagnostic only; the primary forced binary result has zero categorical proximity"
        ),
        "relaxed_treatment_change_rate": (
            "sensitivity diagnostic; fraction changed by optimization when the observed treatment is allowed"
        ),
        "relaxed_categorical_proximity": (
            "sensitivity diagnostic; categorical similarity from the relaxed counterfactual result"
        ),
        "policy_change_rate": (
            "diagnostic only; measures policy intervention frequency rather than predictive quality"
        ),
        "policy_proximity": (
            "diagnostic only; measures retention under the outcome-maximizing policy"
        ),
        "validity_conditioned_implausibility": (
            "diagnostic only; each teacher's value is computed on a different valid subset and must be read with validity and n_valid"
        ),
        "implausibility": (
            "sensitivity diagnostic; computed only for treatment-changing relaxed counterfactuals and read with n_relaxed_changed"
        ),
    }
    for metric, reason in diagnostic_reasons.items():
        ranking[f"{metric}_score"] = np.nan
        statuses.append(
            {"metric": metric, "used_for_selection": False, "reason": reason}
        )
    if not used:
        raise ValueError(
            "Validity and factual OOF RMSE contain no information for teacher selection"
        )
    ranking["selection_score"] = ranking[used].mean(axis=1)
    order = {name: index for index, name in enumerate(teacher_order)}
    ranking["configured_order"] = ranking.teacher.map(order)
    ranking = ranking.sort_values(
        ["selection_score", "configured_order"], ascending=[False, True], kind="stable"
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking.drop(columns="configured_order"), pd.DataFrame(statuses), used


def compare_counterfactual_teachers(bundle: DatasetBundle, config: CFComparisonConfig):
    """Compare teachers on development OOF explanations; never inspect the holdout."""
    if bundle.treatment_type != "binary":
        raise ValueError("This comparison currently supports binary treatment only")
    root = Path(config.output_dir)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "counterfactuals").mkdir(exist_ok=True)

    development_idx, test_idx, holdout_strategy = holdout_indices(
        bundle, config.test_size, config.seed, config.shuffle
    )
    development = bundle.subset(development_idx)
    folds = counterfactual_splits(
        development.treatment,
        development.outcome,
        config.counterfactual_folds,
        config.seed,
    )
    manifest = {
        "status": "running",
        "purpose": "development-only counterfactual teacher selection",
        "dataset_metadata": bundle.metadata,
        "development_ids": development.row_ids,
        "untouched_test_ids": bundle.row_ids[test_idx],
        "holdout_stratification": holdout_strategy,
        "counterfactual_folds": config.counterfactual_folds,
        "folds": [],
    }
    progress = ExperimentProgress(
        len(folds) * len(config.teachers), config.show_progress, "Comparing CF teachers"
    )
    metric_rows, explanation_rows = [], []
    try:
        for fold, (fit_idx, validation_idx) in enumerate(folds):
            train = development.subset(fit_idx)
            validation = development.subset(validation_idx)
            cf_config = config.cf
            for teacher in config.teachers:
                started = time.perf_counter()
                learner = LearnerConfig(
                    teacher,
                    config.teacher_params.get(teacher, {}),
                    seed=config.seed + 1000 * fold,
                )
                response = ResponseModel(learner, config.outcome_type).fit(
                    with_treatment(train.X, train.treatment), train.outcome
                )
                optimizer = CounterfactualTreatmentOptimizer(
                    cf_config, seed=config.seed + fold
                )
                result = optimizer.generate(
                    validation.X,
                    validation.treatment,
                    validation.outcome,
                    lambda X, t: response.predict(with_treatment(X, t)),
                    treatment_type="binary",
                    outcome_type=config.outcome_type,
                )
                runtime = time.perf_counter() - started
                (
                    metrics,
                    policy_treatment,
                    policy_changes,
                    conditioned_distances,
                    relaxed_changes,
                    implausibility_distances,
                ) = _fold_metrics(train, validation, result, teacher, fold, runtime)
                metric_rows.append(metrics)
                if config.save_oof_counterfactuals:
                    explanation_rows.append(
                        pd.DataFrame(
                            {
                                "row_id": validation.row_ids,
                                "fold": fold,
                                "teacher": teacher,
                                "treatment_observed": validation.treatment,
                                "outcome_observed": validation.outcome,
                                "outcome_factual_predicted": result.factual_outcome,
                                "treatment_cf": result.treatment,
                                "outcome_cf_predicted": result.outcome,
                                "objective": result.objective,
                                "valid": result.valid,
                                "absolute_distance": np.abs(
                                    result.treatment - validation.treatment
                                ),
                                "binary_treatment_distance": np.abs(
                                    result.treatment - validation.treatment
                                ),
                                "binary_treatment_proximity": 1.0
                                - np.abs(result.treatment - validation.treatment),
                                "policy_treatment": policy_treatment,
                                "policy_change": policy_changes,
                                "policy_proximity": 1.0 - policy_changes,
                                "validity_conditioned_implausibility": conditioned_distances,
                                "relaxed_treatment_cf": result.relaxed_treatment,
                                "relaxed_outcome_cf_predicted": result.relaxed_outcome,
                                "relaxed_objective": result.relaxed_objective,
                                "relaxed_treatment_changed": relaxed_changes > 0,
                                "relaxed_binary_treatment_distance": relaxed_changes,
                                "relaxed_categorical_proximity": 1.0 - relaxed_changes,
                                "implausibility": implausibility_distances,
                            }
                        )
                    )
                progress.advance(teacher, f"fold {fold+1}/{len(folds)}")
            manifest["folds"].append(
                {
                    "fold": fold,
                    "fitting_ids": train.row_ids,
                    "validation_ids": validation.row_ids,
                }
            )

        fold_metrics = pd.DataFrame(metric_rows)
        summary = _summarize(fold_metrics)
        ranking, metric_status, score_columns = _rank_teachers(summary, config.teachers)
        selected = str(ranking.iloc[0].teacher)
        fold_metrics.to_csv(root / "metrics" / "fold_metrics.csv", index=False)
        summary.to_csv(root / "metrics" / "summary.csv", index=False)
        ranking.to_csv(root / "metrics" / "ranking.csv", index=False)
        metric_status.to_csv(root / "metrics" / "metric_status.csv", index=False)
        if explanation_rows:
            pd.concat(explanation_rows, ignore_index=True).to_csv(
                root / "counterfactuals" / "oof.csv", index=False
            )
        selected_payload = {
            "selected_teacher": selected,
            "teacher_params": config.teacher_params.get(selected, {}),
            "selection_score_columns": score_columns,
            "selection_scope": "development OOF only; untouched test not evaluated",
            "selection_note": (
                "Forced distance/proximity, relaxed sensitivity metrics, policy metrics, "
                "and both implausibility variants are diagnostics only; "
                "selection uses informative development-OOF validity and factual RMSE."
            ),
            "uplift_cf_config": {
                "training_strategy": "augment",
                "exclude_original": True,
                "teacher_learner": selected,
                "teacher_params": config.teacher_params.get(selected, {}),
            },
        }
        write_json(root / "selected_teacher.json", selected_payload)
        manifest.update(
            {
                "status": "complete",
                "selected_teacher": selected,
                "selection_score_columns": score_columns,
                "metric_status": metric_status.to_dict("records"),
            }
        )
        return {
            "selected_teacher": selected,
            "fold_metrics": fold_metrics,
            "summary": summary,
            "ranking": ranking,
            "metric_status": metric_status,
            "development_ids": development.row_ids,
            "test_ids": bundle.row_ids[test_idx],
        }
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        progress.close()
