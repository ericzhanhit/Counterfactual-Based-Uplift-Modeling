"""Explicit discounted-price reproduction and genuine continuous extensions."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.base import DatasetBundle
from ..learners.factory import LearnerConfig
from ..learners.continuous import ContinuousSLearner
from ..counterfactual.optimizer import (
    CounterfactualTreatmentOptimizer,
    CounterfactualResult,
)
from ..evaluation.pipeline import (
    describe_counterfactuals,
    counterfactual_training_sample,
    counterfactual_splits,
)
from ..evaluation.cross_validation import (
    run_experiment,
    holdout_indices,
)


class PriceBinarizer:
    """Fit proxy on training only; regular-price comparisons use provided values."""

    def __init__(self, settings, metadata):
        self.settings, self.metadata = settings, metadata
        if settings.get("binarization") not in (
            "regular_price",
            "fixed_threshold",
            "train_median",
        ):
            raise ValueError(
                "Choose binarization: regular_price, fixed_threshold, train_median"
            )

    def __call__(self, train, val):
        method = self.settings["binarization"]
        if method == "train_median":
            threshold = float(np.median(train.treatment))
        elif method == "fixed_threshold":
            threshold = float(self.settings["threshold"])
            if not np.isfinite(threshold):
                raise ValueError("Threshold must be finite")
        else:
            if "regular_prices" not in self.metadata:
                raise ValueError(
                    "regular_price mode requires dataset.regular_price_col"
                )
            threshold = None

        def convert(bundle):
            regular = (
                np.asarray(self.metadata["regular_prices"])[bundle.row_ids.astype(int)]
                if method == "regular_price"
                else threshold
            )
            return DatasetBundle(
                bundle.X,
                (bundle.treatment < regular).astype(int),
                bundle.outcome,
                metadata=bundle.metadata | {"binarization": method},
                row_ids=bundle.row_ids,
            )

        return convert(train), convert(val), {"method": method, "threshold": threshold}


def _continuous_oof_counterfactuals(training, cfg, learner_config):
    """Generate every continuous-treatment training CF out of fold."""
    n = len(training.X)
    arrays = [np.empty(n) for _ in range(4)]
    chosen_t, chosen_y, factual_y, objective = arrays
    valid = np.empty(n, dtype=bool)
    candidates, candidate_outcomes = [None] * n, [None] * n
    relaxed_t, relaxed_y, relaxed_objective = [np.empty(n) for _ in range(3)]
    relaxed_candidates, relaxed_candidate_outcomes = [None] * n, [None] * n
    records = []
    for fold, (fit_idx, held_idx) in enumerate(
        counterfactual_splits(
            training.treatment,
            training.outcome,
            cfg.counterfactual_folds,
            learner_config.seed,
        )
    ):
        model = ContinuousSLearner(learner_config).fit(
            training.X.iloc[fit_idx],
            training.treatment[fit_idx],
            training.outcome[fit_idx],
        )
        result = CounterfactualTreatmentOptimizer(
            cfg.cf, learner_config.seed + fold
        ).generate(
            training.X.iloc[held_idx],
            training.treatment[held_idx],
            training.outcome[held_idx],
            model.predict_response,
            "continuous",
            "continuous",
        )
        chosen_t[held_idx], chosen_y[held_idx] = result.treatment, result.outcome
        factual_y[held_idx], objective[held_idx], valid[held_idx] = (
            result.factual_outcome,
            result.objective,
            result.valid,
        )
        relaxed_t[held_idx], relaxed_y[held_idx], relaxed_objective[held_idx] = (
            result.relaxed_treatment,
            result.relaxed_outcome,
            result.relaxed_objective,
        )
        for local, original in enumerate(held_idx):
            candidates[original] = result.candidates[local]
            candidate_outcomes[original] = result.candidate_outcomes[local]
            relaxed_candidates[original] = result.relaxed_candidates[local]
            relaxed_candidate_outcomes[original] = result.relaxed_candidate_outcomes[
                local
            ]
        records.append(
            {
                "fold": fold,
                "fit_ids": training.row_ids[fit_idx],
                "held_out_ids": training.row_ids[held_idx],
            }
        )
    return (
        CounterfactualResult(
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
        ),
        records,
    )


def run_continuous(bundle, cfg, mode="continuous_reproduction", settings=None):
    settings = settings or {}
    if bundle.treatment_type != "continuous":
        raise ValueError("Continuous experiments require continuous treatments")
    if mode == "continuous_reproduction":
        adapter = PriceBinarizer(settings, bundle.metadata)
        # Split without using an all-data median; the proxy is learned on training only.
        return run_experiment(bundle, cfg, training_adapter=adapter)
    if mode != "continuous_general":
        raise ValueError("Unknown continuous mode")
    if cfg.estimators != ["SL"]:
        raise ValueError(
            "Genuine continuous extension currently implements SL dose-response only; set estimators=['SL']"
        )
    if cfg.outcome_type != "continuous":
        raise ValueError("Continuous extension currently requires continuous outcomes")
    grid = np.asarray(settings.get("dose_grid", []), dtype=float)
    reference = settings.get("reference_dose")
    lo, hi = cfg.cf.lower_bound, cfg.cf.upper_bound
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.isfinite(grid).all()
        or reference is None
        or not np.isfinite(reference)
    ):
        raise ValueError("Provide finite dose_grid (>=2 values) and reference_dose")
    if (
        lo is None
        or hi is None
        or lo >= hi
        or np.any(grid < lo)
        or np.any(grid > hi)
        or not lo <= reference <= hi
    ):
        raise ValueError("Dose grid and reference must lie within explicit CF bounds")
    root = Path(cfg.output_dir)
    for d in ["metrics", "predictions", "counterfactuals", "figures"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    dev, test, _ = holdout_indices(bundle, cfg.test_size, cfg.seed, cfg.shuffle)
    if not len(test):
        raise ValueError("A nonempty untouched test set is required")
    training, evaluation = bundle.subset(dev), bundle.subset(test)
    manifest = {
        "status": "running",
        "mode": mode,
        "test_ids": bundle.row_ids[test],
        "training_ids": training.row_ids,
        "counterfactual_folds": cfg.counterfactual_folds,
        "nuisance_folds": cfg.nuisance_folds,
        "counterfactual_cross_fitting": {},
        "metric_note": "Factual RMSE only; binary AUUC/Qini intentionally unavailable",
    }
    metrics, predictions, cf_metrics, doses = [], [], [], []
    try:
        for base in cfg.learners:
            lc = LearnerConfig(base, cfg.learner_params.get(base, {}), seed=cfg.seed)
            observed = ContinuousSLearner(lc).fit(
                training.X, training.treatment, training.outcome
            )
            cf = None
            if "cf" in cfg.variants:
                teacher_name = cfg.cf.teacher_learner or base
                teacher_params = (
                    cfg.cf.teacher_params
                    if cfg.cf.teacher_learner
                    else cfg.learner_params.get(base, {})
                )
                teacher_cfg = LearnerConfig(teacher_name, teacher_params, seed=lc.seed)
                cf, records = _continuous_oof_counterfactuals(
                    training, cfg, teacher_cfg
                )
                manifest["counterfactual_cross_fitting"][base] = records
                table, cm = describe_counterfactuals(training, cf, cfg, base)
                cf_X, cf_t, cf_y, accepted = counterfactual_training_sample(
                    training.X, training.treatment, training.outcome, cf, cfg.cf
                )
                cf_metrics.append({"split": "training", **cm})
                if cfg.save_counterfactuals:
                    table.to_csv(
                        root / "counterfactuals" / f"training_{base}.csv", index=False
                    )
            for variant in cfg.variants:
                name = f"{base}__{'CF-' if variant == 'cf' else ''}SL-dose"
                model = (
                    observed
                    if variant == "benchmark"
                    else ContinuousSLearner(lc).fit(cf_X, cf_t, cf_y)
                )
                factual = model.predict_response(evaluation.X, evaluation.treatment)
                metrics.append(
                    {
                        "split": "test",
                        "model": name,
                        "factual_rmse": float(
                            np.sqrt(np.mean((factual - evaluation.outcome) ** 2))
                        ),
                    }
                )
                response = np.column_stack(
                    [model.predict_response(evaluation.X, d) for d in grid]
                )
                reference_y = model.predict_response(evaluation.X, reference)
                recommended = grid[response.argmax(axis=1)]
                predictions.append(
                    pd.DataFrame(
                        {
                            "row_id": evaluation.row_ids,
                            "split": "test",
                            "model": name,
                            "treatment": evaluation.treatment,
                            "outcome": evaluation.outcome,
                            "factual_prediction": factual,
                            "recommended_dose": recommended,
                        }
                    )
                )
                for j, d in enumerate(grid):
                    doses.append(
                        pd.DataFrame(
                            {
                                "row_id": evaluation.row_ids,
                                "split": "test",
                                "model": name,
                                "dose": d,
                                "response": response[:, j],
                                "effect_vs_reference": response[:, j] - reference_y,
                            }
                        )
                    )
        metric_df, prediction_df = pd.DataFrame(metrics), pd.concat(
            predictions, ignore_index=True
        )
        metric_df.to_csv(root / "metrics" / "test_metrics.csv", index=False)
        metric_df.set_index("model").drop(columns="split").to_csv(
            root / "metrics" / "summary.csv"
        )
        prediction_df.to_csv(root / "predictions" / "test.csv", index=False)
        dose_df = pd.concat(doses, ignore_index=True)
        dose_df.to_csv(root / "predictions" / "dose_response.csv", index=False)
        if cf_metrics:
            pd.DataFrame(cf_metrics).to_csv(
                root / "metrics" / "counterfactual_metrics.csv", index=False
            )
        if cfg.make_figures:
            from ..visualization.dose_plots import plot_dose_response

            plot_dose_response(dose_df, root / "figures")
        manifest["status"] = "complete"
        return {
            "metrics": metric_df,
            "predictions": prediction_df,
            "manifest": manifest,
        }
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", str(exc)
        raise
