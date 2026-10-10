"""Separate manuscript one-versus-rest and general native multi-arm experiments."""
from dataclasses import replace
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from ..data.base import DatasetBundle
from ..learners.factory import LearnerConfig, ResponseModel
from ..learners.multi import MultiTreatmentEstimator, categorical_treatment
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
    _export_evaluation,
)

LOG = logging.getLogger(__name__)


def _multi_oof_counterfactuals(training, cfg, learner_config, arms):
    """Generate every multi-treatment training CF out of fold."""
    n = len(training.X)
    chosen_t, chosen_y, factual_y, objective = [np.empty(n) for _ in range(4)]
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
        response = ResponseModel(learner_config, cfg.outcome_type).fit(
            categorical_treatment(
                training.X.iloc[fit_idx], training.treatment[fit_idx]
            ),
            training.outcome[fit_idx],
        )
        result = CounterfactualTreatmentOptimizer(
            cfg.cf, learner_config.seed + fold
        ).generate(
            training.X.iloc[held_idx],
            training.treatment[held_idx],
            training.outcome[held_idx],
            lambda X, t: response.predict(categorical_treatment(X, t)),
            "multi",
            cfg.outcome_type,
            feasible=arms,
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


def run_multi(bundle, cfg, mode="multi_general", settings=None):
    settings = settings or {}
    if bundle.treatment_type != "multi":
        raise ValueError("Multi experiments need a multi DatasetBundle")
    control = settings.get("control", 0)
    arms = np.unique(bundle.treatment)
    if control not in arms or len(arms) < 3:
        raise ValueError("Need control and at least two active categories")
    active = arms[arms != control]
    if mode == "multi_reproduction":
        result = {}
        for arm in active:
            binary = DatasetBundle(
                bundle.X,
                (bundle.treatment == arm).astype(int),
                bundle.outcome,
                metadata=bundle.metadata | {"contrast": f"{arm} versus all other arms"},
                row_ids=bundle.row_ids,
            )
            result[str(arm)] = run_experiment(
                binary,
                replace(cfg, output_dir=str(Path(cfg.output_dir) / f"arm_{arm}")),
            )
        return result
    if mode != "multi_general":
        raise ValueError("Unknown multi mode")
    root = Path(cfg.output_dir)
    for d in [
        "metrics",
        "predictions",
        "ranking_tables",
        "curves",
        "counterfactuals",
        "figures",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)
    dev, test, split_strategy = holdout_indices(
        bundle, cfg.test_size, cfg.seed, cfg.shuffle
    )
    if not len(test):
        raise ValueError("A nonempty untouched test set is required")
    train, val = bundle.subset(dev), bundle.subset(test)
    manifest = {
        "status": "running",
        "mode": mode,
        "control": control,
        "arms": arms,
        "training_ids": train.row_ids,
        "test_ids": bundle.row_ids[test],
        "holdout_stratification": split_strategy,
        "counterfactual_folds": cfg.counterfactual_folds,
        "nuisance_folds": cfg.nuisance_folds,
        "counterfactual_cross_fitting": {},
        "multi_dml_final_model": "joint linear feature interactions; base learner controls nuisance models",
        "category_distance": "numeric L1 codes; category coding matters",
    }
    metric_rows, pred_rows, rec_rows, cf_rows = [], [], [], []
    try:
        LOG.info(
            "Training native multi-treatment models once; evaluating untouched test set"
        )
        if set(np.unique(train.treatment)) != set(arms):
            raise ValueError("Training set missing an arm")
        for base in cfg.learners:
            lc = LearnerConfig(base, cfg.learner_params.get(base, {}), seed=cfg.seed)
            cf = None
            if "cf" in cfg.variants:
                teacher_name = cfg.cf.teacher_learner or base
                teacher_params = (
                    cfg.cf.teacher_params
                    if cfg.cf.teacher_learner
                    else cfg.learner_params.get(base, {})
                )
                teacher_cfg = LearnerConfig(teacher_name, teacher_params, seed=lc.seed)
                cf, records = _multi_oof_counterfactuals(train, cfg, teacher_cfg, arms)
                manifest["counterfactual_cross_fitting"][base] = records
                table, metrics = describe_counterfactuals(train, cf, cfg, base)
                cf_X, cf_t, cf_y, accepted = counterfactual_training_sample(
                    train.X, train.treatment, train.outcome, cf, cfg.cf
                )
                cf_rows.append({"split": "training", **metrics})
                if cfg.save_counterfactuals:
                    table.to_csv(
                        root / "counterfactuals" / f"training_{base}.csv", index=False
                    )
                if set(np.unique(cf.treatment)) != set(arms):
                    raise ValueError(
                        "CF optimization removed an arm; native multi estimation cannot identify all contrasts"
                    )
            for variant in cfg.variants:
                X = train.X if variant == "benchmark" else cf_X
                t = train.treatment if variant == "benchmark" else cf_t
                y = train.outcome if variant == "benchmark" else cf_y
                kind = (
                    cfg.outcome_type
                    if variant == "benchmark" or cfg.cf.target == "observed"
                    else "continuous"
                )
                for name in cfg.estimators:
                    label = f"{base}__{'CF-' if variant == 'cf' else ''}{name}"
                    model = MultiTreatmentEstimator(
                        name, lc, kind, control, cfg.nuisance_folds, cfg.propensity_clip
                    ).fit(X, t, y)
                    uplift = model.predict_uplift(val.X)
                    rec_rows.append(
                        pd.DataFrame(
                            {
                                "row_id": val.row_ids,
                                "split": "test",
                                "model": label,
                                "recommended_treatment": model.recommend(val.X),
                            }
                        )
                    )
                    for j, arm in enumerate(model.active_):
                        keep = np.isin(val.treatment, [control, arm])
                        if not keep.any():
                            continue
                        comparison = DatasetBundle(
                            val.X[keep],
                            (val.treatment[keep] == arm).astype(int),
                            val.outcome[keep],
                            row_ids=val.row_ids[keep],
                        )
                        rows, preds = _export_evaluation(
                            comparison,
                            {f"{label}__arm_{arm}": uplift[keep, j]},
                            "test",
                            cfg,
                            root,
                        )
                        metric_rows.extend(rows)
                        pred_rows.extend(preds)
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(root / "metrics" / "test_metrics.csv", index=False)
        summary_metrics = [
            "auuc",
            "qini_coefficient",
            "auuc_normalized",
            "qini_normalized",
        ]
        summary = metrics.set_index("model")[summary_metrics]
        summary.to_csv(root / "metrics" / "summary.csv")
        predictions = pd.concat(pred_rows, ignore_index=True)
        predictions.to_csv(root / "predictions" / "test.csv", index=False)
        pd.concat(rec_rows).to_csv(
            root / "predictions" / "recommendations.csv", index=False
        )
        if cf_rows:
            pd.DataFrame(cf_rows).to_csv(
                root / "metrics" / "counterfactual_metrics.csv", index=False
            )
        if cfg.make_figures:
            from ..visualization.uplift_plots import plot_predictions

            for arm in active:
                chosen = predictions[predictions.model.str.endswith(f"__arm_{arm}")]
                if not chosen.empty:
                    plot_predictions(chosen, root / "figures" / f"arm_{arm}")
        manifest["status"] = "complete"
        return {"metrics": metrics, "predictions": predictions, "manifest": manifest}
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", str(exc)
        raise
