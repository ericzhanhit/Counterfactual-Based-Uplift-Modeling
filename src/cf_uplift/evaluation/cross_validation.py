"""Single 70/30 train/test experiment with training-only cross-fitting."""
from pathlib import Path
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .pipeline import ExperimentConfig, FittedFold
from ..data.base import DatasetBundle
from ..metrics.uplift import evaluate_uplift, cumulative_curves
from ..metrics.ranking import ranking_table
from .progress import ExperimentProgress

LOG = logging.getLogger(__name__)


def write_json(path: Path, payload):
    """Write JSON for explicit result artifacts (not experiment metadata)."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def choose_strata(t, y, minimum: int):
    """Prefer joint T/Y for binary outcomes, then T, then unstratified splitting."""
    options = []
    if np.isin(y, [0, 1]).all():
        options.append(
            (np.array([f"{a}_{b}" for a, b in zip(t, y)]), "treatment_outcome")
        )
    options.append((np.asarray(t), "treatment"))
    for strata, name in options:
        if np.unique(strata, return_counts=True)[1].min() >= minimum:
            return strata, name
    return None, "unstratified"


def holdout_indices(bundle, test_size=0.3, seed=42, shuffle=True):
    indices = np.arange(len(bundle.X))
    if test_size == 0:
        return indices, np.array([], dtype=int), "none"
    if not shuffle:
        tr, te = train_test_split(indices, test_size=test_size, shuffle=False)
        return tr, te, "ordered"
    strata, strategy = choose_strata(bundle.treatment, bundle.outcome, 2)
    # Small holdouts may not have room for every joint stratum.
    if strata is not None and min(
        int(np.ceil(len(indices) * test_size)), int(len(indices) * (1 - test_size))
    ) < len(np.unique(strata)):
        strata, strategy = None, "unstratified"
    tr, te = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=strata
    )
    return tr, te, strategy


def _export_evaluation(bundle, scores, split, cfg, root, runtime_seconds=np.nan):
    metrics, predictions = [], []
    for model, prediction in scores.items():
        if (
            np.shape(prediction) != np.shape(bundle.outcome)
            or not np.isfinite(prediction).all()
        ):
            raise ValueError(f"Invalid predictions: {model}")
        metrics.append(
            {
                "split": split,
                "model": model,
                "runtime_seconds": runtime_seconds,
                **evaluate_uplift(
                    bundle.outcome, bundle.treatment, prediction, cfg.undefined_prefix
                ),
            }
        )
        pred = pd.DataFrame(
            {
                "row_id": bundle.row_ids,
                "split": split,
                "model": model,
                "treatment": bundle.treatment,
                "outcome": bundle.outcome,
                "uplift": prediction,
            }
        )
        predictions.append(pred)
        stem = f"{split}_{model}"
        ranking_table(bundle.outcome, bundle.treatment, prediction, cfg.n_bins).to_csv(
            root / "ranking_tables" / f"{stem}.csv", index=False
        )
        cumulative_curves(bundle.outcome, bundle.treatment, prediction).to_csv(
            root / "curves" / f"{stem}.csv", index=False
        )
    return metrics, predictions


def _export_cf(fitted, cfg, root):
    if cfg.save_counterfactuals and fitted.cf_tables_:
        pd.concat(fitted.cf_tables_).to_csv(
            root / "counterfactuals" / "training.csv", index=False
        )
    if fitted.candidate_tables_:
        pd.concat(fitted.candidate_tables_).to_csv(
            root / "counterfactuals" / "training_candidates.csv", index=False
        )
    return [{"split": "training", **m} for m in fitted.cf_metrics_]


def run_experiment(
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    *,
    training_indices=None,
    test_indices=None,
    training_adapter=None,
) -> dict:
    """Fit once on the training partition and evaluate the untouched test set.

    The optional adapter fits a treatment proxy on training only and returns
    training/test binary bundles plus metadata. It is used solely for explicitly
    configured continuous reproduction modes.
    """
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for directory in [
        "metrics",
        "predictions",
        "ranking_tables",
        "curves",
        "counterfactuals",
        "figures",
    ]:
        (root / directory).mkdir(exist_ok=True)
    if training_indices is None:
        dev_idx, test_idx, holdout_strategy = holdout_indices(
            bundle, cfg.test_size, cfg.seed, cfg.shuffle
        )
    else:
        dev_idx, test_idx, holdout_strategy = (
            np.asarray(training_indices),
            np.asarray(test_indices),
            "provided",
        )
    if set(dev_idx) & set(test_idx) or len(set(dev_idx) | set(test_idx)) != len(
        bundle.X
    ):
        raise ValueError("Training and test indices must partition the dataset")
    if not len(test_idx):
        raise ValueError(
            "A nonempty untouched test set is required; set test_size between 0 and 1"
        )
    training = bundle.subset(dev_idx)
    manifest = {
        "status": "running",
        "dataset_metadata": bundle.metadata,
        "feature_names": bundle.feature_names,
        "training_ids": training.row_ids,
        "test_ids": bundle.row_ids[test_idx],
        "holdout_stratification": holdout_strategy,
        "counterfactual_folds": cfg.counterfactual_folds,
        "nuisance_folds": cfg.nuisance_folds,
        "metric_convention": "fraction-trapezoid AUUC/Qini with terminal-uplift normalization",
    }
    all_metrics, all_predictions, cf_metrics = [], [], []
    fits_per_job = len(cfg.learners) * len(cfg.estimators) * len(cfg.variants)
    progress = ExperimentProgress(
        fits_per_job, cfg.show_progress, cfg.progress_description
    )
    try:
        test = bundle.subset(test_idx)
        adapter_info = {}
        if training_adapter:
            training, test, adapter_info = training_adapter(training, test)
        LOG.info(
            "Training once on %d rows; evaluating untouched test set (%d rows)",
            len(training.X),
            len(test.X),
        )
        started = time.perf_counter()
        fitted = FittedFold(
            cfg,
            cfg.seed,
            progress_callback=lambda model: progress.advance(model, "training fit"),
        ).fit(training)
        runtime = time.perf_counter() - started
        metrics, predictions = _export_evaluation(
            test, fitted.predict(test.X), "test", cfg, root, runtime
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        cf_metrics.extend(_export_cf(fitted, cfg, root))
        manifest["training_adapter"] = adapter_info
        manifest["counterfactual_cross_fitting"] = getattr(
            fitted, "cf_generation_splits_", {}
        )
        metric_df = pd.DataFrame(all_metrics)
        metric_df.to_csv(root / "metrics" / "test_metrics.csv", index=False)
        aggregate = metric_df.set_index("model").drop(columns=["split"])
        aggregate.to_csv(root / "metrics" / "summary.csv")
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions.to_csv(root / "predictions" / "test.csv", index=False)
        if cf_metrics:
            cf_df = pd.DataFrame(cf_metrics)
            cf_df.to_csv(root / "metrics" / "counterfactual_metrics.csv", index=False)
            cf_df.set_index("learner").drop(columns=["split"]).to_csv(
                root / "metrics" / "counterfactual_summary.csv"
            )
        if cfg.make_figures:
            from ..visualization.uplift_plots import plot_predictions

            plot_predictions(predictions, root / "figures")
        manifest["status"] = "complete"
        return {
            "metrics": metric_df,
            "summary": aggregate,
            "predictions": predictions,
            "manifest": manifest,
        }
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        progress.close()
