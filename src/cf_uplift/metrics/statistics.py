"""Paired manuscript comparisons, kept separate from Algorithm 1 fitting."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite vector in [0,1]")
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1)
    return out


def paired_model_comparisons(
    results,
    metrics=("auuc_normalized", "qini_normalized"),
):
    """Two-sided paired Wilcoxon tests, BH-FDR, and rank-biserial effects.

    The default report metrics are terminal-uplift normalized. Pairs containing
    an undefined normalized value are excluded independently for each metric.
    Callers may still pass explicit metric names for sensitivity analyses.
    """
    df = pd.DataFrame(results).copy()
    run_col = "run" if "run" in df else "fold"
    required = {"dataset", "base_learner", "estimator", "variant", run_col, *metrics}
    missing = required - set(df)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    rows = []
    keys = ["dataset", "base_learner", "estimator"]
    for key, group in df.groupby(keys, dropna=False):
        wide = group.pivot(index=run_col, columns="variant", values=list(metrics))
        for metric in metrics:
            if (metric, "benchmark") not in wide or (metric, "cf") not in wide:
                continue
            pair = wide[[(metric, "benchmark"), (metric, "cf")]].dropna()
            diff = pair[(metric, "cf")] - pair[(metric, "benchmark")]
            if len(diff) == 0:
                continue
            if np.allclose(diff, 0):
                stat, p, effect = 0.0, 1.0, 0.0
            else:
                test = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                stat, p = float(test.statistic), float(test.pvalue)
                ranks = pd.Series(np.abs(diff)).rank(method="average").to_numpy()
                effect = float(
                    (
                        ranks[diff.to_numpy() > 0].sum()
                        - ranks[diff.to_numpy() < 0].sum()
                    )
                    / ranks.sum()
                )
            rows.append(
                dict(
                    zip(keys, key),
                    metric=metric,
                    n_pairs=len(diff),
                    mean_difference=float(diff.mean()),
                    median_difference=float(diff.median()),
                    wilcoxon_statistic=stat,
                    p_value=p,
                    rank_biserial=effect,
                )
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_value_bh"] = benjamini_hochberg(out.p_value)
    return out
