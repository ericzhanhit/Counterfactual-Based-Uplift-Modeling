"""Transparent per-bin observed outcomes with shared stable descending ranking."""
import numpy as np
import pandas as pd
from .uplift import ranked_observations, cumulative_curves


def ranking_table(y_true, treatment, uplift_score, n_bins: int = 10) -> pd.DataFrame:
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    r = ranked_observations(y_true, treatment, uplift_score)
    c = cumulative_curves(y_true, treatment, uplift_score)
    bins = np.array_split(np.arange(len(r)), min(n_bins, len(r)))
    rows = []
    for b, indices in enumerate(bins, 1):
        group = r.iloc[indices]
        tr, ct = group[group.treatment == 1].y, group[group.treatment == 0].y
        end = c.iloc[indices[-1] + 1]
        rows.append(
            {
                "bin": b,
                "n": len(group),
                "n_treated": len(tr),
                "n_control": len(ct),
                "treated_outcome_total": tr.sum(),
                "control_outcome_total": ct.sum(),
                "treated_outcome_rate": tr.mean(),
                "control_outcome_rate": ct.mean(),
                "observed_uplift": tr.mean() - ct.mean(),
                "mean_predicted_uplift": group.score.mean(),
                "cumulative_treated_outcomes": end.treated_outcomes,
                "cumulative_control_outcomes": end.control_outcomes,
                "cumulative_n_treated": end.n_treated,
                "cumulative_n_control": end.n_control,
                "cumulative_uplift": end.uplift,
                "cumulative_qini": end.qini,
                "cumulative_population_percentage": 100 * end.population_fraction,
            }
        )
    return pd.DataFrame(rows)


def uplift_by_quantile(y_true, treatment, uplift_score, n_bins=10):
    return ranking_table(y_true, treatment, uplift_score, n_bins)


def qini_by_quantile(y_true, treatment, uplift_score, n_bins=10):
    return ranking_table(y_true, treatment, uplift_score, n_bins)
