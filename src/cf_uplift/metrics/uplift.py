"""Observed held-out binary assignment metrics, with explicit manuscript conventions."""
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from ..data.base import vectors


def ranked_observations(y_true, treatment, uplift_score) -> pd.DataFrame:
    score = np.asarray(uplift_score, dtype=float)
    t, y = vectors(treatment, y_true, len(score))
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("Uplift scores must be a finite vector")
    # Stable ties preserve input row order; all curves and deciles share this rule.
    order = np.argsort(-score, kind="stable")
    return pd.DataFrame(
        {"position": order, "y": y[order], "treatment": t[order], "score": score[order]}
    )


def cumulative_curves(y_true, treatment, uplift_score) -> pd.DataFrame:
    r = ranked_observations(y_true, treatment, uplift_score)
    t, y = r.treatment.to_numpy(), r.y.to_numpy()
    n = np.arange(1, len(r) + 1)
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    rt, rc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    ratio = np.divide(nt, nc, out=np.full(len(r), np.nan), where=nc != 0)
    qini = rt - rc * ratio
    curves = pd.DataFrame(
        {
            "n": n,
            "population_fraction": n / len(r),
            "n_treated": nt,
            "n_control": nc,
            "treated_outcomes": rt,
            "control_outcomes": rc,
            "uplift": rt - rc,
            "qini": qini,
        }
    )
    return pd.concat(
        [pd.DataFrame([{c: 0.0 for c in curves.columns}]), curves], ignore_index=True
    )


def uplift_curve(y_true, treatment, uplift_score):
    c = cumulative_curves(y_true, treatment, uplift_score)
    return c.population_fraction.to_numpy(), c.uplift.to_numpy()


def qini_curve(y_true, treatment, uplift_score):
    c = cumulative_curves(y_true, treatment, uplift_score)
    return c.population_fraction.to_numpy(), c.qini.to_numpy()


def _qini_for_area(c, undefined_prefix: str):
    values = c.qini.to_numpy().copy()
    if c.n_treated.iloc[-1] == 0 or c.n_control.iloc[-1] == 0:
        return values * np.nan
    if undefined_prefix == "interpolate":
        valid = np.isfinite(values)
        values = np.interp(
            c.population_fraction, c.population_fraction[valid], values[valid]
        )
    elif undefined_prefix != "nan":
        raise ValueError("undefined_prefix must be interpolate or nan")
    return values


def auuc(y_true, treatment, uplift_score) -> float:
    """Trapezoid area of Eq. (15) on fraction targeted, no terminal normalization."""
    x, curve = uplift_curve(y_true, treatment, uplift_score)
    return float(trapezoid(curve, x))


def qini_coefficient(
    y_true, treatment, uplift_score, undefined_prefix: str = "interpolate"
) -> float:
    """Area above endpoint random line; undefined prefix interpolated explicitly.

    Raw curves retain NaNs where Nc=0. An absent arm in the entire sample yields
    NaN, not interpolation. This coherent area convention differs from literal
    Eq. (18)'s mean-response baseline; both are exported by evaluate_uplift.
    """
    c = cumulative_curves(y_true, treatment, uplift_score)
    q = _qini_for_area(c, undefined_prefix)
    return float(trapezoid(q, c.population_fraction) - q[-1] / 2)


def evaluate_uplift(
    y_true, treatment, uplift_score, undefined_prefix="interpolate"
) -> dict:
    c = cumulative_curves(y_true, treatment, uplift_score)
    q = _qini_for_area(c, undefined_prefix)
    last = c.iloc[-1]
    raw_auuc = float(trapezoid(c.uplift, c.population_fraction))
    raw_qini = float(trapezoid(q, c.population_fraction) - q[-1] / 2)
    terminal_uplift = float(last.uplift)
    # Endpoint normalization is useful for scale-free comparisons, but is
    # undefined (and numerically misleading) when the overall uplift is zero.
    normalized_auuc = (
        raw_auuc / terminal_uplift
        if np.isfinite(terminal_uplift) and terminal_uplift != 0
        else np.nan
    )
    normalized_qini = (
        raw_qini / terminal_uplift
        if np.isfinite(terminal_uplift) and terminal_uplift != 0
        else np.nan
    )
    return {
        "auuc": raw_auuc,
        "qini_coefficient": raw_qini,
        "auuc_normalized": normalized_auuc,
        "qini_normalized": normalized_qini,
    }
