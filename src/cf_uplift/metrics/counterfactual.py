"""Counterfactual-generation metrics.

Missing/undefined values propagate, never masquerade as zero.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def validity(
    counterfactual_outcomes,
    factual_outcomes,
    observed_treatments,
    counterfactual_treatments,
    *,
    treatment_type="binary",
) -> float:
    """Binary positive-uplift validity for treatment-changing counterfactuals.

    A 0→1 transition is valid when ``ŷ_cf > ŷ_factual``. A 1→0 transition
    is valid when ``ŷ_cf <= ŷ_factual``. Unchanged treatments are invalid.
    Non-binary extensions retain strict counterfactual-improvement validity.
    """
    counterfactual = np.asarray(counterfactual_outcomes, dtype=float)
    factual = np.asarray(factual_outcomes, dtype=float)
    observed_t = np.asarray(observed_treatments, dtype=float)
    counterfactual_t = np.asarray(counterfactual_treatments, dtype=float)
    if (
        counterfactual.size == 0
        or not (
            counterfactual.shape
            == factual.shape
            == observed_t.shape
            == counterfactual_t.shape
        )
        or not np.isfinite(counterfactual).all()
        or not np.isfinite(factual).all()
        or not np.isfinite(observed_t).all()
        or not np.isfinite(counterfactual_t).all()
    ):
        return float("nan")
    if treatment_type != "binary":
        return float(np.mean(counterfactual > factual))
    if (
        not np.isin(observed_t, [0, 1]).all()
        or not np.isin(counterfactual_t, [0, 1]).all()
    ):
        return float("nan")
    control_to_treatment = (
        (observed_t == 0) & (counterfactual_t == 1) & (counterfactual > factual)
    )
    treatment_to_control = (
        (observed_t == 1) & (counterfactual_t == 0) & (counterfactual <= factual)
    )
    return float(np.mean(control_to_treatment | treatment_to_control))


def binary_treatment_distance(original_treatments, counterfactual_treatments) -> float:
    """Mean paired Hamming distance for binary treatments; lower is closer.

    This is the fraction of observations whose counterfactual treatment differs
    from the factual treatment. Unlike MAD-normalized proximity, it is always
    defined for nonempty, finite binary inputs. It is a diagnostic rather than a
    teacher-selection criterion when the optimizer requires treatment changes.
    """
    factual = np.asarray(original_treatments, dtype=float)
    counterfactual = np.asarray(counterfactual_treatments, dtype=float)
    if factual.size == 0 or factual.shape != counterfactual.shape:
        return float("nan")
    if (
        not np.isfinite(factual).all()
        or not np.isfinite(counterfactual).all()
        or not np.isin(factual, [0, 1]).all()
        or not np.isin(counterfactual, [0, 1]).all()
    ):
        return float("nan")
    return float(np.mean(np.abs(counterfactual - factual)))


def binary_treatment_proximity(original_treatments, counterfactual_treatments) -> float:
    """Binary treatment similarity, one minus binary treatment distance."""
    distance = binary_treatment_distance(original_treatments, counterfactual_treatments)
    return 1.0 - distance if np.isfinite(distance) else float("nan")


def policy_proximity(
    observed_treatments,
    counterfactual_treatments,
    factual_outcomes,
    counterfactual_outcomes,
    *,
    return_details=False,
):
    """Proximity of the outcome-maximizing binary policy to observed treatment.

    The policy recommends the counterfactual treatment only when its predicted
    outcome is strictly greater than the factual prediction; ties retain the
    observed treatment. Policy proximity is one minus the policy change rate.
    """
    observed = np.asarray(observed_treatments, dtype=float)
    counterfactual_t = np.asarray(counterfactual_treatments, dtype=float)
    factual_y = np.asarray(factual_outcomes, dtype=float)
    counterfactual_y = np.asarray(counterfactual_outcomes, dtype=float)
    if (
        observed.size == 0
        or not (
            observed.shape
            == counterfactual_t.shape
            == factual_y.shape
            == counterfactual_y.shape
        )
        or not np.isfinite(observed).all()
        or not np.isfinite(counterfactual_t).all()
        or not np.isfinite(factual_y).all()
        or not np.isfinite(counterfactual_y).all()
        or not np.isin(observed, [0, 1]).all()
        or not np.isin(counterfactual_t, [0, 1]).all()
    ):
        if return_details:
            return (
                float("nan"),
                float("nan"),
                np.full(observed.shape, np.nan),
                np.full(observed.shape, np.nan),
            )
        return float("nan")
    recommended = np.where(counterfactual_y > factual_y, counterfactual_t, observed)
    changes = np.abs(recommended - observed)
    change_rate = float(np.mean(changes))
    value = 1.0 - change_rate
    return (value, change_rate, recommended, changes) if return_details else value


def validity_conditioned_implausibility(
    generated_X,
    counterfactual_treatments,
    valid_mask,
    reference_X,
    reference_treatments,
    *,
    return_distances=False,
):
    """Mean treatment-matched implausibility among valid counterfactuals.

    For each generated ``(x_i, t_i_cf)``, this implements
    ``min_{j: t_j=t_i_cf} (1/p) sum_l |x_il-x_jl|/MAD_l``.  MADs are learned
    only from the observed reference population. Numeric missing values are
    filled with reference medians and categorical columns are deterministically
    integer-coded before distance calculation. Constant (zero-MAD) columns
    carry no distance information and are omitted from both the sum and ``p``.
    The metric is undefined when no informative column or no treatment-matched
    reference observation exists. Invalid rows are returned as NaN and excluded
    from the aggregate, whose denominator is the number of valid rows.
    """
    generated = pd.DataFrame(generated_X).reset_index(drop=True).copy()
    reference = pd.DataFrame(reference_X).reset_index(drop=True).copy()
    if list(generated.columns) != list(reference.columns):
        raise ValueError("Generated and reference features must have identical columns")
    cf_t = np.asarray(counterfactual_treatments)
    valid = np.asarray(valid_mask, dtype=bool)
    ref_t = np.asarray(reference_treatments)
    if (
        len(generated) != len(cf_t)
        or len(generated) != len(valid)
        or len(reference) != len(ref_t)
        or len(generated) == 0
    ):
        distances = np.full(len(generated), np.nan)
        return (float("nan"), distances) if return_distances else float("nan")

    generated_values, reference_values = [], []
    for column in reference.columns:
        ref_column, gen_column = reference[column], generated[column]
        if pd.api.types.is_numeric_dtype(ref_column):
            ref_numeric = pd.to_numeric(ref_column, errors="coerce").to_numpy(
                float, copy=True
            )
            gen_numeric = pd.to_numeric(gen_column, errors="coerce").to_numpy(
                float, copy=True
            )
            finite = np.isfinite(ref_numeric)
            fill = float(np.median(ref_numeric[finite])) if finite.any() else 0.0
            ref_numeric[~finite] = fill
            gen_numeric[~np.isfinite(gen_numeric)] = fill
        else:
            ref_text = ref_column.astype("string").fillna("<missing>")
            gen_text = gen_column.astype("string").fillna("<missing>")
            levels = pd.Index(
                pd.unique(pd.concat([ref_text, gen_text], ignore_index=True))
            )
            mapping = {value: index for index, value in enumerate(levels)}
            ref_numeric = ref_text.map(mapping).to_numpy(float)
            gen_numeric = gen_text.map(mapping).to_numpy(float)
        reference_values.append(ref_numeric)
        generated_values.append(gen_numeric)
    ref_array = np.column_stack(reference_values)
    gen_array = np.column_stack(generated_values)
    medians = np.median(ref_array, axis=0)
    mad = np.median(np.abs(ref_array - medians), axis=0)
    informative = np.isfinite(mad) & (mad > 0)
    distances = np.full(len(generated), np.nan)
    if informative.any():
        scaled_reference = ref_array[:, informative] / mad[informative]
        scaled_generated = gen_array[:, informative] / mad[informative]
        for treatment in np.unique(cf_t):
            generated_rows = np.flatnonzero(cf_t == treatment)
            reference_rows = np.flatnonzero(ref_t == treatment)
            if not len(reference_rows):
                continue
            # Exact 1-NN avoids materializing the full generated-by-reference
            # distance matrix. sklearn selects an appropriate search backend.
            nearest = NearestNeighbors(
                n_neighbors=1, metric="manhattan", algorithm="brute", n_jobs=1
            ).fit(scaled_reference[reference_rows])
            treatment_distances = nearest.kneighbors(
                scaled_generated[generated_rows], return_distance=True
            )[0][:, 0]
            distances[generated_rows] = treatment_distances / informative.sum()
    conditioned_distances = np.where(valid, distances, np.nan)
    value = (
        float(np.mean(distances[valid]))
        if valid.any() and np.isfinite(distances[valid]).all()
        else float("nan")
    )
    return (value, conditioned_distances) if return_distances else value


def implausibility(
    generated_X,
    counterfactual_treatments,
    reference_X,
    reference_treatments,
    *,
    return_distances=False,
):
    """Original treatment-matched implausibility over all supplied CF rows."""
    n_rows = len(pd.DataFrame(generated_X))
    return validity_conditioned_implausibility(
        generated_X,
        counterfactual_treatments,
        np.ones(n_rows, dtype=bool),
        reference_X,
        reference_treatments,
        return_distances=return_distances,
    )
