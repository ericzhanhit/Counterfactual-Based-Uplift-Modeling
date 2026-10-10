"""Local dataset adapters; never download data or learn preprocessing globally."""
from pathlib import Path
import numpy as np
import pandas as pd
from .base import DatasetBundle
from .synthetic import generate_synthetic_data


def load_csv(
    path,
    treatment_col: str,
    outcome_col: str,
    treatment_type="binary",
    feature_columns=None,
    drop_columns=None,
    treatment_mapping=None,
    dataset="custom",
):
    df = pd.read_csv(path)
    required = [treatment_col, outcome_col]
    if not set(required) <= set(df):
        raise ValueError(f"Dataset must contain {required}")
    t = df[treatment_col]
    if treatment_mapping:
        t = t.map(treatment_mapping)
    features = feature_columns or [
        c for c in df if c not in required + (drop_columns or [])
    ]
    if treatment_col in features or outcome_col in features:
        raise ValueError("Features cannot include treatment or outcome columns")
    return DatasetBundle(
        df[features],
        t.to_numpy(),
        df[outcome_col].to_numpy(),
        treatment_type,
        {"dataset": dataset, "source_file": Path(path).name},
    )


def load_lenta(path, **kwargs):
    # 195 source columns minus treatment/outcome = the manuscript's 193 X columns.
    # Gender remains categorical and is encoded inside each fitting fold.
    return load_csv(
        path,
        treatment_col="group",
        outcome_col="response_att",
        treatment_mapping={"test": 1, "control": 0},
        dataset="lenta",
        **kwargs,
    )


def load_megafon(path, **kwargs):
    return load_csv(
        path,
        treatment_col="treatment_group",
        outcome_col="conversion",
        drop_columns=["id"],
        treatment_mapping={"treatment": 1, "control": 0},
        dataset="megafon",
        **kwargs,
    )


def load_hillstrom(path, outcome_col="visit"):
    return load_csv(
        path,
        treatment_col="segment",
        outcome_col=outcome_col,
        treatment_type="multi",
        treatment_mapping={"No E-Mail": 0, "Womens E-Mail": 1, "Mens E-Mail": 2},
        feature_columns=[
            "recency",
            "history_segment",
            "history",
            "mens",
            "womens",
            "zip_code",
            "newbie",
            "channel",
        ],
        dataset="hillstrom",
    )


def load_dominicks(path, regular_price_col=None):
    bundle = load_csv(
        path,
        "price",
        "logmove",
        "continuous",
        drop_columns=[regular_price_col] if regular_price_col else [],
        dataset="dominicks",
    )
    if regular_price_col:
        values = pd.read_csv(path, usecols=[regular_price_col])[
            regular_price_col
        ].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Regular prices must be finite")
        bundle.metadata["regular_prices"] = values.tolist()
    return bundle


def load_x5(clients_path, train_path, purchases_path, cutoff=None):
    """Join raw X5 tables and reproduce seven row-local historical features.

    Input train table must include treatment_flg and target. Supply purchases
    strictly before intervention, or cutoff with transaction_datetime available.
    Unlike the old complete-case deletion, missing values remain for fold imputation.
    """
    clients, train, purchases = (
        pd.read_csv(p) for p in [clients_path, train_path, purchases_path]
    )
    if clients.client_id.duplicated().any() or train.client_id.duplicated().any():
        raise ValueError("clients and train must contain unique client_id")
    if cutoff is not None:
        purchases = purchases[
            pd.to_datetime(purchases.transaction_datetime) < pd.Timestamp(cutoff)
        ]
    counts = purchases.groupby("client_id")[["transaction_id", "product_id"]].nunique()
    merged = train.merge(
        clients, on="client_id", how="left", validate="one_to_one"
    ).join(counts, on="client_id")
    for prefix in ["first_issue", "first_redeem"]:
        dt = pd.to_datetime(merged[prefix + "_date"], errors="coerce")
        if cutoff is not None:
            dt = dt.where(dt < pd.Timestamp(cutoff))
        merged[prefix + "_time"] = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds()
    merged["issue_redeem_delay"] = merged.first_redeem_time - merged.first_issue_time
    columns = [
        "age",
        "gender",
        "first_issue_time",
        "first_redeem_time",
        "issue_redeem_delay",
        "transaction_id",
        "product_id",
    ]
    return DatasetBundle(
        merged[columns],
        merged.treatment_flg.to_numpy(),
        merged.target.to_numpy(),
        metadata={
            "dataset": "x5",
            "purchase_cutoff": cutoff,
            "requires_pre_treatment_provenance": cutoff is None,
        },
    )


def load_dataset(spec: dict, seed=42):
    spec = spec.copy()
    name = spec.pop("name", "synthetic")
    if name == "synthetic":
        return generate_synthetic_data(seed=seed, **spec)
    loaders = {
        "csv": load_csv,
        "lenta": load_lenta,
        "megafon": load_megafon,
        "hillstrom": load_hillstrom,
        "dominicks": load_dominicks,
        "x5": load_x5,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset {name}")
    return loaders[name](**spec)
