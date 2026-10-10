"""Validated datasets. Missing covariates are imputed inside training models."""
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import pandas as pd

TreatmentType = Literal["binary", "multi", "continuous"]


def frame(X) -> pd.DataFrame:
    """Copy features while preserving row order, never aligning labels implicitly."""
    result = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if result.ndim != 2 or result.shape[1] == 0 or not result.columns.is_unique:
        raise ValueError("X must have unique columns and at least one feature")
    result.columns = result.columns.astype(str)
    if not result.columns.is_unique or "__treatment__" in result:
        raise ValueError("Feature names must be unique; __treatment__ is reserved")
    return result.reset_index(drop=True)


def vectors(treatment, y, n: int, binary: bool = True):
    t, y = np.asarray(treatment), np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != n or len(y) != n or n == 0:
        raise ValueError("X, treatment and outcome must have equal nonzero lengths")
    if pd.isna(t).any() or not np.isfinite(y).all():
        raise ValueError("Treatment/outcome cannot be missing or infinite")
    if binary and not np.isin(t, [0, 1]).all():
        raise ValueError("Binary estimator/metric requires treatment encoded as 0/1")
    return t.copy(), y.copy()


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    treatment: np.ndarray
    outcome: np.ndarray
    treatment_type: TreatmentType = "binary"
    metadata: dict[str, Any] = field(default_factory=dict)
    row_ids: np.ndarray | None = None

    def __post_init__(self):
        self.X = frame(self.X)
        if self.treatment_type not in ("binary", "multi", "continuous"):
            raise ValueError("Unknown treatment_type")
        self.treatment, self.outcome = vectors(
            self.treatment, self.outcome, len(self.X), self.treatment_type == "binary"
        )
        if (
            self.treatment_type == "continuous"
            and not np.isfinite(self.treatment.astype(float)).all()
        ):
            raise ValueError("Continuous treatments must be finite")
        if self.row_ids is None:
            self.row_ids = np.arange(len(self.X))
        self.row_ids = np.asarray(self.row_ids)
        if len(self.row_ids) != len(self.X) or len(np.unique(self.row_ids)) != len(
            self.X
        ):
            raise ValueError("row_ids must be unique and match rows")

    @property
    def feature_names(self) -> list[str]:
        return self.X.columns.tolist()

    def subset(self, indices) -> "DatasetBundle":
        return DatasetBundle(
            self.X.iloc[indices],
            self.treatment[indices],
            self.outcome[indices],
            self.treatment_type,
            self.metadata.copy(),
            self.row_ids[indices],
        )
