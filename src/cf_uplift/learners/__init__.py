from .s_learner import SLearner
from .t_learner import TLearner
from .x_learner import XLearner
from .dml import DML
from .drl import DRLearner
from .factory import LearnerConfig, get_base_learner

ESTIMATORS = {
    "SL": SLearner,
    "TL": TLearner,
    "XL": XLearner,
    "DML": DML,
    "DRL": DRLearner,
}


def make_estimator(name: str, **kwargs):
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown uplift estimator: {name}")
    return ESTIMATORS[name](**kwargs)
