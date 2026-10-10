"""Treatment-only Eq. (5) random search with explicit Algorithm 1 conventions."""
from dataclasses import dataclass, field
from collections.abc import Callable
import numpy as np
from ..data.base import frame


@dataclass
class CounterfactualConfig:
    method: str = "manuscript_random_search"
    n_candidates: int = 100
    lambda_gain: float = 1.0
    # The optimization method generates forced and relaxed results, and this
    # flag selects the Stage-2 view. The random-perturbation ablation samples
    # from the relaxed feasible set so binary assignment remains genuinely
    # random rather than collapsing to a deterministic treatment flip.
    exclude_original: bool = True
    target: str = "predicted"
    lower_bound: float | None = None
    upper_bound: float | None = None
    resolution: float | None = None
    # Stage-2 construction. `replace` reproduces the manuscript. `selective`
    # retains factual rows unless the CF objective clears min_objective.
    # `augment` keeps factual rows and appends accepted CF rows.
    training_strategy: str = "augment"
    # None implements A_i = 1 for every generated counterfactual. A numeric
    # value implements A_i = 1[J_i > min_objective].
    min_objective: float | None = None
    # Optional directional binary-uplift rule:
    # A_i = 1[mu_hat_i(1) - mu_hat_i(0) > min_uplift].
    min_uplift: float | None = None
    # None means a matched teacher: use the same learner family and parameters
    # as the Stage-2 base learner. A name selects a common independent teacher.
    teacher_learner: str | None = None
    teacher_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.training_strategy not in ("replace", "selective", "augment"):
            raise ValueError("training_strategy must be replace, selective, or augment")
        if (
            self.training_strategy in ("selective", "augment")
            and self.target != "predicted"
        ):
            raise ValueError(
                "Improved CF strategies require predicted counterfactual outcomes"
            )
        if self.min_objective is not None and not np.isfinite(self.min_objective):
            raise ValueError("min_objective must be finite or None")
        if self.min_uplift is not None and not np.isfinite(self.min_uplift):
            raise ValueError("min_uplift must be finite or None")
        if self.min_objective is not None and self.min_uplift is not None:
            raise ValueError("Choose either min_objective or min_uplift, not both")

    def acceptance_mask(
        self,
        objective,
        treatment=None,
        factual_outcome=None,
        counterfactual_outcome=None,
    ) -> np.ndarray:
        """Return A_i; by default every generated counterfactual is accepted."""
        objective = np.asarray(objective, dtype=float)
        if not np.isfinite(objective).all():
            raise ValueError("Counterfactual objectives must be finite")
        if self.min_uplift is not None:
            t = np.asarray(treatment, dtype=float)
            factual = np.asarray(factual_outcome, dtype=float)
            counterfactual = np.asarray(counterfactual_outcome, dtype=float)
            if not (
                t.shape == factual.shape == counterfactual.shape == objective.shape
            ):
                raise ValueError(
                    "Directional uplift acceptance inputs must have equal shapes"
                )
            if not np.isin(t, [0, 1]).all():
                raise ValueError(
                    "Directional uplift acceptance requires binary treatment"
                )
            if not np.isfinite(factual).all() or not np.isfinite(counterfactual).all():
                raise ValueError("Directional uplift predictions must be finite")
            uplift = np.where(
                t == 0, counterfactual - factual, factual - counterfactual
            )
            return uplift > self.min_uplift
        if self.min_objective is None:
            return np.ones(objective.shape, dtype=bool)
        return objective > self.min_objective


@dataclass
class CounterfactualResult:
    """Primary result plus a relaxed sensitivity result.

    For manuscript optimization, primary fields exclude observed treatment and
    ``relaxed_*`` fields include it. For random-perturbation ablation, both views
    contain the same outcome-independent random assignment. Primary fields are
    used for validity and, by default, Stage-2 fitting.
    """

    treatment: np.ndarray
    outcome: np.ndarray
    factual_outcome: np.ndarray
    objective: np.ndarray
    valid: np.ndarray
    candidates: list[np.ndarray]
    candidate_outcomes: list[np.ndarray]
    relaxed_treatment: np.ndarray | None = None
    relaxed_outcome: np.ndarray | None = None
    relaxed_objective: np.ndarray | None = None
    relaxed_candidates: list[np.ndarray] | None = None
    relaxed_candidate_outcomes: list[np.ndarray] | None = None


class CounterfactualTreatmentOptimizer:
    """X is immutable; predict_outcome(X, T) must be trained on this training fold.

    Discrete Gaussian proposals are projected to nearest numeric feasible category,
    deduplicated, and supplemented by enumerating the finite feasible set (Algorithm
    1). Binary therefore has one candidate when original treatment is excluded.
    Continuous proposals are T + N(0,1), clipped to bounds and optionally quantized.
    No desired-range filter is added to Algorithm 1's unconstrained objective.
    """

    def __init__(self, config: CounterfactualConfig | None = None, seed: int = 42):
        self.config, self.seed = config or CounterfactualConfig(), seed
        c = self.config
        if c.method not in (
            "manuscript_random_search",
            "random_treatment_perturbation",
        ):
            raise ValueError(
                "method must be manuscript_random_search or "
                "random_treatment_perturbation"
            )
        if c.n_candidates < 1 or not np.isfinite(c.lambda_gain) or c.lambda_gain <= 0:
            raise ValueError("Positive candidate count and lambda_gain are required")
        if c.target not in ("predicted", "observed"):
            raise ValueError(
                "CF target must be predicted (Algorithm 1) or observed (sensitivity analysis)"
            )

    def generate_candidates(
        self,
        treatment,
        treatment_type: str = "binary",
        feasible=None,
        *,
        exclude_original: bool | None = None,
    ) -> list[np.ndarray]:
        t = np.asarray(treatment, dtype=float)
        if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
            raise ValueError("Treatments must be a nonempty finite vector")
        rng = np.random.default_rng(self.seed)
        c = self.config
        exclude = c.exclude_original if exclude_original is None else exclude_original
        if treatment_type in ("binary", "multi"):
            categories = np.asarray(
                [0, 1] if treatment_type == "binary" else feasible, dtype=float
            )
            if (
                categories.ndim != 1
                or len(np.unique(categories)) < 2
                or not np.isfinite(categories).all()
            ):
                raise ValueError(
                    "Discrete search needs at least two finite numeric categories"
                )
            categories = np.unique(categories)
            if not np.isin(t, categories).all():
                raise ValueError("Observed treatment is not in feasible set")
        elif treatment_type == "continuous":
            lo, hi = c.lower_bound, c.upper_bound
            if lo is None or hi is None or not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError(
                    "Continuous search requires finite lower_bound < upper_bound"
                )
            if ((t < lo) | (t > hi)).any():
                raise ValueError("Observed treatment falls outside feasible bounds")
            if c.resolution is not None and (
                not np.isfinite(c.resolution) or c.resolution <= 0
            ):
                raise ValueError("resolution must be positive")
        else:
            raise ValueError("Unknown treatment_type")
        result = []
        for original in t:
            proposals = original + rng.normal(size=c.n_candidates)
            if treatment_type == "continuous":
                candidates = np.clip(proposals, lo, hi)
                if c.resolution is not None:
                    candidates = np.clip(
                        lo + np.round((candidates - lo) / c.resolution) * c.resolution,
                        lo,
                        hi,
                    )
                if not exclude:
                    candidates = np.append(candidates, original)
            else:
                projected = categories[
                    np.abs(proposals[:, None] - categories).argmin(axis=1)
                ]
                candidates = np.concatenate([projected, categories])
            candidates = np.unique(candidates)
            if exclude:
                candidates = candidates[candidates != original]
            if not len(candidates):
                raise ValueError(
                    "No feasible alternative found; change bounds/resolution/search budget"
                )
            result.append(candidates)
        return result

    def objective(self, predicted, baseline, candidate, original) -> np.ndarray:
        """lambda*(E[Y|X,T'] - E[Y|X,T]) - |T'-T|, not a hard-label objective."""
        return self.config.lambda_gain * (np.asarray(predicted) - baseline) - np.abs(
            np.asarray(candidate) - original
        )

    def optimize(self, candidates, outcomes, baseline: float, original: float):
        scores = self.objective(outcomes, baseline, candidates, original)
        if not np.isfinite(scores).all():
            raise ValueError("Outcome model produced nonfinite predictions")
        # Tie-break by least distance, then lowest numeric treatment.
        order = np.lexsort((candidates, np.abs(candidates - original), -scores))
        return int(order[0]), scores

    def generate(
        self,
        X,
        treatment,
        outcome,
        predict_outcome: Callable,
        treatment_type: str = "binary",
        outcome_type: str = "binary",
        feasible=None,
    ) -> CounterfactualResult:
        X = frame(X)
        t, y = np.asarray(treatment, dtype=float), np.asarray(outcome, dtype=float)
        if len(t) != len(X) or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError(
                "CF generation inputs must have equal lengths and finite outcomes"
            )
        if self.config.method == "random_treatment_perturbation":
            return self._generate_random_perturbation(
                X,
                t,
                predict_outcome,
                treatment_type,
                outcome_type,
                feasible,
            )
        # Evaluate the relaxed set once. The forced Algorithm-1 set is the
        # treatment-changing subset of the same candidates, so both views use
        # identical out-of-fold predictions.
        relaxed_candidates = self.generate_candidates(
            t, treatment_type, feasible, exclude_original=False
        )
        baseline = np.asarray(predict_outcome(X, t), dtype=float)
        if baseline.shape != t.shape or not np.isfinite(baseline).all():
            raise ValueError("Outcome predictor must return one finite value per row")
        relaxed_outcomes = []
        forced_candidates, forced_outcomes = [], []
        chosen_t, chosen_y, chosen_j = [], [], []
        relaxed_t, relaxed_y, relaxed_j = [], [], []
        # Batch evaluation bounds memory and avoids one model call per customer.
        batch_size = 512
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            lengths = [len(a) for a in relaxed_candidates[start:end]]
            rows = np.repeat(np.arange(start, end), lengths)
            values = np.concatenate(relaxed_candidates[start:end])
            pred = np.asarray(predict_outcome(X.iloc[rows], values), dtype=float)
            if pred.shape != values.shape or not np.isfinite(pred).all():
                raise ValueError("Invalid candidate predictions")
            batch_outcomes = np.split(pred, np.cumsum(lengths)[:-1])
            for i, prediction in enumerate(batch_outcomes, start):
                relaxed_candidate = relaxed_candidates[i]
                relaxed_k, relaxed_scores = self.optimize(
                    relaxed_candidate, prediction, baseline[i], t[i]
                )
                relaxed_outcomes.append(prediction)
                relaxed_t.append(relaxed_candidate[relaxed_k])
                relaxed_y.append(prediction[relaxed_k])
                relaxed_j.append(relaxed_scores[relaxed_k])

                changed = relaxed_candidate != t[i]
                candidate = relaxed_candidate[changed]
                candidate_prediction = prediction[changed]
                if not len(candidate):
                    raise ValueError(
                        "No feasible alternative found; change bounds/resolution/search budget"
                    )
                forced_k, forced_scores = self.optimize(
                    candidate, candidate_prediction, baseline[i], t[i]
                )
                forced_candidates.append(candidate)
                forced_outcomes.append(candidate_prediction)
                chosen_t.append(candidate[forced_k])
                chosen_y.append(candidate_prediction[forced_k])
                chosen_j.append(forced_scores[forced_k])
        chosen_y = np.asarray(chosen_y)
        if outcome_type not in ("binary", "continuous"):
            raise ValueError("Unknown outcome_type")
        if treatment_type == "binary":
            changed_to_treatment = (
                (t == 0) & (np.asarray(chosen_t) == 1) & (chosen_y > baseline)
            )
            changed_to_control = (
                (t == 1) & (np.asarray(chosen_t) == 0) & (chosen_y <= baseline)
            )
            valid = changed_to_treatment | changed_to_control
        else:
            # The asymmetric binary rule has no unique analogue for unordered
            # multi-arm or continuous treatments; retain improvement validity.
            valid = chosen_y > baseline
        return CounterfactualResult(
            np.asarray(chosen_t),
            chosen_y,
            baseline,
            np.asarray(chosen_j),
            valid,
            forced_candidates,
            forced_outcomes,
            np.asarray(relaxed_t),
            np.asarray(relaxed_y),
            np.asarray(relaxed_j),
            relaxed_candidates,
            relaxed_outcomes,
        )

    def _generate_random_perturbation(
        self,
        X,
        treatment,
        predict_outcome: Callable,
        treatment_type: str,
        outcome_type: str,
        feasible,
    ) -> CounterfactualResult:
        """Select treatment randomly, without using predicted outcomes.

        Candidate construction is unchanged, but one candidate is sampled
        uniformly for each row instead of maximizing the counterfactual
        objective.  The relaxed candidate set is used intentionally.  For a
        binary treatment this means an outcome-independent Bernoulli assignment
        and an expected 50% perturbation rate.  Conditioning on a change would
        reduce the binary ablation to the deterministic flip used by the forced
        optimization path.

        Predicted outcomes, objectives, validity, and Stage-2 acceptance are
        still calculated exactly as in the main method.  In particular,
        ``min_uplift=0`` rejects unchanged random assignments because their
        predicted directional uplift is zero.
        """
        t = np.asarray(treatment, dtype=float)
        if outcome_type not in ("binary", "continuous"):
            raise ValueError("Unknown outcome_type")

        candidates = self.generate_candidates(
            t,
            treatment_type,
            feasible,
            exclude_original=False,
        )
        baseline = np.asarray(predict_outcome(X, t), dtype=float)
        if baseline.shape != t.shape or not np.isfinite(baseline).all():
            raise ValueError("Outcome predictor must return one finite value per row")

        # Keep candidate generation and random selection on separate seeded
        # streams so the selection remains stable if proposal internals change.
        rng = np.random.default_rng(self.seed + 1_000_003)
        chosen_indices = np.asarray(
            [rng.integers(0, len(row_candidates)) for row_candidates in candidates]
        )
        chosen_t = np.asarray(
            [
                row_candidates[index]
                for row_candidates, index in zip(candidates, chosen_indices)
            ]
        )

        candidate_outcomes: list[np.ndarray] = []
        batch_size = 512
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            lengths = [len(a) for a in candidates[start:end]]
            rows = np.repeat(np.arange(start, end), lengths)
            values = np.concatenate(candidates[start:end])
            predictions = np.asarray(
                predict_outcome(X.iloc[rows], values), dtype=float
            )
            if predictions.shape != values.shape or not np.isfinite(predictions).all():
                raise ValueError("Invalid candidate predictions")
            candidate_outcomes.extend(
                np.split(predictions, np.cumsum(lengths)[:-1])
            )

        chosen_y = np.asarray(
            [
                row_outcomes[index]
                for row_outcomes, index in zip(candidate_outcomes, chosen_indices)
            ]
        )
        objective = self.objective(chosen_y, baseline, chosen_t, t)
        if not np.isfinite(objective).all():
            raise ValueError("Outcome model produced nonfinite predictions")

        if treatment_type == "binary":
            changed_to_treatment = (t == 0) & (chosen_t == 1) & (chosen_y > baseline)
            changed_to_control = (t == 1) & (chosen_t == 0) & (chosen_y <= baseline)
            valid = changed_to_treatment | changed_to_control
        else:
            valid = (chosen_t != t) & (chosen_y > baseline)

        # There is only one random selection, so the primary and sensitivity
        # views coincide. Copies prevent downstream mutation from aliasing them.
        return CounterfactualResult(
            chosen_t,
            chosen_y,
            baseline,
            objective,
            valid,
            candidates,
            candidate_outcomes,
            chosen_t.copy(),
            chosen_y.copy(),
            objective.copy(),
            [a.copy() for a in candidates],
            [a.copy() for a in candidate_outcomes],
        )
