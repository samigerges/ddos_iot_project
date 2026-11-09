"""OOF-only bias search with explicit minority-recall and Benign-FPR limits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .decisions import validate_probabilities


@dataclass(frozen=True)
class DecisionConstraints:
    benign_class_id: int
    minimum_recall: dict[int, float]
    max_benign_fpr: float

    def __post_init__(self) -> None:
        ids = [self.benign_class_id, *self.minimum_recall]
        if any(type(c) is not int or c < 0 for c in ids):
            raise ValueError("Constraint class IDs must be nonnegative integers.")
        if not self.minimum_recall or self.benign_class_id in self.minimum_recall:
            raise ValueError("Protect at least one attack class, excluding Benign.")
        for value in [self.max_benign_fpr, *self.minimum_recall.values()]:
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Constraint rates must be finite fractions in [0, 1].")

    def as_dict(self) -> dict:
        return {"benign_class_id": self.benign_class_id,
                "minimum_recall": {str(c): float(v) for c, v in self.minimum_recall.items()},
                "max_benign_fpr": self.max_benign_fpr}


def decision_statistics(y: np.ndarray, predicted: np.ndarray, k: int) -> dict:
    matrix = np.bincount(y * k + predicted, minlength=k * k).reshape(k, k)
    support, counts, tp = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    recall = np.divide(tp, support, out=np.zeros(k, dtype=float), where=support != 0)
    f1 = np.divide(2 * tp, support + counts, out=np.zeros(k, dtype=float), where=(support + counts) != 0)
    return {"macro_f1": float(f1.mean()), "recall": recall.tolist(), "f1": f1.tolist(),
            "support": support.tolist(), "true_positives": tp.tolist()}


def constraint_statistics(stats: dict, constraints: DecisionConstraints) -> dict:
    support, tp = stats["support"], stats["true_positives"]
    b = constraints.benign_class_id
    deficits = {str(c): max(0., floor - stats["recall"][c])
                for c, floor in constraints.minimum_recall.items()}
    fpr = 1. - stats["recall"][b]
    # Count comparisons avoid a one-sample tolerance hidden in rounded rates.
    recall_ok = all(tp[c] + 1e-9 >= floor * support[c]
                    for c, floor in constraints.minimum_recall.items())
    fpr_ok = support[b] - tp[b] <= constraints.max_benign_fpr * support[b] + 1e-9
    return {"feasible": bool(recall_ok and fpr_ok), "recall_constraints_pass": bool(recall_ok),
            "fpr_constraint_pass": bool(fpr_ok), "benign_false_positive_rate": float(fpr),
            "recall_deficits": deficits,
            "violation": float(sum(deficits.values()) + max(0., fpr - constraints.max_benign_fpr))}


def reference_constraints(y: np.ndarray, predicted: np.ndarray, k: int, *, benign_id: int,
                          protected_ids: Iterable[int]) -> DecisionConstraints:
    protected = tuple(protected_ids)
    if not protected or len(set(protected)) != len(protected):
        raise ValueError("Protected classes must be nonempty and distinct.")
    if any(type(c) is not int or not 0 <= c < k for c in (benign_id, *protected)):
        raise ValueError("Constraint class IDs do not match the probability columns.")
    stats = decision_statistics(y, predicted, k)
    if min(stats["support"]) <= 0:
        raise ValueError("Reference labels must cover every class.")
    return DecisionConstraints(benign_id, {c: stats["recall"][c] for c in protected},
                               1. - stats["recall"][benign_id])


def tune_constrained_biases(y: np.ndarray, probabilities: np.ndarray, *,
                            constraints: DecisionConstraints, grid: tuple[float, ...],
                            passes: int = 3,
                            starts: Iterable[np.ndarray] = ()) -> tuple[np.ndarray | None, dict]:
    """Deterministic multistart coordinate search; no validation inputs are accepted.

    First reduce summed rate violations, then maximize Macro-F1 once feasible.
    A feasible search path never accepts an infeasible move. Failure to find a
    feasible point is a search outcome, not proof that one does not exist.
    """
    p = validate_probabilities(probabilities)
    y = np.asarray(y)
    k = p.shape[1]
    if y.shape != (len(p),) or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("OOF labels must be a nonempty aligned integer vector.")
    if np.any(y < 0) or np.any(y >= k) or np.bincount(y, minlength=k).min() == 0:
        raise ValueError("OOF labels must cover every probability column.")
    if max(constraints.benign_class_id, *constraints.minimum_recall) >= k:
        raise ValueError("Constraint classes do not match probability columns.")
    if not grid or not np.isfinite(grid).all() or type(passes) is not int or passes < 1:
        raise ValueError("Use a finite nonempty grid and positive integer passes.")
    anchor = int(np.argmax(np.bincount(y, minlength=k)))
    base = np.log(np.maximum(p, 1e-12)).astype(np.float64)
    initial = [np.zeros(k)]
    for value in starts:
        value = np.asarray(value, dtype=float)
        if value.shape != (k,) or not np.isfinite(value).all():
            raise ValueError("Each starting bias must contain one finite value per class.")
        value = value - value[anchor]
        if not any(np.array_equal(value, old) for old in initial):
            initial.append(value)
    values = sorted(set((*grid, 0.)), key=lambda v: (abs(v), v))
    row_ids = np.arange(len(y))
    histories, best = [], None

    def assess(predicted: np.ndarray) -> dict:
        stats = decision_statistics(y, predicted, k)
        return {**stats, **constraint_statistics(stats, constraints)}

    def better(left: dict, right: dict) -> bool:
        if left["feasible"] != right["feasible"]:
            return left["feasible"]
        if not left["feasible"] and abs(left["violation"] - right["violation"]) > 1e-12:
            return left["violation"] < right["violation"]
        return left["macro_f1"] > right["macro_f1"] + 1e-12

    baseline = assess(base.argmax(axis=1))
    # Try the protected classes early, then every other non-anchor class.
    eligible = list(dict.fromkeys([*constraints.minimum_recall, constraints.benign_class_id,
                                  *range(k)]))
    eligible = [c for c in eligible if c != anchor]
    for start_id, start in enumerate(initial):
        bias = start.copy()
        scores = base + bias
        predicted = scores.argmax(axis=1)
        current = assess(predicted)
        history = []
        for iteration in range(passes):
            changed = False
            for c in eligible:
                competitor = predicted.copy()
                own = np.flatnonzero(predicted == c)
                if len(own):
                    others = scores[own].copy()
                    others[:, c] = -np.inf
                    competitor[own] = others.argmax(axis=1)
                competitor_score = scores[row_ids, competitor]
                best_value, local_best, local_predictions = bias[c], current, predicted
                for value in values:
                    if value == bias[c]:
                        continue
                    own_score = base[:, c] + value
                    wins = (own_score > competitor_score) | ((own_score == competitor_score) & (c < competitor))
                    proposed = np.where(wins, c, competitor)
                    stats = assess(proposed)
                    if better(stats, local_best):
                        best_value, local_best, local_predictions = value, stats, proposed
                if better(local_best, current):
                    bias[c], current, predicted = best_value, local_best, local_predictions
                    scores[:, c] = base[:, c] + best_value
                    changed = True
                    history.append({"pass": iteration + 1, "class_id": c, "bias": float(best_value),
                                    "macro_f1": current["macro_f1"], "feasible": current["feasible"],
                                    "violation": current["violation"]})
            if not changed:
                break
        histories.append({"start": start_id, "initial_bias": start.tolist(), "final_bias": bias.tolist(),
                          "final_statistics": current, "history": history})
        if best is None or better(current, best[1]):
            best = (bias.copy(), current)
    bias, stats = best
    report = {"status": "feasible" if stats["feasible"] else "no_feasible_point_found",
              "bias": bias.tolist() if stats["feasible"] else None,
              "constraints": constraints.as_dict(), "baseline": baseline, "tuned": stats,
              "grid": values, "passes": passes, "anchor_class_id": anchor, "starts": histories,
              "method": "Multistart coordinate search; feasibility first, then Macro-F1. No global-optimum guarantee.",
              "score_role": "Training OOF optimization diagnostics; not independent performance estimates.",
              "test_used_for_tuning": False, "validation_used_for_tuning": False}
    return (bias if stats["feasible"] else None), report
