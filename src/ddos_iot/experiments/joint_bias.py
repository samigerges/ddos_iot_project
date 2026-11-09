"""Joint Benign/target bias moves that can escape constrained coordinate traps."""

import numpy as np

from .decisions import validate_probabilities
from .recall_constraints import constraint_statistics, decision_statistics


def tune_joint_biases(y, probabilities, constraints, initial, *, grid, passes=1):
    p = validate_probabilities(probabilities)
    y = np.asarray(y)
    k = p.shape[1]
    bias = np.asarray(initial, dtype=float).copy()
    if y.shape != (len(p),) or not len(y) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= k):
        raise ValueError("Joint tuning needs aligned nonempty integer labels.")
    if np.bincount(y, minlength=k).min() == 0 or bias.shape != (k,) or not np.isfinite(bias).all():
        raise ValueError("Cover every class and provide one finite initial bias per column.")
    if not grid or not np.isfinite(grid).all() or type(passes) is not int or passes < 1:
        raise ValueError("Invalid joint search settings.")
    if max(constraints.benign_class_id, *constraints.minimum_recall) >= k:
        raise ValueError("Constraint class mapping differs from probabilities.")
    base = np.log(np.maximum(p, 1e-12)).astype(np.float64)
    scores = base + bias
    predicted = scores.argmax(axis=1)
    rows = np.arange(len(y))

    def assess(ids):
        stats = decision_statistics(y, ids, k)
        return {**stats, **constraint_statistics(stats, constraints)}

    def better(a, b):
        if a["feasible"] != b["feasible"]: return a["feasible"]
        if not a["feasible"] and abs(a["violation"] - b["violation"]) > 1e-12:
            return a["violation"] < b["violation"]
        return a["macro_f1"] > b["macro_f1"] + 1e-12

    current = assess(predicted)
    initial_stats = current
    history = []
    benign = constraints.benign_class_id
    for iteration in range(passes):
        changed = False
        for target in constraints.minimum_recall:
            competitor = predicted.copy()
            own = np.flatnonzero(np.isin(predicted, [benign, target]))
            if len(own):
                others = scores[own].copy()
                others[:, [benign, target]] = -np.inf
                competitor[own] = others.argmax(axis=1)
            # With only two classes there is no outside competitor.
            other_score = scores[rows, competitor]
            if k == 2: other_score = np.full(len(y), -np.inf)
            values_b = sorted(set((*grid, float(bias[benign]))), key=lambda v: (abs(v), v))
            values_t = sorted(set((*grid, float(bias[target]))), key=lambda v: (abs(v), v))
            best_stats, best_pred, best_values = current, predicted, (bias[benign], bias[target])
            for b in values_b:
                benign_score = base[:, benign] + b
                wins = (benign_score > other_score) | ((benign_score == other_score) & (benign < competitor))
                ids_b = np.where(wins, benign, competitor)
                score_b = np.where(wins, benign_score, other_score)
                for t in values_t:
                    if b == bias[benign] and t == bias[target]: continue
                    target_score = base[:, target] + t
                    wins_t = (target_score > score_b) | ((target_score == score_b) & (target < ids_b))
                    proposed = np.where(wins_t, target, ids_b)
                    stats = assess(proposed)
                    if better(stats, best_stats):
                        best_stats, best_pred, best_values = stats, proposed, (b, t)
            if better(best_stats, current):
                bias[benign], bias[target] = best_values
                scores[:, benign] = base[:, benign] + bias[benign]
                scores[:, target] = base[:, target] + bias[target]
                current, predicted = best_stats, best_pred
                history.append({"pass": iteration + 1, "target_id": target,
                                "benign_bias": float(bias[benign]), "target_bias": float(bias[target]),
                                "macro_f1": current["macro_f1"], "feasible": current["feasible"]})
                changed = True
        if not changed: break
    return (bias if current["feasible"] else None), {
        "status": "feasible" if current["feasible"] else "no_feasible_point_found",
        "bias": bias.tolist() if current["feasible"] else None, "initial": initial_stats, "tuned": current,
        "constraints": constraints.as_dict(), "grid": list(grid), "passes": passes, "history": history,
        "method": "Joint Benign/target moves; feasibility first, then Macro-F1. Finite heuristic, not a global optimum.",
        "validation_used_for_tuning": False, "test_used_for_tuning": False,
        "score_role": "Training-only OOF optimization diagnostics."}


def coordinate_start(bias, report):
    if bias is not None: return bias
    starts = report["starts"]
    chosen = min(starts, key=lambda s: (not s["final_statistics"]["feasible"], s["final_statistics"]["violation"],
                                       -s["final_statistics"]["macro_f1"]))
    return np.asarray(chosen["final_bias"])
