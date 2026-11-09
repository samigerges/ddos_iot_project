"""Deterministic multiclass bias tuning using training-only OOF predictions."""

from __future__ import annotations

import numpy as np


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must be a finite N x K array in [0, 1].")
    if len(p) and not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Probability rows must sum to one.")
    return p


def predict_with_bias(probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    p = validate_probabilities(probabilities)
    bias = np.asarray(bias)
    if bias.shape != (p.shape[1],) or not np.isfinite(bias).all():
        raise ValueError("Provide one finite bias per probability column.")
    return np.argmax(np.log(np.maximum(p, 1e-12)) + bias, axis=1)


def class_f1(y: np.ndarray, predicted: np.ndarray, num_classes: int) -> np.ndarray:
    support = np.bincount(y, minlength=num_classes)
    predicted_counts = np.bincount(predicted, minlength=num_classes)
    tp = np.bincount(y[y == predicted], minlength=num_classes)
    return np.divide(2 * tp, support + predicted_counts, out=np.zeros(num_classes, dtype=float),
                     where=(support + predicted_counts) != 0)


def tune_biases(y: np.ndarray, probabilities: np.ndarray, *, grid: tuple[float, ...],
                 passes: int = 2, f1_cutoff: float = 0.95) -> tuple[np.ndarray, dict]:
    """Coordinate search; test and independent validation labels never enter here.

    Compare each class against its strongest competitor in O(N) per trial,
    instead of allocating an N x K score matrix for every grid point.
    """
    p = validate_probabilities(probabilities)
    y = np.asarray(y)
    k = p.shape[1]
    if y.shape != (len(p),) or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("OOF labels must be a nonempty aligned integer vector.")
    if np.any(y < 0) or np.any(y >= k) or np.bincount(y, minlength=k).min() == 0:
        raise ValueError("OOF labels must cover every probability column.")
    if not grid or not np.isfinite(grid).all() or passes < 1 or not 0 < f1_cutoff <= 1:
        raise ValueError("Invalid bias tuning settings.")
    scores = np.log(np.maximum(p, 1e-12)).astype(np.float64)
    bias = np.zeros(k)
    predicted = scores.argmax(axis=1)
    original_f1 = class_f1(y, predicted, k)
    current_score = float(original_f1.mean())
    baseline = current_score
    # Anchor the largest class at zero; tune only imperfect classes to limit cost/variance.
    anchor = int(np.argmax(np.bincount(y, minlength=k)))
    eligible = [c for c in range(k) if c != anchor and original_f1[c] < f1_cutoff]
    rows = np.arange(len(y))
    history = []
    for iteration in range(passes):
        improved = False
        for c in eligible:
            base_score = np.log(np.maximum(p[:, c], 1e-12)).astype(np.float64)
            competitor = predicted.copy()
            own = np.flatnonzero(predicted == c)
            if len(own):
                others = scores[own].copy()
                others[:, c] = -np.inf
                competitor[own] = others.argmax(axis=1)
            competitor_score = scores[rows, competitor]
            best_bias, best_score, best_predictions = bias[c], current_score, predicted
            for value in sorted(set((*grid, 0., float(bias[c]))), key=lambda v: (abs(v), v)):
                if value == bias[c]:
                    continue
                candidate_score = base_score + value
                wins = (candidate_score > competitor_score) | ((candidate_score == competitor_score) & (c < competitor))
                proposed = np.where(wins, c, competitor)
                score = float(class_f1(y, proposed, k).mean())
                if score > best_score + 1e-12:
                    best_bias, best_score, best_predictions = value, score, proposed
            if best_score > current_score + 1e-12:
                bias[c], current_score, predicted = best_bias, best_score, best_predictions
                scores[:, c] = base_score + best_bias
                improved = True
                history.append({"pass": iteration + 1, "class_id": c, "bias": best_bias,
                                "oof_tuning_macro_f1": best_score})
        if not improved:
            break
    return bias, {"bias": bias.tolist(), "anchor_class_id": anchor, "eligible_class_ids": eligible,
                  "baseline_oof_macro_f1": baseline, "tuned_oof_macro_f1": current_score,
                  "score_role": "Training-only optimization score, not an independent performance estimate.",
                  "history": history, "test_used_for_tuning": False, "validation_used_for_tuning": False}
