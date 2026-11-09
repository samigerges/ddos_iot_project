"""Class weighting shared by training and controlled multiclass experiments."""

from __future__ import annotations

import numpy as np


def balanced_class_weights(y: np.ndarray, num_classes: int, power: float = 1.0) -> np.ndarray:
    """Temper inverse frequency, preserving a mean *sample* weight of one."""
    if not np.isfinite(power) or power < 0:
        raise ValueError("Class weight power must be finite and nonnegative.")
    values = np.asarray(y)
    if values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Labels must be a nonempty integer vector.")
    if np.any(values < 0) or np.any(values >= num_classes):
        raise ValueError("Labels are outside the configured class range.")
    counts = np.bincount(values, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Every training partition must contain all {num_classes} classes.")
    # Work in log space so even large user-supplied powers cannot overflow.
    log_weights = power * np.log(len(values) / (num_classes * counts))
    weights = np.exp(log_weights - log_weights.max())
    weights /= np.dot(counts / len(values), weights)
    if not np.isfinite(weights).all() or np.any(weights.astype(np.float32) <= 0):
        raise ValueError("Class weight power is too large for positive float32 weights.")
    return weights.astype(np.float32)
