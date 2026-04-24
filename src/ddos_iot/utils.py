"""Small shared utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def configure_logging(verbose: bool) -> None:
    """Configure command-line logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def class_distribution(target: pd.Series) -> dict[str, int]:
    """Return class counts as JSON-friendly strings."""

    counts = target.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def make_json_safe(value: Any) -> Any:
    """Convert common numpy/pandas/path objects into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def predict_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return class-1 scores from a fitted binary classifier."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
    return model.predict(features).astype(float)
