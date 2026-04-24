"""Shared project constants."""

from __future__ import annotations

from typing import Literal


TARGET_COLUMN = "target"
UNKNOWN_CATEGORY = "__unknown__"
SUPPORTED_EXTENSIONS = {".csv"}
CLASS_NAMES = ["Benign", "DDoS"]

RebalanceStrategy = Literal["none", "undersample", "oversample"]
ModelName = Literal[
    "random_forest",
    "xgboost",
    "hist_gradient_boosting",
    "lightgbm",
    "naive_bayes",
]
