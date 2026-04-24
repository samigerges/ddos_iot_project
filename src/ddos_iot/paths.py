"""Shared project paths and runtime cache setup."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "train"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_COMPARISON_DIR = MODELS_DIR / "requested_model_comparison"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

CACHE_DIR = PROJECT_ROOT / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"


def configure_runtime_environment() -> None:
    """Configure stable runtime cache locations for local execution."""

    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
