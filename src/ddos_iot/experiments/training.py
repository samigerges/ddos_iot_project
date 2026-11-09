"""One fold's selector, early stopping, and refit, all inside its training rows."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ddos_iot.multiclass import (
    FeatureTransform, best_rounds, build_model, fit_features, fit_model,
    predict_probabilities, scoring_device_for,
)
from ddos_iot.weighting import balanced_class_weights
from .config import Candidate, ExperimentConfig


@dataclass
class FittedPartition:
    model: Any
    features: FeatureTransform
    report: dict[str, Any]


def training_args(config: ExperimentConfig, candidate: Candidate, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_classes=config.num_classes, n_estimators=config.n_estimators,
        learning_rate=config.learning_rate, max_depth=config.max_depth,
        early_stopping_rounds=config.early_stopping_rounds, random_state=seed,
        n_jobs=config.n_jobs, device=config.device, gpu_id=config.gpu_id,
        lightgbm_device="cpu", log_period=0, feature_selection=candidate.feature_selection,
        selector_sample_size=config.selector_sample_size,
        # Fix the selector's weighting across the weight ablation.
        class_weight="balanced", class_weight_power=1.0,
    )


def fit_partition(x: pd.DataFrame, y: np.ndarray, strata: np.ndarray, config: ExperimentConfig,
                  candidate: Candidate, seed: int) -> FittedPartition:
    started = time.perf_counter()
    fit, stop = train_test_split(np.arange(len(y)), test_size=config.early_stopping_size,
                                random_state=seed, stratify=strata)
    args = training_args(config, candidate, seed)
    features = fit_features(x.iloc[fit], y[fit], args)
    transformed = features.transform(x)
    model = build_model(candidate.model, args, config.n_estimators, True)
    weights = balanced_class_weights(y[fit], config.num_classes, candidate.weight_power)
    fit_model(candidate.model, model, transformed.iloc[fit], y[fit], weights, args,
              (transformed.iloc[stop], y[stop]))
    rounds = best_rounds(candidate.model, model, config.n_estimators)
    del model
    gc.collect()
    model = build_model(candidate.model, args, rounds, False)
    weights = balanced_class_weights(y, config.num_classes, candidate.weight_power)
    fit_model(candidate.model, model, transformed, y, weights, args)
    return FittedPartition(model, features, {
        "training_rows": len(y), "selector_fit_rows": len(fit), "early_stopping_rows": len(stop),
        "selected_features": features.columns, "selected_feature_count": len(features.columns),
        "feature_selection": candidate.feature_selection, "selector_weight_power": 1.0,
        "model_weight_power": candidate.weight_power, "class_weights": weights,
        "best_rounds": rounds, "training_seconds": time.perf_counter() - started,
        "preprocessing_scope": "Inner fitting rows only; excludes early stopping and scoring rows.",
        "training_device": config.device if candidate.model != "lightgbm" else "cpu",
    })


def partition_probabilities(fitted: FittedPartition, x: pd.DataFrame, config: ExperimentConfig,
                            candidate: Candidate, seed: int) -> np.ndarray:
    args = training_args(config, candidate, seed)
    return predict_probabilities(candidate.model, fitted.model, fitted.features.transform(x),
                                 config.prediction_batch_size, device=scoring_device_for(candidate.model, args),
                                 gpu_id=config.gpu_id)
