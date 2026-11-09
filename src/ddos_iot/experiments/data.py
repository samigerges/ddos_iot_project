"""Common raw-row splits for both label granularities; never open the test here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ddos_iot.attack_families import RAW_TO_FAMILY, map_family_labels
from ddos_iot.multiclass import load_raw_split, read_schema
from .config import ExperimentConfig
from .storage import file_digest


@dataclass
class TrainingData:
    x: pd.DataFrame
    y: np.ndarray
    raw_labels: np.ndarray
    row_ids: np.ndarray
    feature_hashes: np.ndarray
    encoder: LabelEncoder
    report: dict[str, Any]


def load_training_data(config: ExperimentConfig) -> TrainingData:
    features = read_schema(config.train_input_path, config.label_column)
    if len(features) != 46:
        raise ValueError("Controlled feature ablations require all 46 original numeric features.")
    frame, report = load_raw_split(
        config.train_input_path, config.label_column, config.max_train_per_raw_class,
        config.chunk_size, config.data_seed, 34, preserve_row_index=True)
    raw = frame[config.label_column]
    unknown = sorted(set(raw) - set(RAW_TO_FAMILY))
    if unknown or len(set(raw)) != 34:
        raise ValueError(f"The shared source must contain all 34 original labels. Unknown={unknown}")
    labels = map_family_labels(raw) if config.num_classes == 8 else raw
    encoder = LabelEncoder().fit(labels)
    raw_encoder = LabelEncoder().fit(raw)
    x = frame[features]
    report.update(source_sha256=file_digest(config.train_input_path),
                  original_feature_count=len(features), class_count=config.num_classes,
                  raw_label_mapping={str(i): label for i, label in enumerate(raw_encoder.classes_)},
                  raw_row_ids="Zero-based CSV data-row positions, excluding the header.",
                  test_file_read=False, sampled=bool(config.max_train_per_raw_class))
    return TrainingData(x, encoder.transform(labels), raw_encoder.transform(raw).astype(np.int16),
                        frame.index.to_numpy(dtype=np.int64),
                        pd.util.hash_pandas_object(x, index=False).to_numpy(), encoder, report)


def development_split(data: TrainingData, config: ExperimentConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratify on raw subtypes so 8/34-class runs use identical raw row splits."""
    development, validation = train_test_split(
        np.arange(len(data.y)), test_size=config.validation_size,
        random_state=seed, stratify=data.raw_labels)
    development, validation = np.sort(development), np.sort(validation)
    counts = pd.Series(data.raw_labels[development]).value_counts()
    if counts.min() < config.folds:
        raise ValueError("Every raw subtype needs at least folds development rows. Increase the source sample.")
    return development, validation


def overlap_report(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Diagnostic overlap of numeric flow vectors after float32 parsing."""
    left_unique = np.unique(left)
    overlaps = np.isin(right, left_unique)
    return {"right_rows": len(right), "overlapping_right_rows": int(overlaps.sum()),
            "overlapping_right_fraction": float(overlaps.mean()) if len(right) else 0.,
            "method": "64-bit hashes of all numeric float32 features; labels excluded.",
            "policy": "Reported, not removed. A grouped split is a separate protocol."}


def load_test_data(config: ExperimentConfig, data: TrainingData) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Called exclusively after the validation winner and final models are saved."""
    features = read_schema(config.test_input_path, config.label_column)
    if set(features) != set(data.x.columns):
        raise ValueError("Raw test features do not match the training schema.")
    frame, report = load_raw_split(config.test_input_path, config.label_column, 0,
                                   config.chunk_size, config.data_seed, 34, preserve_row_index=True)
    raw = frame[config.label_column]
    if set(raw) != set(RAW_TO_FAMILY):
        raise ValueError("The test source must cover all 34 original labels.")
    labels = map_family_labels(raw) if config.num_classes == 8 else raw
    x = frame[list(data.x.columns)]
    report["training_overlap"] = overlap_report(
        data.feature_hashes, pd.util.hash_pandas_object(x, index=False).to_numpy())
    report["source_sha256"] = file_digest(config.test_input_path)
    return x, data.encoder.transform(labels), report
