"""Verified published-training input and explicit fit/tune/validation partitions."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ddos_iot.attack_families import FAMILY_NAMES, RAW_TO_FAMILY
from .storage import file_digest


@dataclass
class StudyData:
    x: pd.DataFrame
    y: np.ndarray
    encoder: LabelEncoder
    metadata: dict


def load_study_data(config) -> StudyData:
    manifest = json.loads((config.subset_directory / "preparation_report.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["test_csv_parsed"]:
        raise ValueError("Require completed, training-only reference preparation.")
    entry = manifest["training_subsets"][config.source_path.name]
    if file_digest(config.source_path) != entry["sha256"]:
        raise ValueError("Published training subset differs from its verified manifest.")
    features = entry["features"]
    if len(features) != 23:
        raise ValueError("This study requires the published 23-feature representation.")
    schema = list(pd.read_csv(config.source_path, nrows=0).columns)
    if schema != [*features, "label"]:
        raise ValueError("Published schema/order changed.")
    dtypes = {f: "float32" for f in features}; dtypes["label"] = "string"
    parts = list(pd.read_csv(config.source_path, dtype=dtypes, chunksize=250_000))
    frame = pd.concat(parts, ignore_index=True)
    del parts
    expected = set(FAMILY_NAMES) if config.num_classes == 8 else set(RAW_TO_FAMILY)
    if len(frame) != entry["rows"] or set(frame.label.dropna()) != expected or frame.label.isna().any():
        raise ValueError("Published rows/labels differ from their inventory.")
    if not np.isfinite(frame[features].to_numpy()).all():
        raise ValueError("Published training features must be finite.")
    encoder = LabelEncoder().fit(frame.label)
    y = encoder.transform(frame.label)
    return StudyData(frame[features], y, encoder, {
        "source_path": str(config.source_path.resolve()), "source_sha256": entry["sha256"],
        "rows": len(y), "features": features, "classes": encoder.classes_.tolist(),
        "source_representation": entry["representation"], "upstream_cv_warning": entry["cv_warning"],
        "test_file_read": False})


def study_split(y, config, seed):
    development, validation = train_test_split(np.arange(len(y)), test_size=config.validation_size,
                                               random_state=seed, stratify=y)
    fitting, tuning = train_test_split(development, test_size=config.tuning_size_within_development,
                                       random_state=seed + 1, stratify=y[development])
    result = tuple(np.sort(ids) for ids in (fitting, tuning, validation))
    if any(np.bincount(y[ids], minlength=config.num_classes).min() < 2 for ids in result):
        raise ValueError("Each fit/tune/validation partition needs at least two rows per class.")
    return result


def protected_ids(classes, config):
    return tuple(i for i, label in enumerate(classes)
                 if (label if config.num_classes == 8 else RAW_TO_FAMILY[label]) in config.protected_families)
