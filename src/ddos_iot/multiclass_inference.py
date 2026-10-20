"""Prediction-only use of persisted multiclass bundles; no training or CV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ddos_iot.artifacts import atomic_output
from ddos_iot.multiclass import load_multiclass_bundle

LOGGER = logging.getLogger(__name__)


def run_multiclass_prediction(model_path: Path, input_file: Path, output_file: Path,
                              chunk_size: int = 100_000, *,
                              expected_num_classes: int | None = None) -> dict[str, Any]:
    """Stream raw feature rows through the stored selector, scaler, and classifier."""
    if chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if output_file.resolve() in (input_file.resolve(), model_path.resolve()):
        raise ValueError("Prediction output must not overwrite the input or model file.")
    if output_file.exists():
        raise FileExistsError(f"Prediction output already exists: {output_file}. Choose a new --output-file.")
    bundle = load_multiclass_bundle(model_path)
    if expected_num_classes is not None and len(bundle.encoder.classes_) != expected_num_classes:
        raise ValueError(f"The saved model does not match the selected {expected_num_classes}-class task.")
    header = pd.read_csv(input_file, nrows=0)
    missing = sorted(set(bundle.features.columns) - set(header.columns))
    if missing:
        raise ValueError(f"Missing raw model features: {missing}")
    LOGGER.info("Loaded saved %s (%s classes); no training or CV will run", bundle.name, len(bundle.encoder.classes_))
    rows = 0
    with atomic_output(output_file) as stream:
        stream.write(b"predicted_label\n")
        with pd.read_csv(input_file, usecols=bundle.features.columns, dtype="float32", chunksize=chunk_size) as reader:
            for chunk in reader:
                predictions = pd.DataFrame({"predicted_label": bundle.predict(chunk)})
                stream.write(predictions.to_csv(index=False, header=False).encode("utf-8"))
                rows += len(chunk)
    LOGGER.info("Saved %s predictions to %s", f"{rows:,}", output_file)
    return {"model": bundle.name, "class_count": len(bundle.encoder.classes_), "rows": rows,
            "model_path": str(model_path.resolve()), "output_file": str(output_file.resolve())}
