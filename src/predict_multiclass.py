"""Predict eight families or 34 labels with an existing model, without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.multiclass_inference import run_multiclass_prediction
from ddos_iot.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Trusted local model.joblib from a multiclass run.")
    parser.add_argument("--input-file", type=Path, required=True, help="Raw CSV with the model's numeric features.")
    parser.add_argument("--output-file", type=Path, default=Path("outputs/predictions/multiclass_predictions.csv"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    configure_logging(args.verbose)
    run_multiclass_prediction(args.model_path, args.input_file, args.output_file, args.chunk_size)


if __name__ == "__main__":
    main()
