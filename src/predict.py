"""Run predictions with a trained sklearn model on processed feature files."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.modeling import run_prediction
from ddos_iot.paths import PREDICTIONS_DIR
from ddos_iot.utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/lightgbm/model.joblib"))
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=PREDICTIONS_DIR / "predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run prediction."""

    args = parse_args()
    configure_logging(args.verbose)
    run_prediction(args)


if __name__ == "__main__":
    main()
