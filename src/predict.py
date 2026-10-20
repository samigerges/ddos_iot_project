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
    parser.add_argument(
        "--explain-local-shap",
        action="store_true",
        help="Generate local SHAP explanations for the prediction input rows.",
    )
    parser.add_argument(
        "--background-file",
        type=Path,
        default=None,
        help="Processed CSV used as the SHAP background. Defaults to the model's processed train.csv when available.",
    )
    parser.add_argument(
        "--shap-output-dir",
        type=Path,
        default=None,
        help="Directory for prediction-time SHAP outputs. Defaults to a sibling folder next to --output-file.",
    )
    parser.add_argument("--shap-background-size", type=int, default=200)
    parser.add_argument("--shap-waterfall-count", type=int, default=10)
    parser.add_argument("--shap-max-display", type=int, default=15)
    parser.add_argument(
        "--shap-model-agnostic-sample-size",
        type=int,
        default=0,
        help="Rows to explain for slower model-agnostic SHAP fallback. Use 0 to explain all input rows.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run prediction."""

    args = parse_args()
    configure_logging(args.verbose)
    run_prediction(args)


if __name__ == "__main__":
    main()
