"""Train and evaluate sklearn models for Benign vs DDoS detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.modeling import run_training
from ddos_iot.utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-path", type=Path, default=Path("models/lightgbm"))
    parser.add_argument(
        "--model",
        choices=(
            "random_forest",
            "xgboost",
            "hist_gradient_boosting",
            "lightgbm",
            "naive_bayes",
        ),
        default="lightgbm",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=15)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--hgb-max-iter", type=int, default=150)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--hgb-l2", type=float, default=0.0)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--lgbm-estimators", type=int, default=200)
    parser.add_argument("--lgbm-max-depth", type=int, default=-1)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.1)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--explain-shap", action="store_true", help="Save SHAP global and local explanations.")
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=1000,
        help="Rows sampled from test data for SHAP.",
    )
    parser.add_argument(
        "--shap-background-size",
        type=int,
        default=200,
        help="Rows sampled from train data as SHAP background.",
    )
    parser.add_argument("--shap-local-count", type=int, default=10, help="Number of local SHAP rows to explain.")
    parser.add_argument("--shap-max-display", type=int, default=15, help="Maximum features shown in SHAP plots.")
    parser.add_argument(
        "--shap-model-agnostic-sample-size",
        type=int,
        default=100,
        help="Maximum explained rows for slower model-agnostic SHAP fallback.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run model training and evaluation."""

    args = parse_args()
    configure_logging(args.verbose)
    run_training(args)


if __name__ == "__main__":
    main()
