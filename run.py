"""Run the full IoT DDoS detection pipeline from one command."""

from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ddos_iot.modeling import run_model_comparison
from ddos_iot.preprocessing import run_preprocessing
from ddos_iot.utils import configure_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_MODELS = (
    "random_forest",
    "xgboost",
    "hist_gradient_boosting",
    "lightgbm",
    "naive_bayes",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the complete pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train-input-path", type=Path, default=Path("data/raw/train/train.csv"))
    parser.add_argument("--test-input-path", type=Path, default=Path("data/raw/test/test.csv"))
    parser.add_argument("--processed-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-output-path", type=Path, default=Path("models/requested_model_comparison"))
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing processed files.")
    parser.add_argument("--skip-model-comparison", action="store_true", help="Only create processed files.")

    parser.add_argument("--label-column", default=None)
    parser.add_argument("--top-features", type=int, default=25)
    parser.add_argument("--rebalance", choices=("none", "undersample", "oversample"), default="undersample")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=0,
        help="Processed training rows to use after preprocessing. Use 0 to keep all rows.",
    )

    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument(
        "--select-metric",
        choices=("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"),
        default="f1",
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
    parser.add_argument(
        "--explain-shap",
        action="store_true",
        help="Generate SHAP global and local explanations for the selected best model.",
    )
    parser.add_argument(
        "--shap-for-all-models",
        action="store_true",
        help="When SHAP is enabled, explain every successful model.",
    )
    parser.add_argument("--shap-sample-size", type=int, default=1000)
    parser.add_argument("--shap-background-size", type=int, default=200)
    parser.add_argument("--shap-local-count", type=int, default=10)
    parser.add_argument("--shap-max-display", type=int, default=15)
    parser.add_argument("--shap-model-agnostic-sample-size", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_preprocessing_args(args: argparse.Namespace) -> Namespace:
    """Map full-pipeline arguments to preprocessing arguments."""

    return Namespace(
        train_input_path=args.train_input_path,
        test_input_path=args.test_input_path,
        output_path=args.processed_path,
        label_column=args.label_column,
        top_features=args.top_features,
        rebalance=args.rebalance,
        random_state=args.random_state,
        max_rows_per_file=args.max_rows_per_file,
        verbose=args.verbose,
    )


def build_model_comparison_args(args: argparse.Namespace) -> Namespace:
    """Map full-pipeline arguments to model-comparison arguments."""

    return Namespace(
        processed_path=args.processed_path,
        output_path=args.models_output_path,
        models=tuple(args.models),
        select_metric=args.select_metric,
        threshold=args.threshold,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        train_sample_size=args.train_sample_size,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        hgb_max_iter=args.hgb_max_iter,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_l2=args.hgb_l2,
        xgb_estimators=args.xgb_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        lgbm_estimators=args.lgbm_estimators,
        lgbm_max_depth=args.lgbm_max_depth,
        lgbm_num_leaves=args.lgbm_num_leaves,
        lgbm_learning_rate=args.lgbm_learning_rate,
        lgbm_subsample=args.lgbm_subsample,
        lgbm_colsample_bytree=args.lgbm_colsample_bytree,
        no_plots=args.no_plots,
        explain_shap=args.explain_shap,
        shap_for_all_models=args.shap_for_all_models,
        shap_sample_size=args.shap_sample_size,
        shap_background_size=args.shap_background_size,
        shap_local_count=args.shap_local_count,
        shap_max_display=args.shap_max_display,
        shap_model_agnostic_sample_size=args.shap_model_agnostic_sample_size,
        verbose=args.verbose,
    )


def main() -> None:
    """Run preprocessing and model comparison."""

    args = parse_args()
    configure_logging(args.verbose)

    if args.skip_preprocess and args.skip_model_comparison:
        raise ValueError("Nothing to run: both --skip-preprocess and --skip-model-comparison were passed.")

    if not args.skip_preprocess:
        LOGGER.info("Starting preprocessing")
        run_preprocessing(build_preprocessing_args(args))
    else:
        LOGGER.info("Skipping preprocessing; using %s", args.processed_path)

    if not args.skip_model_comparison:
        LOGGER.info("Starting model comparison")
        run_model_comparison(build_model_comparison_args(args))
    else:
        LOGGER.info("Skipping model comparison")

    LOGGER.info("Pipeline finished")
    LOGGER.info("Processed data: %s", args.processed_path)
    LOGGER.info("Model comparison: %s", args.models_output_path)


if __name__ == "__main__":
    main()
