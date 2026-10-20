"""Train the paper's three boosting models on 34 raw labels or eight attack families."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.multiclass import MODEL_NAMES, run_multiclass_training
from ddos_iot.utils import configure_logging


def parse_args(argv: list[str] | None = None, *, default_num_classes: int = 34) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-classes", type=int, choices=(8, 34), default=default_num_classes)
    parser.add_argument("--data-protocol", choices=("existing", "paper"), default=None,
                        help="Default: existing split for 34 classes; paper sampling/new split for eight.")
    parser.add_argument("--source-input-path", nargs="+", type=Path,
                        help="Raw source CSVs for paper mode; defaults to train-input-path ONLY.")
    parser.add_argument("--sampling-target", type=int, default=5_000_000)
    parser.add_argument("--min-samples-per-class", type=int, default=1000,
                        help="Minimum per family before splitting in eight-class paper mode.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="New test fraction in paper mode, separate from inner validation-size.")
    parser.add_argument("--cv-preprocessing", choices=("fold", "train"), default=None,
                        help="fold: fit within CV folds; train: fit once before CV as in the paper. "
                             "Default: fold for 34 classes, train for eight.")
    parser.add_argument("--train-input-path", type=Path, default=Path("data/raw/train/train.csv"))
    parser.add_argument("--test-input-path", type=Path, default=Path("data/raw/test/test.csv"))
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                        help="Device for every CV/final fit; GPU also enables XGBoost CV prediction on CUDA.")
    parser.add_argument("--lightgbm-device", choices=("cpu", "gpu"), default=None,
                        help="Override --device for LightGBM (CPU is useful after an OpenCL failure).")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-train-per-class", type=int, default=0,
                        help="Uniform random sample per class; 0 uses every valid training row.")
    parser.add_argument("--max-test-per-class", type=int, default=0,
                        help="0 evaluates every valid test row. A positive value is for smoke tests only.")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced",
                        help="balanced uses the paper's N / (K * class_count) weights.")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--feature-selection", choices=("median", "none"), default="median",
                        help="Keep gain strictly above the median of ALL 46 original features (default).")
    parser.add_argument("--selector-sample-size", type=int, default=200_000)
    parser.add_argument("--cv-folds", type=int, choices=(0, 5), default=5,
                        help="Five-fold grid-search tuning is ON by default; 0 explicitly disables tuning.")
    parser.add_argument("--param-grid", type=Path,
                        help="JSON grid per model using n_estimators, learning_rate, and max_depth.")
    parser.add_argument("--refit-full", action=argparse.BooleanOptionalAction, default=True,
                        help="Refit using all selected training rows and validation-selected tree counts.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-period", type=int, default=25)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--counts-only", action="store_true",
                        help="Scan labels and save raw per-class counts without training.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.data_protocol = args.data_protocol or ("paper" if args.num_classes == 8 else "existing")
    args.cv_preprocessing = args.cv_preprocessing or ("train" if args.num_classes == 8 else "fold")
    args.output_path = args.output_path or Path(f"models/multiclass_{args.num_classes}_tuned")
    if args.data_protocol == "paper":
        if args.num_classes != 8:
            parser.error("--data-protocol paper currently supports the eight-class family task only")
        if args.max_train_per_class or args.max_test_per_class:
            parser.error("Paper sampling uses --sampling-target and --min-samples-per-class, not per-split caps")
        args.source_input_path = args.source_input_path or [args.train_input_path]
        if args.sampling_target < 8 * args.min_samples_per_class or args.min_samples_per_class <= 0:
            parser.error("--sampling-target must be at least 8 * --min-samples-per-class, with positive values")
    elif args.source_input_path:
        parser.error("--source-input-path requires --data-protocol paper")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    for name in ("chunk_size", "n_estimators", "max_depth", "early_stopping_rounds",
                 "selector_sample_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_per_class", "max_test_per_class", "gpu_id", "log_period"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if not 0 < args.validation_size < 1 or not 0 < args.learning_rate <= 1:
        parser.error("validation-size must be between 0 and 1; learning-rate must be in (0, 1]")
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a positive integer")
    if args.feature_selection == "median" and args.selector_sample_size < args.num_classes:
        parser.error(f"--selector-sample-size must be at least {args.num_classes}")
    if args.param_grid and not args.cv_folds:
        parser.error("--param-grid requires --cv-folds 5")
    return args


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    run_multiclass_training(args)


if __name__ == "__main__":
    main()
