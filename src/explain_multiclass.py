"""Explain a saved 8-/34-class model with SHAP; no training or CV."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.multiclass_shap import run_multiclass_shap
from ddos_iot.utils import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Trusted local multiclass model.joblib.")
    parser.add_argument("--input-file", type=Path, required=True, help="Raw feature CSV to explain (label is optional/ignored).")
    parser.add_argument("--output-path", type=Path, help="New folder; default: a unique run under the model's shap/ directory.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-scan-rows", type=int, default=50_000, help="Rows scanned per CSV; 0 scans the full CSV in chunks.")
    parser.add_argument("--method", choices=("auto", "native", "permutation"), default="auto")
    parser.add_argument("--background-file", type=Path, help="Separate training CSV; required only for permutation SHAP/fallback.")
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--fallback-size", type=int, default=20, help="Maximum explained rows for the slower permutation method.")
    parser.add_argument("--local-count", type=int, default=5)
    parser.add_argument("--max-display", type=int, default=15)
    parser.add_argument("--class-label", help="Class for exported class plots; arrays contain all classes regardless.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2, help="Positive CPU thread count; SHAP does not use the training GPU.")
    parser.add_argument("--no-plots", action="store_true", help="Save arrays/tables only; the dashboard can plot them later.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if min(args.sample_size, args.background_size, args.fallback_size, args.max_display, args.n_jobs) <= 0:
        parser.error("Sample sizes, feature display count, and CPU threads must be positive.")
    if args.max_scan_rows < 0 or args.local_count < 0:
        parser.error("Scan limit and local count cannot be negative.")
    if args.method == "permutation" and args.background_file is None:
        parser.error("--method permutation requires --background-file from training data.")
    return args


def main() -> None:
    args = vars(parse_args())
    configure_logging(args.pop("verbose"))
    run_multiclass_shap(**args)


if __name__ == "__main__":
    main()
