"""Preprocess CICIOT-style IoT traffic data for Benign vs DDoS detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.preprocessing import run_preprocessing
from ddos_iot.utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input-path", type=Path, default=Path("data/raw/train/train.csv"))
    parser.add_argument("--test-input-path", type=Path, default=Path("data/raw/test/test.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--top-features", type=int, default=25)
    parser.add_argument(
        "--rebalance",
        choices=("none", "undersample", "oversample"),
        default="undersample",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run preprocessing."""

    args = parse_args()
    configure_logging(args.verbose)
    run_preprocessing(args)


if __name__ == "__main__":
    main()
