"""Pair reference/selected model timings after training, using validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.experiments.benchmarking import benchmark_saved_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(benchmark_saved_models(args.run_dir, seed=args.seed, repeats=args.repeats).to_string(index=False))


if __name__ == "__main__":
    main()
