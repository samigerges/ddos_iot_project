"""Measure paired complete-bundle CPU throughput after research fits finish."""

import argparse
from pathlib import Path

from ddos_iot.experiments.study_benchmark import benchmark_study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeats", type=int, help="Default: one complete rotation of all measured models.")
    args = parser.parse_args()
    print(benchmark_study(args.run_dir, args.output_dir or args.run_dir / "paired_timing_v1",
                          rows=args.rows, repeats=args.repeats)["status"])
