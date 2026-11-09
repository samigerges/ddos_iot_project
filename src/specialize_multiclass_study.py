"""Test a specialist only when training errors meet the declared evidence gate."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_specialist import run_specialist_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expanded-fixed", action="store_true", help="Training-error-driven group expansion and depth-10, fixed-500 specialist variant.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    destination = args.output_dir or args.run_dir / ("specialist_expanded_v2" if args.expanded_fixed else "specialist_v1")
    print(run_specialist_study(args.run_dir, destination, seed=args.seed, expanded_fixed=args.expanded_fixed)["status"])
