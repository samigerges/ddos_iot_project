"""Build an OOF stack only after diversity and latency evidence pass."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_stacking import run_stacking_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta-max-iter", type=int, default=2000)
    parser.add_argument("--meta-solver", choices=("lbfgs", "newton-cg"), default="lbfgs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run_stacking_study(args.run_dir, args.output_dir or args.run_dir / "stack_v1", seed=args.seed,
                              meta_max_iter=args.meta_max_iter, meta_solver=args.meta_solver)["status"])
