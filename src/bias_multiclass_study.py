"""Build the OOF-tuned single-model baseline on the published training study."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_bias import run_single_bias_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run_single_bias_study(args.run_dir, args.output_dir or args.run_dir / "bias_v1", seed=args.seed)["status"])
