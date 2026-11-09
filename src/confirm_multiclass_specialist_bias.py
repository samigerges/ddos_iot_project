"""Confirm all frozen hierarchical decisions using shared base predictions."""
import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_specialist_bias import confirm_specialist_decisions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-confirmation", type=Path, required=True)
    parser.add_argument("--raw-confirmation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(confirm_specialist_decisions(args.base_confirmation, args.raw_confirmation, args.calibration, args.output_dir)["status"])
