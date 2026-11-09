"""Confirm a promising protected specialist/stack on the remaining split seeds."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_advanced_confirmation import confirm_advanced_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--advanced-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--compare-only", action="store_true", help="Measure a frozen candidate across seeds without promoting it or imposing final recall limits.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(confirm_advanced_study(args.confirmation, args.advanced_run, args.output_dir,
                                candidate_override=args.candidate, compare_only=args.compare_only)["status"])
