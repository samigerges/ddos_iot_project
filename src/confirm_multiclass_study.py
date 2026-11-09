"""Confirm frozen training-study finalists across three or more seeds."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_confirmation import confirm_training_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = confirm_training_study(args.run_dir, args.output_dir or args.run_dir / "confirmation_v1", seeds=tuple(args.seeds))
    print(f"Confirmation {report['status']}: {report['selected_candidate']}; test remains unread.")
