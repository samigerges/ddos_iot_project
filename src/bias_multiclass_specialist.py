"""Tune frozen specialist decisions using routed out-of-fold predictions."""
import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_specialist_bias import calibrate_specialist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--specialist-run", type=Path, required=True)
    parser.add_argument("--raw-confirmation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(calibrate_specialist(args.run_dir, args.specialist_run, args.output_dir or args.run_dir / "specialist_bias_v1",
                               raw_confirmation=args.raw_confirmation)["status"])
