"""Render completed study metrics without loading models or datasets."""

import argparse
from pathlib import Path

from ddos_iot.experiments.study_reporting import report_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(report_study(args.run_dir, allow_partial=args.allow_partial))
