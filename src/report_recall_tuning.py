"""Summarize completed recall-constrained tuning without loading models or data."""

import argparse
from pathlib import Path

from ddos_iot.experiments.constrained_reporting import write_constrained_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(write_constrained_report(parser.parse_args().run_dir))
