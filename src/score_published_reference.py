"""Score a predeclared pinned-software reference once, after final model freeze."""

import argparse
from pathlib import Path

from ddos_iot.experiments.paper_reference import score_paper_reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(score_paper_reference(args.reference, args.test_csv, args.protocol, args.output_dir)["macro_f1"])
