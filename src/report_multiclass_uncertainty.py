"""Analyze paired frozen-test predictions without fitting or selecting models."""
import argparse
from pathlib import Path

from ddos_iot.experiments.study_uncertainty import report_test_uncertainty

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(report_test_uncertainty(args.reference, args.candidate, args.output_dir)["macro_f1"])
