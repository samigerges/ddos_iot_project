"""Explicit once-only published-test stage after frozen multiseed confirmation."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_final import evaluate_study_test
from ddos_iot.experiments.study_advanced_final import evaluate_advanced_test
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/paper_v3/datasets.tar.xz"))
    parser.add_argument("--record", type=Path, default=Path("outputs/baseline_audit_v1/sources/zenodo_v3_record.json"))
    parser.add_argument("--test-directory", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--paper-reference", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    snapshot = json.loads((args.confirmation / "selection.json").read_text(encoding="utf-8"))
    evaluate = evaluate_advanced_test if "advanced_confirmed" in snapshot else evaluate_study_test
    print(evaluate(args.confirmation, args.output_dir or args.confirmation / "test_evaluation",
                   archive=args.archive, record_path=args.record, test_directory=args.test_directory,
                   latency_repeats=args.latency_repeats, paper_reference=args.paper_reference)["status"])
