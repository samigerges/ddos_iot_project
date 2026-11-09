"""Retune cached OOF probabilities with reference recall/FPR constraints."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.constrained import export_constrained_choice, run_constrained_tuning
from ddos_iot.experiments.constrained_reporting import write_constrained_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protect", nargs="+", default=["WebBased", "BruteForce"],
                        help="Exact class names whose reference OOF recall must be preserved.")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--export-selected", action="store_true", help="Save a reusable biased development bundle; no retraining.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = args.output_dir or args.run_dir / "recall_constraints_v1"
    report = run_constrained_tuning(args.run_dir, output, protected_classes=tuple(args.protect), passes=args.passes)
    write_constrained_report(output)
    if args.export_selected:
        export_constrained_choice(output)
    print(f"Completed: {report['selected_candidate']}; validation constraints pass: "
          f"{report['selected_validation_constraints_pass']}; test remains unread.")


if __name__ == "__main__":
    main()
