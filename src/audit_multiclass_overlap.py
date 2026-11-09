"""Audit saved development/validation flow overlap by class, using training data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ddos_iot.experiments.overlap_audit import audit_overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_overlap(args.run_dir, args.seed, output_path=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
