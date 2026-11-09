"""Compare a completed local run against downloaded, pinned author source."""

import argparse
from pathlib import Path

from ddos_iot.experiments.baseline_audit import audit_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path("outputs/baseline_audit_v1/sources"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_audit_v1"))
    args = parser.parse_args()
    report = audit_baseline(args.run_dir, args.sources, args.output_dir)
    print(report["conclusion"])


if __name__ == "__main__":
    main()
