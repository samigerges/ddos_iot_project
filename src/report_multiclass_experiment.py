"""Render saved validation metrics and trade-off plots without loading models."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddos_iot.experiments.reporting import create_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(create_validation_report(args.run_dir, allow_partial=args.allow_partial))


if __name__ == "__main__":
    main()
