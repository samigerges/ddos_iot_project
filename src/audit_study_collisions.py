"""Audit exact cross-label feature collisions using training data only."""

import argparse
from pathlib import Path

from ddos_iot.experiments.study_collisions import audit_training_collisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(audit_training_collisions(args.run_dir, args.output_dir or args.run_dir / "collision_audit")["exact_conflicting_groups"])
