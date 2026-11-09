"""Extract and inventory the verified paper training subsets; leave tests unparsed."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.reference_data import prepare_reference_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/paper_v3/datasets.tar.xz"))
    parser.add_argument("--record", type=Path, default=Path("outputs/baseline_audit_v1/sources/zenodo_v3_record.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/paper_v3/training_subsets"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = prepare_reference_training(args.archive, args.record, args.output_dir)
    for name, entry in result["training_subsets"].items():
        print(f"{name}: {entry['rows']:,} rows, {entry['feature_count']} features")
