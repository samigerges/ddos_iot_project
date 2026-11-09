"""Train the paper reference using its isolated pinned software versions."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.paper_reference import train_paper_reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, choices=[8, 34], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(train_paper_reference(args.classes, args.output_dir)["status"])
