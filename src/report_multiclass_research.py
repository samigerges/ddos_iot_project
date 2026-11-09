"""Build a read-only comparison of completed multiclass research stages."""

import argparse
from pathlib import Path

from ddos_iot.experiments.research_reporting import report_research


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=[Path("outputs/studies/8class_v1"), Path("outputs/studies/34class_v1")])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/studies/comparison_v1"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(report_research(args.run_dirs, args.output_dir, plots=not args.no_plots))
