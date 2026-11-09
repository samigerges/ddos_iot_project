"""Run the staged research pipeline sequentially; final tests require an explicit flag."""

import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.research_pipeline import run_research_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=[Path("configs/studies/8class.json"), Path("configs/studies/34class.json")])
    parser.add_argument("--state-dir", type=Path, default=Path("outputs/studies/research_pipeline_v1"))
    parser.add_argument("--paper-python", type=Path)
    parser.add_argument("--wait-existing", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run_research_pipeline(args.configs, args.state_dir, paper_python=args.paper_python,
                               wait_existing=args.wait_existing, final_test=args.final_test)["status"])
