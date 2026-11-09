"""Plan or run staged XGBoost/CatBoost studies without final-test access."""

import argparse
import json
import logging
from pathlib import Path

from ddos_iot.experiments.study import run_training_study
from ddos_iot.experiments.study_config import load_study_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "plan":
        print(json.dumps({"config": config.as_dict(), "screening_trials_per_seed": len(config.capacity) + len(config.weight_powers) + int(config.catboost),
                          "final_test_read": False, "source_features": 23,
                          "note": "Published preselected features; development validation is not the paper test score."}, indent=2))
    else:
        report = run_training_study(config)
        print(f"Study {report['status']}; test remains unread; output: {config.output_path}")
