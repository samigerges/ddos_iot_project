"""Plan/run controlled multiclass ablations, or evaluate their frozen winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ddos_iot.experiments.config import experiment_plan, load_config
from ddos_iot.experiments.runner import evaluate_frozen_test, run_development
from ddos_iot.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
    final = commands.add_parser("evaluate-test")
    final.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_logging(False)
    if args.command == "evaluate-test":
        result = evaluate_frozen_test(args.run_dir)
    else:
        config = load_config(args.config)
        result = experiment_plan(config) if args.command == "plan" else run_development(config)
    if args.command == "plan":
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({key: result[key] for key in ("status", "selected_candidate", "selected_decision", "seconds")
                          if key in result}, indent=2))
        if result["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
