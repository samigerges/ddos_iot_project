"""Freeze both multiclass choices together, then execute the frozen final protocol."""
import argparse
import logging
from pathlib import Path

from ddos_iot.experiments.study_protocol import freeze_research_protocol, evaluate_research_protocol, select_under_recall_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    select = actions.add_parser("select")
    select.add_argument("--evidence", type=Path, required=True)
    select.add_argument("--candidate", required=True)
    select.add_argument("--maximum-recall-loss", type=float, default=0., help="Fraction: 0.01 is one percentage point.")
    select.add_argument("--scope", choices=("each_seed", "mean"), default="each_seed")
    select.add_argument("--policy-note", required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    freeze = actions.add_parser("freeze")
    freeze.add_argument("--8class-confirmation", dest="eight", type=Path, required=True)
    freeze.add_argument("--34class-confirmation", dest="thirty_four", type=Path, required=True)
    freeze.add_argument("--development-report", type=Path, required=True)
    freeze.add_argument("--recall-policy", required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--archive", type=Path, default=Path("data/raw/paper_v3/datasets.tar.xz"))
    freeze.add_argument("--record", type=Path, default=Path("outputs/baseline_audit_v1/sources/zenodo_v3_record.json"))
    freeze.add_argument("--paper-python", type=Path, default=Path(".cache/paper_runtime3112/python.exe"))
    evaluate = actions.add_parser("evaluate")
    evaluate.add_argument("--protocol-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "select":
        select_under_recall_policy(args.evidence, args.candidate, args.output_dir,
            maximum_recall_loss=args.maximum_recall_loss, scope=args.scope, policy_note=args.policy_note)
        print("selected under declared policy")
    elif args.action == "freeze":
        freeze_research_protocol({8: args.eight, 34: args.thirty_four}, args.output_dir,
            archive=args.archive, record=args.record, recall_policy=args.recall_policy,
            paper_python=args.paper_python, development_report=args.development_report)
        print("frozen")
    else:
        print(evaluate_research_protocol(args.protocol_dir)["status"])
