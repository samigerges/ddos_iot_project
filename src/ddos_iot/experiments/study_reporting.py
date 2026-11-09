"""Read-only summaries of staged studies, confirmations and conditional models."""

import json
from pathlib import Path

import pandas as pd

from .storage import write_table


def report_study(root: Path, *, allow_partial=False):
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["status"] not in ("complete", "skipped", "partial_failure") and not allow_partial:
        raise ValueError("Wait for study completion or explicitly request a provisional report.")
    destination = root / ("provisional_report" if report["status"] not in ("complete", "skipped", "partial_failure") else "report")
    destination.mkdir(exist_ok=True)
    lines = ["# Multiclass training study", "", f"Status: **{report['status']}**.", ""]
    if report["status"] == "skipped":
        lines += [report["reason"], "", "This stage did not advance. The reason, evidence and any preserved partial artifacts are recorded in the adjacent JSON.", ""]
    else:
        lines += [report.get("protocol", report.get("split_scope", "Training-only choices are frozen before evaluation.")), "",
                  report.get("upstream_feature_selection", ""), ""]
    if "results" in report and report.get("selections"):
        rows = report["results"]
        lines += ["## Training-only screening", "", "| Seed | Candidate | Macro-F1 | Rounds | Recall/FPR policy | Training s |",
                  "|---:|---|---:|---:|---|---:|"]
        for row in rows:
            lines.append(f"| {row['seed']} | {row['candidate']} | {row['macro_f1']:.5f} | {row['rounds']} | {row['reference_constraints_pass']} | {row['training_seconds']:.1f} |")
        lines += ["", "These are candidate/round-selection diagnostics, not independent performance estimates.", ""]
    rows = report.get("validation_results", report.get("results", []))
    per_class = []
    if rows:
        phase = "Test" if report.get("test_csv_opened") else "Validation"
        lines += [f"## {phase} results", "", "| Seed | Candidate | Macro-F1 | Weighted-F1 | Benign FPR | CPU us/flow |",
                  "|---:|---|---:|---:|---:|---:|"]
        for row in rows:
            timing = row.get("inference_us_per_sample")
            if timing is None: timing = row.get("shared_probability_inference_us")
            time_text = f"{timing:.2f}" if timing is not None else "not measured"
            lines.append(f"| {row['seed']} | {row['candidate']} | {row['macro_f1']:.5f} | {row['weighted_f1']:.5f} | {100 * row['benign_false_positive_rate']:.2f}% | {time_text} |")
            folder = None
            if "artifact_folder" in row:
                folder = Path(row["artifact_folder"]) / "validation"
            else:
                for possible in (root / f"seed_{row['seed']}/finalists" / row["candidate"] / "validation",
                                 root / row["candidate"] / "validation",
                                 root / f"seed_{row['seed']}" / row["candidate"] / "test"):
                    if (possible / "classification_report.csv").exists():
                        folder = possible; break
            if folder is not None:
                classes = pd.read_csv(folder / "classification_report.csv", index_col=0)
                for label, values in classes.iterrows():
                    if label in ("accuracy", "macro avg", "weighted avg"): continue
                    per_class.append({"seed": row["seed"], "candidate": row["candidate"], "class": label,
                                      **{name: float(values[name]) for name in ("precision", "recall", "f1-score", "support")}})
        write_table(destination / "per_class.csv", pd.DataFrame(per_class))
        if per_class:
            grouped = pd.DataFrame(per_class).groupby(["candidate", "class"], sort=False)
            summary = grouped.agg(precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                                  f1_mean=("f1-score", "mean"), f1_std=("f1-score", lambda s: s.std(ddof=0)),
                                  recall_std=("recall", lambda s: s.std(ddof=0)), seeds=("seed", "nunique")).reset_index()
            write_table(destination / "per_class_summary.csv", summary)
        lines += ["", "[All class precision/recall/F1/support](per_class.csv). Timing is batched CPU throughput on already selected/scaled features, not raw-flow feature extraction or single-flow latency. Stack probability and decision timings are separate measurements.", ""]
    if "selections" in report:
        lines += ["## Frozen choices", ""]
        for seed, selection in report["selections"].items():
            lines.append(f"- Seed {seed}: unrestricted `{selection['best_macro']}`; recall/FPR guarded `{selection['guarded_macro']}`.")
        lines += ["", "Choices use training-only tuning scores. Reference and finalists are refitted to development rows, then evaluated on the reserved validation split. Published features were selected upstream on the full author training split, so this is not fold-local feature-selection validation.", ""]
    if "selected_candidate" in report:
        lines += [f"Frozen confirmed candidate: `{report['selected_candidate']}`.", ""]
    if "paper_reference_evaluation" in report:
        lines += [report["paper_reference_evaluation"], ""]
    lines += ["## Scope", "",
              "Model, split, hyperparameter, source-hash and selection details are in the adjacent JSON files. Additional complexity is judged on full-system performance including routing mistakes. Repeated split seeds do not represent independent captures/devices. Published-test reproduction and raw-data generalization are different questions.", "",
              "No final test is a source of threshold, hyperparameter, model or seed selection. Final test execution is a separate once-only command after multiseed selection is frozen.", ""]
    path = destination / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
