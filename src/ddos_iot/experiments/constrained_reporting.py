"""Compact evidence report for OOF-selected recall-constrained operating points."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .storage import write_table


def write_constrained_report(output: Path) -> Path:
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError("Wait for constrained tuning and validation to complete.")
    source = Path(report["source_run"])
    original = json.loads((source / "experiment_report.json").read_text(encoding="utf-8"))
    records, class_records = [], []
    for stage, rows in [("original", original["results"]), ("constrained", report["validation_results"])]:
        for row in rows:
            metrics_folder = ((source / f"seed_{row['seed']}" / row["candidate"] / "validation" / row["decision"])
                              if stage == "original" else output / f"seed_{row['seed']}" / row["candidate"] / "validation")
            classes = pd.read_csv(metrics_folder / "classification_report.csv", index_col=0)
            for label in report["classes"]:
                class_records.append({"stage": stage, "candidate": row["candidate"], "decision": row["decision"],
                                      "seed": row["seed"], "class": label,
                                      **{metric: float(classes.loc[label, metric]) for metric in
                                         ("precision", "recall", "f1-score", "support")}})
            record = {key: row[key] for key in ("candidate", "decision", "seed", "macro_f1", "weighted_f1",
                                               "accuracy", "benign_false_positive_rate")}
            record.update(stage=stage, selected_by_constrained_oof=bool(row.get("selected_by_oof", False)))
            for name in report["protected_classes"]:
                for metric in ("precision", "recall", "f1-score", "support"):
                    record[f"{name}_{metric}"] = float(classes.loc[name, metric])
            records.append(record)
    frame = pd.DataFrame(records)
    write_table(output / "operating_points.csv", frame)
    write_table(output / "per_class_comparison.csv", pd.DataFrame(class_records))
    reference = frame.loc[(frame.stage == "original") & (frame.candidate == report["reference_candidate"]) & (frame.decision == "argmax")]
    chosen = frame.loc[frame.selected_by_constrained_oof]
    seed_count = len(chosen)
    delta_f1 = chosen.macro_f1.mean() - reference.macro_f1.mean()
    delta_fpr = chosen.benign_false_positive_rate.mean() - reference.benign_false_positive_rate.mean()
    rows = [("Reference argmax", reference),
            ("Previous unconstrained F1 leader", frame.loc[(frame.stage == "original") &
                 (frame.candidate == original["selected_candidate"]) & (frame.decision == original["selected_decision"])]),
            ("Recall-constrained OOF choice", chosen)]
    lines = ["# Recall-constrained eight-class comparison" if len(report["classes"]) == 8 else "# Recall-constrained multiclass comparison", "",
             f"Completed {seed_count} seed(s) without model retraining or test evaluation. "
             f"The OOF-selected candidate is `{report['selected_candidate']}` with newly tuned biases.", "",
             f"Compared with reference argmax, mean validation Macro-F1 changes by **{delta_f1:+.6f}** "
             f"and Benign false-positive rate by **{100 * delta_fpr:+.3f} percentage points**.", "",
             "## Operating points", "",
             "| Condition | Macro-F1 | Benign FPR | " + " | ".join(f"{c} recall" for c in report["protected_classes"]) + " |",
             "|---|---:|---:|" + "---:|" * len(report["protected_classes"])]
    for name, group in rows:
        lines.append(f"| {name} | {group.macro_f1.mean():.4f} | {100 * group.benign_false_positive_rate.mean():.2f}% | " +
                     " | ".join(f"{100 * group[f'{c}_recall'].mean():.2f}%" for c in report["protected_classes"]) + " |")
    lines += ["", "## Constraints and selection", "",
              "Each seed's minimum attack recalls and maximum Benign FPR were derived from the reference's **OOF argmax predictions**, never from validation or test labels. The same limits were applied to all candidates. The search tries zero bias and the previous OOF-tuned bias, using a fixed grid and three coordinate passes by default. Feasibility takes priority over Macro-F1; once feasible, a path cannot accept a violating step.", "",
              "The best mean feasible OOF Macro-F1 selected the candidate. All candidate biases and that selection were saved and hashed before validation features or labels were loaded. OOF optimization scores are not independent performance estimates. This is a finite heuristic search, not proof of global optimality or infeasibility.", ""]
    for seed, constraints in report["constraints_by_seed"].items():
        floors = ", ".join(f"{report['classes'][int(c)]} recall >= {100 * v:.4f}%" for c, v in constraints["minimum_recall"].items())
        lines.append(f"- Seed {seed}: {floors}; Benign FPR <= {100 * constraints['max_benign_fpr']:.4f}% on OOF.")
    unavailable = [f"{r['candidate']} (seed {r['seed']})" for r in report["oof_results"] if not r["feasible"]]
    gain_description = (
        "This search did not show a clear recall-preserving validation Macro-F1 gain. Treat false-alarm changes separately from F1 improvements."
        if delta_f1 <= .001 else
        "A validation Macro-F1 gain was observed. It still requires the declared recall/FPR checks and confirmation across seeds.")
    lines += ["", f"The selected candidate preserves the reference's corresponding validation recalls and FPR in every seed: **{report['selected_validation_constraints_pass']}**. OOF feasibility alone does not guarantee validation feasibility; individual candidate results record this distinction.", "",
              ("No feasible grid-search point was found for: " + ", ".join(unavailable) + ".") if unavailable else "All candidates had a feasible OOF point.", "",
              "## Limits and next experiment", "",
              "The existing validation cohort was used in earlier ablation comparisons; this is not a fresh final test. One seed cannot establish a reproducible gain. All reported protected-class recalls/F1 and all candidate operating points are retained, including rejected choices.", "",
              gain_description + " Resolve paper baseline protocol mismatches before investing in more complex models. Confirm the resulting finalists across multiple seeds. A candidate that fails validation requirements is not accepted as a final model.", "",
              "Inference still uses the same base classifier and a bias vector. No new latency benchmark was run for this operating point; earlier timing of a different selected base model must not be reused as its measured latency.", "",
              "## Artifacts", "",
              "- [All original and constrained validation operating points](operating_points.csv)",
              "- [Every class precision, recall, F1, and support](per_class_comparison.csv)",
              "- [OOF feasibility and tuning scores](oof_comparison.csv)",
              "- [Constrained validation results](validation_comparison.csv)",
              "- [Frozen OOF selection](selection.json)",
              "- [Full diagnostics and scope](report.json)", ""]
    if (output / "verification_report.json").exists():
        lines += ["- [Independent artifact verification](verification_report.json)", ""]
    if (output / "benchmark_models").exists():
        lines += ["Research bundles in `benchmark_models/` apply the saved biases automatically. Their metadata records validation-policy status; these are not full-source final models.", ""]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
