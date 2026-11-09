"""Compare completed research stages without loading data or trained models."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from ddos_iot.attack_families import FAMILY_MEMBERS, RAW_TO_FAMILY
from .storage import write_report, write_table


def _class_report(root, row, phase):
    if row.get("artifact_folder"):
        return Path(row["artifact_folder"]) / phase / "classification_report.csv"
    candidates = (root / f"seed_{row['seed']}/finalists" / row["candidate"] / phase,
                  root / row["candidate"] / phase,
                  root / f"seed_{row['seed']}" / row["candidate"] / phase)
    return next((p / "classification_report.csv" for p in candidates if (p / "classification_report.csv").exists()), None)


def target_error_summary(confusions, task):
    targets = {"WebBased", "Recon", "BruteForce"} if task == 8 else {
        raw for family in ("WebBased", "Recon", "BruteForce") for raw in FAMILY_MEMBERS[family]}
    selected = confusions.loc[confusions.true_label.isin(targets)].copy()
    total = int(selected["count"].sum())
    benign = "Benign" if task == 8 else "BenignTraffic"
    to_benign = int(selected.loc[selected.predicted_label == benign, "count"].sum())
    within = 0 if task == 8 else int(selected.loc[
        selected.true_label.map(RAW_TO_FAMILY) == selected.predicted_label.map(RAW_TO_FAMILY), "count"].sum())
    return {"target_errors": total, "to_benign": to_benign, "within_family": within,
            "other_family": total - to_benign - within,
            "to_benign_fraction": to_benign / total if total else 0.,
            "within_family_fraction": within / total if total else 0.,
            "other_family_fraction": (total - to_benign - within) / total if total else 0.}, selected


def report_research(roots, output, *, plots=True):
    rows, classes, stages, error_summaries, error_pairs = [], [], [], [], []
    timing_exclusions = []
    analyzed_references = set()
    seen_tasks = set()
    for root in map(Path, roots):
        base = json.loads((root / "report.json").read_text(encoding="utf-8"))
        task = base["num_classes"]
        if task in seen_tasks:
            raise ValueError("Compare one source study per task; different studies must not share a reference by seed.")
        seen_tasks.add(task)
        paths = [root / "report.json", *sorted(root.glob("*/report.json"))]
        for path in paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            stage = "base" if path.parent == root else path.parent.name
            stages.append({"classes": task, "stage": stage, "status": report["status"],
                           "reason": report.get("reason", ""), "source": str(path.resolve())})
            if report["status"] != "complete":
                continue
            phase = "test" if report.get("test_csv_opened") else "validation"
            audit_path = path.parent / "timing_audit.json"
            invalid_timings = {}
            if phase == "test" and audit_path.exists():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit["status"] != "complete":
                    raise ValueError("Finish the timing audit before reporting its exclusions.")
                invalid_timings = {(r["seed"], r["candidate"]): r for r in audit.get("invalid_timed_passes", [])}
            # Base 'results' are tuning diagnostics and must not enter this table.
            observed = report.get("validation_results", [] if path.parent == root else report.get("results", []))
            for row in observed:
                if "macro_f1" not in row or "seed" not in row:
                    continue
                timing = row.get("inference_us_per_sample")
                timing_scope = "complete bundle" if timing is not None else "sum of separate components"
                if timing is None and row.get("shared_probability_inference_us") is not None:
                    timing = row["shared_probability_inference_us"] + row.get("decision_us_per_sample", 0.)
                identity = {"classes": task, "phase": phase, "stage": stage,
                            "candidate": row["candidate"], "seed": row["seed"]}
                recorded_timing = timing
                invalid = invalid_timings.get((row["seed"], row["candidate"]))
                if invalid:
                    if timing is None or abs(timing - invalid["original_inference_us_per_sample"]) > 1e-6:
                        raise ValueError("Timing audit does not match the recorded pass.")
                    timing_exclusions.append({**identity, "recorded_inference_us_per_sample": timing,
                        "reason": invalid["reason"], "audit": str(audit_path.resolve())})
                    timing, timing_scope = None, "Excluded sleep-interrupted timed pass; original classification scores and timing preserved."
                rows.append({**identity, **{k: row.get(k) for k in ("macro_f1", "weighted_f1", "accuracy",
                    "benign_false_positive_rate", "training_seconds", "reference_constraints_pass")},
                    "inference_us_per_sample": timing, "recorded_inference_us_per_sample": recorded_timing,
                    "timing_scope": timing_scope if invalid else row.get("timing_note", timing_scope),
                    "source": str(path.resolve())})
                per_class = _class_report(path.parent, row, phase)
                if per_class is not None and per_class.exists():
                    for label, values in pd.read_csv(per_class, index_col=0).iterrows():
                        if label in ("accuracy", "macro avg", "weighted avg"):
                            continue
                        classes.append({**identity, "class": label,
                            **{k: float(values[k]) for k in ("precision", "recall", "f1-score", "support")}})
                    reference_key = (task, phase, row["seed"])
                    confusion_path = per_class.with_name("confusions.csv")
                    if row["candidate"] == "paper_fixed" and reference_key not in analyzed_references and confusion_path.exists():
                        statistics, pairs = target_error_summary(pd.read_csv(confusion_path), task)
                        error_summaries.append({**identity, **statistics})
                        error_pairs.extend({**identity, **pair} for pair in pairs.to_dict(orient="records"))
                        analyzed_references.add(reference_key)
    frame, per_class = pd.DataFrame(rows), pd.DataFrame(classes)
    if not per_class.empty:
        class_references = {}
        for _, row in per_class.loc[per_class.candidate == "paper_fixed"].iterrows():
            key = (row["classes"], row.phase, row.seed, row["class"])
            metrics = (row.recall, row["f1-score"])
            if key in class_references and any(abs(a - b) > 1e-12 for a, b in zip(metrics, class_references[key])):
                raise ValueError("Reference class metrics disagree for the same task/phase/seed/label.")
            class_references[key] = metrics
        for metric, position in (("recall", 0), ("f1-score", 1)):
            name = "recall_delta_vs_reference" if position == 0 else "f1_delta_vs_reference"
            per_class[name] = [r[metric] - class_references[key][position] if key in class_references else None
                for _, r in per_class.iterrows() for key in [(r["classes"], r.phase, r.seed, r["class"])]]
    if not frame.empty:
        references = {}
        for _, row in frame.loc[frame.candidate == "paper_fixed"].iterrows():
            key = (row["classes"], row.phase, row.seed)
            metrics = (row.macro_f1, row.benign_false_positive_rate)
            if key in references and any(abs(a - b) > 1e-12 for a, b in zip(metrics, references[key])):
                raise ValueError("Reference metrics disagree for the same task/phase/seed.")
            references[key] = metrics
        frame["macro_f1_delta_vs_reference"] = [r.macro_f1 - references[key][0] if key in references else None
            for _, r in frame.iterrows() for key in [(r["classes"], r.phase, r.seed)]]
        frame["benign_fpr_delta_vs_reference"] = [r.benign_false_positive_rate - references[key][1] if key in references else None
            for _, r in frame.iterrows() for key in [(r["classes"], r.phase, r.seed)]]
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "stages.csv", pd.DataFrame(stages))
    write_table(output / "comparison.csv", frame)
    write_table(output / "timing_exclusions.csv", pd.DataFrame(timing_exclusions,
        columns=["classes", "phase", "stage", "candidate", "seed", "recorded_inference_us_per_sample", "reason", "audit"]))
    write_table(output / "per_class.csv", per_class)
    write_table(output / "reference_error_summary.csv", pd.DataFrame(error_summaries))
    write_table(output / "reference_minority_error_destinations.csv", pd.DataFrame(error_pairs))
    summaries = []
    if not frame.empty:
        keys = ["classes", "phase", "stage", "candidate"]
        for key, group in frame.groupby(keys, sort=False):
            if group.seed.duplicated().any():
                raise ValueError("Duplicate candidate/seed results within one stage.")
            item = dict(zip(keys, key))
            item["seeds"] = group.seed.nunique()
            item["valid_timing_passes"] = int(group.inference_us_per_sample.notna().sum())
            item["guard_passes"] = int(group.reference_constraints_pass.fillna(False).sum())
            item["guard_evaluated"] = int(group.reference_constraints_pass.notna().sum())
            for metric in ("macro_f1", "weighted_f1", "benign_false_positive_rate", "inference_us_per_sample", "training_seconds",
                           "macro_f1_delta_vs_reference", "benign_fpr_delta_vs_reference"):
                value = group[metric].dropna()
                item[metric + "_mean"] = float(value.mean()) if len(value) else None
                item[metric + "_std"] = float(value.std(ddof=0)) if len(value) else None
            summaries.append(item)
    summary = pd.DataFrame(summaries)
    write_table(output / "summary.csv", summary)
    if not per_class.empty:
        grouped = per_class.groupby(["classes", "phase", "stage", "candidate", "class"], sort=False)
        class_summary = grouped.agg(f1_mean=("f1-score", "mean"), f1_std=("f1-score", lambda s: s.std(ddof=0)),
            recall_mean=("recall", "mean"), recall_std=("recall", lambda s: s.std(ddof=0)),
            f1_delta_mean=("f1_delta_vs_reference", "mean"), f1_delta_std=("f1_delta_vs_reference", lambda s: s.std(ddof=0)),
            recall_delta_mean=("recall_delta_vs_reference", "mean"), recall_delta_std=("recall_delta_vs_reference", lambda s: s.std(ddof=0)),
            precision_mean=("precision", "mean"), seeds=("seed", "nunique")).reset_index()
        write_table(output / "per_class_summary.csv", class_summary)
        protected = {"WebBased", "Recon", "BruteForce", *(raw for family in ("WebBased", "Recon", "BruteForce") for raw in FAMILY_MEMBERS[family])}
        write_table(output / "minority_class_summary.csv", class_summary.loc[class_summary["class"].isin(protected)])
    lines = ["# Multiclass research comparisons", "", f"Generated {datetime.now(timezone.utc).isoformat()}.", "",
        "Completed stages only. Internal validation is development evidence, not the paper's official-test result. "
        "Repeated splits share the published training source; upstream feature selection is not fold-local.", "",
        "| Classes | Phase | Stage / candidate | Seeds | Macro-F1 mean ± std | Paired Δ Macro-F1 | Weighted-F1 | Benign FPR | CPU us/flow | Guards passed |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        timing = row["inference_us_per_sample_mean"]
        latency = f"{timing:.2f} (n={row['valid_timing_passes']})" if timing is not None else "unmeasured"
        guard = f"{row['guard_passes']}/{row['guard_evaluated']}" if row["guard_evaluated"] else "unmeasured"
        delta_mean, delta_std = row["macro_f1_delta_vs_reference_mean"], row["macro_f1_delta_vs_reference_std"]
        delta = f"{delta_mean:+.5f} ± {delta_std:.5f}" if delta_mean is not None else "unpaired"
        lines.append(f"| {row['classes']} | {row['phase']} | {row['stage']} / {row['candidate']} | {row['seeds']} | "
            f"{row['macro_f1_mean']:.5f} ± {row['macro_f1_std']:.5f} | {delta} | {row['weighted_f1_mean']:.5f} | "
            f"{100 * row['benign_false_positive_rate_mean']:.2f}% | {latency} | {guard} |")
    lines += ["", "Guard checks are provisional reference-recall/FPR comparisons, not final user-selected recall limits. "
        "Comparison-only runs cannot promote a model. A one-seed std of zero supplies no evidence of stability. "
        "Deltas compare the same seed/validation rows with its fixed reference; their std describes paired split variation, not a confidence interval.", "",
        "CPU values are batched throughput on selected/scaled inputs. Some bias/stack rows add separately measured "
        "components; see comparison.csv for timing scope. Training costs in summary.csv include each stage's declared "
        "fit scope; shared OOF work must not be summed again across stages.", "",
        "Sleep-interrupted timed passes are excluded from throughput means using explicit timing audits; their classification scores remain included. "
        "The original reported timings are retained in comparison.csv. [Timing exclusions](timing_exclusions.csv).", "",
        "[All per-class metrics](per_class_summary.csv) · [Target minority classes](minority_class_summary.csv) · "
        "[Individual results](comparison.csv) · [Stage status](stages.csv)", ""]
    if error_summaries:
        lines += ["## Where reference minority-class mistakes go", "",
            "Percentages below are fractions of target-class mistakes, not fractions of all traffic. "
            "Within-family errors apply to the 34-label task. These measurements do not establish an information ceiling.", "",
            "| Classes | Phase | Seeds | To Benign | Within family | Other family |",
            "|---:|---|---:|---:|---:|---:|"]
        for (task, phase), group in pd.DataFrame(error_summaries).groupby(["classes", "phase"], sort=False):
            lines.append(f"| {task} | {phase} | {group.seed.nunique()} | {100 * group.to_benign_fraction.mean():.1f}% | "
                f"{100 * group.within_family_fraction.mean():.1f}% | {100 * group.other_family_fraction.mean():.1f}% |")
        lines += ["", "[Counts by seed](reference_error_summary.csv) · [Actual target-class confusion pairs](reference_minority_error_destinations.csv)", ""]
    if plots and not summary.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for task, group in summary.loc[summary.phase == "validation"].groupby("classes", sort=False):
            fig, ax = plt.subplots(figsize=(10, 6))
            for _, row in group.iterrows():
                label = f"{row.stage}: {row.candidate} (n={row.seeds})"
                ax.errorbar(100 * row.benign_false_positive_rate_mean, row.macro_f1_mean,
                    xerr=100 * row.benign_false_positive_rate_std, yerr=row.macro_f1_std,
                    fmt="o", capsize=3, label=label)
            ax.set(xlabel="Benign false-positive rate (%)", ylabel="Macro-F1",
                   title=f"{task} classes — development validation trade-offs")
            ax.grid(alpha=.25)
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            name = f"{task}class_tradeoffs.png"
            fig.savefig(output / name, dpi=180)
            plt.close(fig)
            lines += [f"![{task}-class validation trade-offs]({name})", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_report(output / "report.json", {"status": "complete", "metric_rows": len(rows), "test_used_for_selection": False,
        "source_stages": stages, "scope": "Read-only comparison of saved reports; no datasets/models loaded, no selection or tuning."})
    return output / "REPORT.md"
