"""Read-only validation comparisons and standalone trade-off figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ddos_iot.paths import configure_runtime_environment
from .config import config_from_dict
from .selection import analyze_selection
from .storage import write_report, write_table


def _class_metrics(run_path: Path, results: list[dict], classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for result in results:
        path = (run_path / f"seed_{result['seed']}" / result["candidate"] /
                "validation" / result["decision"] / "classification_report.csv")
        table = pd.read_csv(path).set_index("class")
        for label in classes:
            values = table.loc[label]
            rows.append({"candidate": result["candidate"], "decision": result["decision"],
                         "seed": result["seed"], "label": label, "precision": values.precision,
                         "recall": values.recall, "f1_score": values["f1-score"], "support": int(values.support)})
    frame = pd.DataFrame(rows)
    summary = []
    for (candidate, decision, label), group in frame.groupby(["candidate", "decision", "label"], sort=False):
        row = {"candidate": candidate, "decision": decision, "label": label,
               "seed_count": group.seed.nunique(), "support_min": int(group.support.min()),
               "support_max": int(group.support.max())}
        for metric in ("precision", "recall", "f1_score"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        summary.append(row)
    return frame, pd.DataFrame(summary)


def create_validation_report(run_path: Path, *, allow_partial: bool = False) -> Path:
    """Read saved metrics only: no model loading, data access, or training."""
    report = json.loads((run_path / "experiment_report.json").read_text(encoding="utf-8"))
    complete = report["status"] in ("complete", "partial_failure")
    if not complete and not allow_partial:
        raise ValueError("The run is incomplete. Use --allow-partial to label a provisional snapshot explicitly.")
    if not report["results"]:
        raise ValueError("No completed validation results are available yet.")
    payload = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    # Historical runs used score-only selection; do not reinterpret their policy.
    payload.setdefault("selection_policy", "macro_f1")
    config = config_from_dict(payload)
    analysis = analyze_selection(report["results"], config)
    summary = analysis.summary
    destination = run_path / ("report" if complete else "provisional_report")
    destination.mkdir(exist_ok=True)
    write_table(destination / "validation_comparison.csv", analysis.comparisons)
    write_table(destination / "validation_summary.csv", summary)
    write_report(destination / "selection_comparison.json", analysis.report)
    mapping = json.loads((run_path / "label_mapping.json").read_text(encoding="utf-8"))
    classes = [mapping[str(i)] for i in range(config.num_classes)]
    per_class, per_class_summary = _class_metrics(run_path, report["results"], classes)
    write_table(destination / "per_class_validation.csv", per_class)
    write_table(destination / "per_class_summary.csv", per_class_summary)
    costs, seen = [], set()
    for result in report["results"]:
        key = result["candidate"], result["seed"]
        if key in seen:
            continue
        seen.add(key)
        folder = run_path / f"seed_{result['seed']}" / result["candidate"]
        training = json.loads((folder / "training.json").read_text())
        oof = json.loads((folder / "oof/report.json").read_text())
        costs.append({"candidate": key[0], "seed": key[1], "development_fit_seconds": training["training_seconds"],
                      "oof_fit_seconds": sum(f["training_seconds"] for f in oof["folds"]),
                      "oof_elapsed_seconds": oof["seconds"], "bias_tuning_seconds": result["bias_tuning_seconds"],
                      "best_rounds": training["best_rounds"], "selected_feature_count": training["selected_feature_count"],
                      "model_bytes": (folder / "model.joblib").stat().st_size})
    write_table(destination / "training_costs.csv", pd.DataFrame(costs))

    configure_runtime_environment()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    colors = plt.get_cmap("tab10")
    handles = []
    for i, candidate in enumerate(config.candidates):
        group = summary.loc[summary.candidate == candidate.name]
        if group.empty:
            continue
        color = colors(i % 10)
        handles.append(Line2D([0], [0], color=color, marker="o", label=candidate.name))
        for axis, x_field in zip(axes, ("benign_false_positive_rate_mean", "inference_us_per_sample_mean")):
            ordered = group.set_index("decision")
            if {"argmax", "bias"}.issubset(ordered.index):
                pair = ordered.loc[["argmax", "bias"]]
                axis.plot(pair[x_field], pair.macro_f1_mean, color=color, alpha=.5, linewidth=1)
            for row in group.itertuples():
                axis.scatter(getattr(row, x_field), row.macro_f1_mean, color=color,
                             marker="o" if row.decision == "argmax" else "D", s=55,
                             edgecolor="white", linewidth=.6, zorder=3)
                if row.seed_count > 1:
                    axis.errorbar(getattr(row, x_field), row.macro_f1_mean, yerr=row.macro_f1_std,
                                  color=color, capsize=3, alpha=.5)
    for axis in axes:
        axis.set_ylabel("Validation Macro-F1")
        axis.grid(alpha=.2)
    axes[0].set_xlabel("Benign false-positive rate")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Detection / false-alarm trade-off")
    axes[1].set_xlabel("CPU inference time (microseconds per flow, batched)")
    axes[1].set_title("Detection / in-run inference timing")
    handles += [Line2D([0], [0], color="gray", marker="o", linestyle="none", label="argmax"),
                Line2D([0], [0], color="gray", marker="D", linestyle="none", label="OOF bias")]
    figure.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=8)
    prefix = "PROVISIONAL — " if not complete else ""
    figure.suptitle(f"{prefix}{config.num_classes}-class validation | {len(config.seeds)} declared seed(s)")
    figure.savefig(destination / "tradeoffs.png", dpi=180)
    figure.savefig(destination / "tradeoffs.svg")
    plt.close(figure)

    targets = ("WebBased", "BruteForce") if config.num_classes == 8 else ("SqlInjection", "DictionaryBruteForce")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, label in zip(axes, targets):
        for i, candidate in enumerate(config.candidates):
            group = per_class_summary.loc[(per_class_summary.candidate == candidate.name) &
                                          (per_class_summary.label == label)]
            if group.empty:
                continue
            color = colors(i % 10)
            ordered = group.set_index("decision")
            if {"argmax", "bias"}.issubset(ordered.index):
                pair = ordered.loc[["argmax", "bias"]]
                axis.plot(pair.precision_mean, pair.recall_mean, color=color, alpha=.5, linewidth=1)
            for row in group.itertuples():
                axis.scatter(row.precision_mean, row.recall_mean, color=color, s=55,
                             marker="o" if row.decision == "argmax" else "D", edgecolor="white", linewidth=.6)
                if row.seed_count > 1:
                    axis.errorbar(row.precision_mean, row.recall_mean, xerr=row.precision_std,
                                  yerr=row.recall_std, color=color, capsize=3, alpha=.5)
        axis.set(xlabel="Precision", ylabel="Recall", title=label, xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=.2)
    figure.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=8)
    figure.suptitle(f"{prefix}Validation precision / recall | {len(config.seeds)} declared seed(s)")
    figure.savefig(destination / "precision_recall.png", dpi=180)
    figure.savefig(destination / "precision_recall.svg")
    plt.close(figure)

    data = json.loads((run_path / "data_report.json").read_text(encoding="utf-8"))
    lines = [f"# {prefix}{config.num_classes}-class validation report", "",
             f"Run status: `{report['status']}`. Training-source pool: {data['selected_rows']:,} rows. "
             f"Declared seeds: {', '.join(map(str, config.seeds))}.", "",
             f"Validation rows per seed: {min(r['rows'] for r in report['results']):,}–"
             f"{max(r['rows'] for r in report['results']):,}.", "",
             "These are validation results used for selection, not final test performance. "
             "Error bars, where present, show standard deviation across seeds, not confidence intervals.", "",
             "| Candidate | Decision | Seeds | Macro-F1 | Benign FPR | CPU us/flow | FPR guard passes |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in summary.itertuples():
        lines.append(f"| {row.candidate} | {row.decision} | {row.seed_count} | {row.macro_f1_mean:.4f} "
                     f"| {row.benign_false_positive_rate_mean:.2%} | {row.inference_us_per_sample_mean:.2f} "
                     f"| {'yes' if row.fpr_guardrail_pass else 'no'} |")
    lines += ["", "![Validation trade-offs](tradeoffs.png)", "",
              "The table/figure above use original in-run timing medians. Consult the paired "
              "post-training check below, when available, before making latency-ratio claims.", "",
              "## Target-class precision and recall", "",
              "| Candidate | Decision | Class | Precision | Recall | F1 | Support per seed |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in per_class_summary.loc[per_class_summary.label.isin(targets)].itertuples():
        lines.append(f"| {row.candidate} | {row.decision} | {row.label} | {row.precision_mean:.2%} "
                     f"| {row.recall_mean:.2%} | {row.f1_score_mean:.4f} | {row.support_min}–{row.support_max} |")
    lines += ["", "![Validation precision and recall](precision_recall.png)", "",
              "Lines connect the two evaluated decision rules; they are not full threshold-sweep curves.", "",
              "All classes, per seed and as mean/std summaries, are in `per_class_validation.csv` "
              "and `per_class_summary.csv`. Recall limits have not been added to model selection.", "",
              "`training_costs.csv` separates the development model fit from OOF model-fitting time, "
              "OOF elapsed time (including scoring/exports), and bias optimization. OOF elapsed time "
              "already includes its model fits; do not add those two columns together.", "",
              "Selection policy and both winners are in [selection_comparison.json](selection_comparison.json). "
              "An incomplete run's leaders are provisional. Timing is repeated CPU batch throughput, "
              "including preprocessing and the decision rule; it is not single-flow latency.", ""]
    for seed in config.seeds:
        benchmark = destination / f"paired_benchmark_seed_{seed}" / "summary.csv"
        if benchmark.exists():
            table = pd.read_csv(benchmark)
            scope = json.loads(benchmark.with_name("scope.json").read_text())
            lines += [f"## Paired post-training timing: seed {seed}", "",
                      f"Full validation warm-up, {scope['repeats']} rotating-order measurement repeats, "
                      f"n_jobs={scope['n_jobs']}. "
                      "The range describes timing variability, not training-seed uncertainty.", "",
                      "| Candidate | Decision | Median us/flow | Min–max us/flow |",
                      "|---|---|---:|---:|"]
            for row in table.itertuples():
                lines.append(f"| {row.candidate} | {row.decision} | {row.median_us:.2f} | {row.min_us:.2f}–{row.max_us:.2f} |")
            lines.append("")
    lines += ["## Feature-vector overlap", ""]
    for seed in config.seeds:
        overlap_path = run_path / f"seed_{seed}" / "overlap.json"
        if overlap_path.exists():
            overlap = json.loads(overlap_path.read_text())
            lines.append(f"Seed {seed}: {overlap['overlapping_right_rows']:,} validation rows "
                         f"({overlap['overlapping_right_fraction']:.2%}) have numeric float32 feature "
                         "vectors also present in development. Labels were excluded from these hashes.")
    lines += ["", "A feature match does not establish that two records are the same physical flow. "
              "These random-row results do not establish generalization to new captures/devices. "
              "A filtered unseen-vector subset is an exploratory check, not grouped retraining or a final test.", ""]
    path = destination / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
