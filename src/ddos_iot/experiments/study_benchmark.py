"""Paired CPU throughput measurements on a common validation cohort."""

import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from ddos_iot.multiclass import MulticlassBundle
from .storage import file_digest, object_digest, write_arrays, write_report, write_table
from .study_config import load_study_config
from .study_data import load_study_data
from .study_models import SpecialistBundle, StackBundle


def _cpu(bundle, threads):
    if isinstance(bundle, MulticlassBundle):
        if bundle.name == "xgboost": bundle.model.set_params(device="cpu", n_jobs=threads)
        elif bundle.name == "catboost" and bundle.model.get_param("thread_count") != threads:
            raise ValueError("CatBoost bundle must already have the declared prediction thread count.")
    elif isinstance(bundle, SpecialistBundle):
        _cpu(bundle.base, threads)
        bundle.specialist.set_params(device="cpu", n_jobs=threads)
    elif isinstance(bundle, StackBundle):
        for base in bundle.bases: _cpu(base, threads)
        bundle.n_jobs = threads
    else:
        raise ValueError("Expected a trusted local published-study bundle.")


def benchmark_study(source: Path, output: Path, *, seed=42, rows=100_000, repeats=None):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    original = read(source / "report.json")
    config = load_study_config(source / "config.json")
    if original["status"] != "complete" or type(rows) is not int or rows < config.num_classes or (repeats is not None and (type(repeats) is not int or repeats < 1)):
        raise ValueError("Require a complete source study and valid benchmark sizes.")
    if object_digest(read(source / "config.json")) != original["config_sha256"]:
        raise ValueError("Source configuration changed.")
    if object_digest(read(source / f"seed_{seed}/selection.json")) != original["selections"][str(seed)]["sha256"]:
        raise ValueError("Source selection changed.")
    specs = []
    def add(stage, candidate, path, observed):
        row = next(r for r in observed if r["seed"] == seed and r["candidate"] == candidate)
        specs.append({"stage": stage, "candidate": candidate, "model": str(path.resolve()),
                      "macro_f1": row["macro_f1"], "model_sha256": file_digest(path)})
    add("base", "paper_fixed", source / f"seed_{seed}/finalists/paper_fixed/model.joblib", original["validation_results"])
    for stage in ("bias_v1", "specialist_expanded_v2", "stack_optimizer_v2", "specialist_bias_v1"):
        if not (source / stage / "report.json").exists(): continue
        report = read(source / stage / "report.json")
        if report["status"] != "complete": continue
        selection = read(source / stage / "selection.json")
        if object_digest(selection) != report["selection_sha256"]:
            raise ValueError("Benchmark candidate selection changed after freezing.")
        candidate = "xgb_oof_bias" if stage == "bias_v1" else selection["best_macro"] if stage.startswith("specialist") else max(report["validation_results"], key=lambda r: r["macro_f1"])["candidate"]
        add(stage, candidate, source / stage / candidate / "model.joblib", report["validation_results"])
    if repeats is None: repeats = len(specs)
    if repeats % len(specs):
        raise ValueError("Use a multiple of the model count so every model occupies every timing position equally.")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "loading", "models": specs, "seed": seed, "num_classes": config.num_classes,
              "inference_threads": config.n_jobs, "results": [], "test_file_read": False,
              "scope": "Common stratified validation cohort; complete bundle prediction, CPU, rotating model order. No accuracy is estimated on the timing subset."}
    write_report(output / "report.json", report)
    try:
        data = load_study_data(config)
        with np.load(source / f"seed_{seed}/splits.npz") as split: cohort = split["validation"].copy()
        if rows < len(cohort):
            cohort, _ = train_test_split(cohort, train_size=rows, random_state=seed, stratify=data.y[cohort])
        cohort = np.sort(cohort)
        write_arrays(output / "cohort.npz", row_ids=cohort, y=data.y[cohort])
        x = data.x.iloc[cohort].copy()
        expected_classes = list(data.encoder.classes_)
        del data
        gc.collect()
        report["cohort_rows"] = len(x)
        bundles, expected = [], []
        with threadpool_limits(limits=config.n_jobs):
            for spec in specs:
                bundle = joblib.load(spec["model"])
                if list(bundle.encoder.classes_) != expected_classes:
                    raise ValueError("Benchmark model class mapping differs from the source study.")
                _cpu(bundle, config.n_jobs)
                bundles.append(bundle)
                expected.append(bundle.predict(x))
            runs = [[] for _ in specs]
            report.update(status="timing", orders=[])
            write_report(output / "report.json", report)
            for repeat in range(repeats):
                order = np.roll(np.arange(len(specs)), -repeat)
                report["orders"].append(order.tolist())
                for index in order:
                    before = time.perf_counter(); predicted = bundles[index].predict(x)
                    runs[index].append(time.perf_counter() - before)
                    np.testing.assert_array_equal(predicted, expected[index])
            for spec, elapsed in zip(specs, runs):
                report["results"].append({**spec, "seconds_runs": elapsed,
                    "inference_us_median": float(np.median(elapsed) * 1e6 / len(x)),
                    "inference_us_std": float(np.std(elapsed) * 1e6 / len(x))})
        write_table(output / "timing.csv", pd.DataFrame(report["results"]))
        baseline = report["results"][0]["inference_us_median"]
        lines = ["# Paired CPU throughput", "", report["scope"], "",
            f"Model seed {seed}; {len(x):,} common rows; {repeats} measured passes per model after a complete cohort warm-up; {config.n_jobs} threads.", "",
            "Macro-F1 comes from the original complete validation cohort. Timing subset labels do not tune or select a model.", "",
            "| Stage / candidate | Full-validation Macro-F1 | Median CPU us/flow | Timing std | Relative to reference |",
            "|---|---:|---:|---:|---:|"]
        for row in report["results"]:
            lines.append(f"| {row['stage']} / {row['candidate']} | {row['macro_f1']:.5f} | {row['inference_us_median']:.2f} | {row['inference_us_std']:.2f} | {row['inference_us_median'] / baseline:.2f}x |")
        lines += ["", "These are batched throughputs on selected/scaled inputs, excluding feature extraction, CSV loading and model loading. "
            f"The common {len(x):,}-row benchmark and full-cohort timings have different scopes; neither is single-flow latency. "
            "Rotating order reduces systematic order effects; it does not eliminate system-load/clock variation.", ""]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for row in report["results"]:
            ax.errorbar(row["inference_us_median"], row["macro_f1"], xerr=row["inference_us_std"],
                        fmt="o", capsize=3, label=f"{row['stage']} / {row['candidate']}")
        ax.set(xlabel=f"Paired batched CPU throughput (us/flow; {len(x):,} common rows)", ylabel="Full-validation Macro-F1",
               title=f"{config.num_classes} classes — seed {seed}: F1 and paired inference cost")
        ax.set_xlim(left=0)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "latency_tradeoffs.png", dpi=180)
        plt.close(fig)
        lines += ["![Validation F1 and paired CPU cost](latency_tradeoffs.png)", ""]
        (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(output / "report.json", report)
    return report
