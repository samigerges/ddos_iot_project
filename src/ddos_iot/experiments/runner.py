"""Orchestrate development ablations and a separately invoked, frozen test stage."""

from __future__ import annotations

from dataclasses import asdict
import gc
import importlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import MulticlassBundle, load_multiclass_bundle
from .config import Candidate, ExperimentConfig, config_from_dict, experiment_plan
from .data import development_split, load_test_data, load_training_data, overlap_report
from .decisions import predict_with_bias, tune_biases
from .evaluation import benchmark, paired_deltas, predictions, save_scores, summarize_seeds
from .oof import generate_oof
from .selection import analyze_selection
from .storage import close_memmap, object_digest, write_arrays, write_report, write_table
from .training import fit_partition

LOGGER = logging.getLogger(__name__)


def check_dependencies(config: ExperimentConfig) -> dict[str, str]:
    required = {c.model for c in config.candidates} | {"numpy", "pandas", "scikit-learn", "joblib"}
    if any(c.feature_selection == "median" for c in config.candidates):
        required.add("lightgbm")
    for name in required:
        importlib.import_module("sklearn" if name == "scikit-learn" else name)
    return {name: importlib.metadata.version(name) for name in sorted(required)}


def run_development(config: ExperimentConfig) -> dict[str, Any]:
    versions = check_dependencies(config)
    # Refuse even a partially populated previous run; checkpoints are not resumable.
    config.output_path.mkdir(parents=True, exist_ok=False)
    write_report(config.output_path / "config.json", config.as_dict())
    report: dict[str, Any] = {
        **experiment_plan(config), "status": "running", "versions": versions,
        "python": platform.python_version(), "platform": platform.platform(),
        "config_sha256": object_digest(config.as_dict()), "results": [], "failures": [],
        "protocol": "Existing raw partition; raw-subtype-stratified development/validation and OOF splits.",
        "cv_preprocessing": "fold", "test_used_for_selection": False,
        "seed_std_ddof": 0,
    }
    report_path = config.output_path / "experiment_report.json"
    write_report(report_path, report)
    started = time.perf_counter()
    try:
        data = load_training_data(config)
        write_report(config.output_path / "data_report.json", data.report)
        write_report(config.output_path / "label_mapping.json", dict(enumerate(data.encoder.classes_)))
        for seed in config.seeds:
            development, validation = development_split(data, config, seed)
            seed_folder = config.output_path / f"seed_{seed}"
            seed_folder.mkdir()
            write_arrays(seed_folder / "splits.npz", development_row_ids=data.row_ids[development],
                         validation_row_ids=data.row_ids[validation],
                         development_y=data.y[development], validation_y=data.y[validation],
                         classes=np.asarray(data.encoder.classes_, dtype=str))
            write_report(seed_folder / "overlap.json", overlap_report(
                data.feature_hashes[development], data.feature_hashes[validation]))
            x_development, x_validation = data.x.iloc[development], data.x.iloc[validation]
            y_development, y_validation = data.y[development], data.y[validation]
            strata = data.raw_labels[development]
            for candidate in config.candidates:
                folder = seed_folder / candidate.name
                folder.mkdir()
                probabilities = fitted = bundle = None
                try:
                    probabilities, oof_report = generate_oof(
                        x_development, y_development, strata, data.row_ids[development],
                        config, candidate, seed, folder / "oof")
                    LOGGER.info("%s seed %s: tuning biases on training OOF probabilities", candidate.name, seed)
                    bias_started = time.perf_counter()
                    bias, bias_report = tune_biases(
                        y_development, probabilities, grid=config.bias_grid,
                        passes=config.bias_passes, f1_cutoff=config.bias_f1_cutoff)
                    bias_report.update(classes=data.encoder.classes_, seconds=time.perf_counter() - bias_started)
                    write_report(folder / "bias.json", bias_report)
                    for decision in ("argmax", "bias"):
                        ids = probabilities.argmax(axis=1) if decision == "argmax" else predict_with_bias(probabilities, bias)
                        save_scores(folder / "oof" / decision, data.encoder.classes_, y_development, ids,
                                    split="oof", role="Training diagnostics; biases optimized on these labels.")
                    close_memmap(probabilities)
                    probabilities = None
                    del ids
                    LOGGER.info("%s seed %s: development refit and independent validation", candidate.name, seed)
                    fitted = fit_partition(x_development, y_development, strata, config, candidate, seed)
                    if candidate.model == "xgboost":
                        fitted.model.set_params(device="cpu")
                    bundle = MulticlassBundle(candidate.model, fitted.model, fitted.features, data.encoder)
                    # Save a reusable argmax model immediately, before validation/reporting.
                    save_joblib_atomic(folder / "model.joblib", bundle)
                    write_report(folder / "training.json", fitted.report)
                    bundle.decision_bias = bias
                    for decision in ("argmax", "bias"):
                        metrics = save_scores(
                            folder / "validation" / decision, data.encoder.classes_, y_validation,
                            predictions(bundle, x_validation, config, decision), split="validation",
                            role="Independent of fold fitting and bias optimization; used for experiment selection.")
                        row = {**metrics, **benchmark(bundle, x_validation, config, decision),
                               "candidate": candidate.name, "decision": decision, "seed": seed,
                               "selected_feature_count": len(fitted.features.columns),
                               "weight_power": candidate.weight_power,
                               "training_seconds": sum(f["training_seconds"] for f in oof_report["folds"]) + fitted.report["training_seconds"],
                               "oof_seconds": oof_report["seconds"], "bias_tuning_seconds": bias_report["seconds"]}
                        report["results"].append(row)
                        write_report(folder / "validation" / decision / "metrics.json", row)
                except Exception as exc:
                    LOGGER.exception("%s seed %s failed; preserving reports and continuing", candidate.name, seed)
                    failure = {"candidate": candidate.name, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                    report["failures"].append(failure)
                    write_report(folder / "failure.json", failure)
                    # Partial decision reports from a failed candidate cannot enter selection.
                    report["results"] = [r for r in report["results"]
                                         if not (r["candidate"] == candidate.name and r["seed"] == seed)]
                finally:
                    close_memmap(probabilities)
                    probabilities = fitted = bundle = None
                    gc.collect()
                    write_report(report_path, report)
            del x_development, x_validation
        if not report["results"]:
            raise RuntimeError("No experiments completed. Inspect per-candidate failure reports.")
        analysis = analyze_selection(report["results"], config)
        write_table(config.output_path / "validation_comparison.csv", analysis.comparisons)
        write_table(config.output_path / "validation_summary.csv", analysis.summary)
        write_table(config.output_path / "validation_deltas.csv", paired_deltas(report["results"], config.reference_name))
        write_report(config.output_path / "selection_report.json", analysis.report)
        report["selection_analysis"] = analysis.report
        winner = analysis.selected
        if winner is None:
            raise RuntimeError("No candidate completed all seeds under the declared selection policy; inspect selection_report.json and reference completeness.")
        selected = next(c for c in config.candidates if c.name == winner["candidate"])
        selection = {
            **analysis.report,
            "candidate": asdict(selected), "decision": winner["decision"],
            "validation_macro_f1_mean": winner["macro_f1_mean"],
            "validation_macro_f1_std": winner["macro_f1_std"],
            "config_sha256": report["config_sha256"], "training_source_sha256": data.report["source_sha256"],
            "classes": data.encoder.classes_, "seeds": {}, "test_file_read": False,
        }
        for seed in config.seeds:
            bias_path = config.output_path / f"seed_{seed}" / selected.name / "bias.json"
            selection["seeds"][str(seed)] = {"bias": json.loads(bias_path.read_text())["bias"]}
        write_report(config.output_path / "selection.json", selection)
        report.update(status="partial_failure" if report["failures"] else "complete",
                      selected_candidate=selected.name, selected_decision=winner["decision"],
                      selection_sha256=object_digest(selection))
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(report_path, report)
    LOGGER.info("Selected %s + %s using validation only. Test remains unread.", report["selected_candidate"], report["selected_decision"])
    return report


def evaluate_frozen_test(run_path: Path) -> dict[str, Any]:
    """Fit the declared winner across all seeds, then read the untouched raw test."""
    config_payload = json.loads((run_path / "config.json").read_text())
    config = config_from_dict(config_payload)
    experiment = json.loads((run_path / "experiment_report.json").read_text())
    selection = json.loads((run_path / "selection.json").read_text())
    if experiment["status"] not in ("complete", "partial_failure"):
        raise ValueError("Development has not completed with a frozen selection.")
    # Hash the stored snapshot, so adding new optional config fields does not
    # invalidate previously frozen experiment selections.
    if object_digest(config_payload) != selection["config_sha256"] or object_digest(selection) != experiment["selection_sha256"]:
        raise ValueError("Frozen configuration or selection was modified. Create a new development run.")
    destination = run_path / "test_evaluation"
    if destination.exists():
        raise FileExistsError("Final test evaluation already exists (possibly partial); it will not be overwritten.")
    data = load_training_data(config)
    if data.report["source_sha256"] != selection["training_source_sha256"]:
        raise ValueError("Training source changed since development. Create a new run.")
    if list(data.encoder.classes_) != selection["classes"]:
        raise ValueError("Class mapping changed since development.")
    destination.mkdir()
    candidate = Candidate(**selection["candidate"])
    report: dict[str, Any] = {"status": "running", "candidate": candidate.name,
                              "decision": selection["decision"], "results": [],
                              "test_used_for_selection": False, "selection_sha256": experiment["selection_sha256"],
                              "class_count": config.num_classes, "training_sampled": data.report["sampled"]}
    started = time.perf_counter()
    write_report(destination / "report.json", report)
    try:
        for seed in config.seeds:
            LOGGER.info("Final training: %s seed %s; test is still unread", candidate.name, seed)
            fitted = fit_partition(data.x, data.y, data.raw_labels, config, candidate, seed)
            if candidate.model == "xgboost":
                fitted.model.set_params(device="cpu")
            bias = np.array(selection["seeds"][str(seed)]["bias"]) if selection["decision"] == "bias" else None
            bundle = MulticlassBundle(candidate.model, fitted.model, fitted.features, data.encoder, bias)
            folder = destination / f"seed_{seed}"
            save_joblib_atomic(folder / "model.joblib", bundle)
            write_report(folder / "training.json", fitted.report)
            del fitted, bundle
            gc.collect()
        # No test access occurs until every final model has been persisted.
        x_test, y_test, test_report = load_test_data(config, data)
        write_report(destination / "data_report.json", test_report)
        for seed in config.seeds:
            folder = destination / f"seed_{seed}"
            bundle = load_multiclass_bundle(folder / "model.joblib")
            metrics = save_scores(folder, data.encoder.classes_, y_test,
                                  predictions(bundle, x_test, config, selection["decision"]),
                                  split="test", role="Final evaluation of the previously frozen validation winner.")
            training = json.loads((folder / "training.json").read_text())
            row = {**metrics, **benchmark(bundle, x_test, config, selection["decision"]),
                   "candidate": candidate.name, "decision": selection["decision"], "seed": seed,
                   "training_seconds": training["training_seconds"],
                   "selected_feature_count": training["selected_feature_count"]}
            report["results"].append(row)
            write_report(folder / "metrics.json", row)
            write_report(destination / "report.json", report)
            del bundle
            gc.collect()
        write_table(destination / "test_comparison.csv", pd.DataFrame(report["results"]))
        write_table(destination / "test_summary.csv", summarize_seeds(report["results"]))
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(destination / "report.json", report)
    return report
