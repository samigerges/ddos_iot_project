"""Single-XGBoost OOF bias baseline before any ensemble is promoted."""

import gc
import json
import logging
from pathlib import Path
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle
from .decisions import predict_with_bias, tune_biases
from .joint_bias import coordinate_start, tune_joint_biases
from .recall_constraints import reference_constraints, tune_constrained_biases
from .storage import close_memmap, object_digest, write_report, write_table
from .study import _score
from .study_config import load_study_config
from .study_data import load_study_data, protected_ids
from .study_stacking import ensure_base_oof
from .study_training import trial_probabilities

LOGGER = logging.getLogger(__name__)


def run_single_bias_study(source: Path, output: Path, *, seed=42):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    original = read(source / "report.json")
    if original["status"] != "complete":
        raise ValueError("Complete the training study before OOF decision tuning.")
    config = load_study_config(source / "config.json")
    if object_digest(read(source / "config.json")) != original["config_sha256"]:
        raise ValueError("Base study configuration changed.")
    parent = read(source / f"seed_{seed}/selection.json")
    if object_digest(parent) != original["selections"][str(seed)]["sha256"]:
        raise ValueError("Base study selection changed.")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "base_oof", "source_study": str(source.resolve()), "seed": seed,
              "validation_results": [], "test_file_read": False,
              "upstream_feature_selection": original["upstream_feature_selection"]}
    write_report(output / "report.json", report)
    p = reference_p = None
    try:
        data = load_study_data(config)
        classes = data.encoder.classes_
        benign = list(classes).index("Benign" if config.num_classes == 8 else "BenignTraffic")
        with np.load(source / f"seed_{seed}/splits.npz") as splits:
            fitting, validation = splits["fitting"].copy(), splits["validation"].copy()
        name = parent["best_xgboost"]
        for base in dict.fromkeys(["paper_fixed", name]):
            ensure_base_oof(data, fitting, parent["trials"][base], config, seed, output / "base_oof" / base)
        p = np.load(output / "base_oof" / name / "probabilities.npy", mmap_mode="r")
        reference_p = np.load(output / "base_oof/paper_fixed/probabilities.npy", mmap_mode="r")
        y = data.y[fitting]
        constraints = reference_constraints(y, reference_p.argmax(axis=1), config.num_classes,
                                            benign_id=benign, protected_ids=protected_ids(classes, config))
        report.update(status="calibrating", search_phase="unrestricted")
        write_report(output / "report.json", report)
        LOGGER.info("Single-model decision search: unrestricted, %s OOF rows", len(y))
        bias, diagnostics = tune_biases(y, p, grid=(-2., -1., -.5, 0., .5, 1., 2.))
        write_report(output / "bias_unrestricted.json", diagnostics)
        report["search_phase"] = "protected_coordinate"
        write_report(output / "report.json", report)
        LOGGER.info("Single-model decision search: protected coordinate")
        protected, protected_report = tune_constrained_biases(y, p, constraints=constraints,
            grid=(-3., -2., -1., -.5, 0., .5, 1., 2., 3.), starts=(bias,))
        write_report(output / "bias_protected.json", protected_report)
        report["search_phase"] = "protected_joint"
        write_report(output / "report.json", report)
        LOGGER.info("Single-model decision search: protected joint")
        joint, joint_report = tune_joint_biases(y, p, constraints, coordinate_start(protected, protected_report),
                                               grid=(-3., -2., -1., -.5, 0., .5, 1., 2., 3.))
        write_report(output / "bias_joint_protected.json", joint_report)
        conditions = {"xgb_oof_bias": bias.tolist()}
        if protected is not None: conditions["xgb_oof_protected_bias"] = protected.tolist()
        if joint is not None: conditions["xgb_oof_joint_protected_bias"] = joint.tolist()
        guarded_choices = []
        if protected is not None: guarded_choices.append((protected_report["tuned"]["macro_f1"], "xgb_oof_protected_bias"))
        if joint is not None: guarded_choices.append((joint_report["tuned"]["macro_f1"], "xgb_oof_joint_protected_bias"))
        guarded_name = max(guarded_choices, key=lambda item: item[0])[1] if guarded_choices else None
        frozen = {"base_candidate": name, "single_model_conditions": conditions, "test_file_read": False,
                  "guarded_macro": guarded_name,
                  "validation_used_for_tuning": False,
                  "calibration_scope": "Five-fold base OOF on fitting partition A; parameters/rounds previously selected on B. Optimization diagnostics, not nested performance estimates."}
        write_report(output / "selection.json", frozen)
        report["selection_sha256"] = object_digest(frozen)
        report.update(status="validation", search_phase="frozen")
        write_report(output / "report.json", report)
        close_memmap(p); close_memmap(reference_p)
        p = reference_p = None
        x_validation, y_validation = data.x.iloc[validation], data.y[validation]
        base = load_multiclass_bundle(source / f"seed_{seed}/finalists" / name / "model.joblib")
        probabilities = trial_probabilities(base, x_validation, replace(config, device="cpu"))
        report["validation_probability_device"] = "cpu"
        with np.load(source / f"seed_{seed}/finalists/paper_fixed/validation/predictions.npz") as saved:
            np.testing.assert_array_equal(saved["y"], y_validation)
            constraints = reference_constraints(y_validation, saved["predicted"], config.num_classes,
                                                benign_id=benign, protected_ids=protected_ids(classes, config))
        base_cost = next(r["inference_us_per_sample"] for r in original["validation_results"] if r["seed"] == seed and r["candidate"] == name)
        fit_cost = read(source / f"seed_{seed}/finalists" / name / "training.json")["training_seconds"]
        fit_cost += sum(f["training_seconds"] for f in read(output / "base_oof" / name / "report.json")["folds"])
        for condition, vector in conditions.items():
            vector = np.asarray(vector)
            predicted = predict_with_bias(probabilities, vector)
            row = _score(output / condition / "validation", y_validation, predicted, classes, "validation", condition, seed, constraints)
            times = []
            for _ in range(config.latency_repeats):
                before = time.perf_counter(); predict_with_bias(probabilities, vector); times.append(time.perf_counter() - before)
            row.update(shared_probability_inference_us=base_cost,
                       decision_us_per_sample=float(np.median(times) * 1e6 / len(validation)),
                       timing_note="Parent's measured base cost plus separately measured decision cost; not paired end-to-end timing.",
                       training_seconds=fit_cost)
            base.decision_bias = vector
            save_joblib_atomic(output / condition / "model.joblib", base)
            report["validation_results"].append(row)
        if object_digest(read(output / "selection.json")) != report["selection_sha256"]:
            raise ValueError("Single-model biases changed during validation.")
        write_table(output / "validation_comparison.csv", pd.DataFrame(report["validation_results"]))
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close_memmap(p); close_memmap(reference_p)
        report["seconds"] = time.perf_counter() - started
        write_report(output / "report.json", report)
    return report
