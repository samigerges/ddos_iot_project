"""OOF decision tuning for a frozen hierarchical specialist."""

import gc
import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from .decisions import predict_with_bias, tune_biases, validate_probabilities
from .joint_bias import coordinate_start, tune_joint_biases
from .recall_constraints import decision_statistics, reference_constraints, tune_constrained_biases
from .storage import close_memmap, file_digest, object_digest, write_arrays, write_report, write_table
from .study import _score
from .study_config import load_study_config
from .study_confirmation import confirmation_summary
from .study_data import load_study_data, protected_ids
from .study_specialist import fit_specialist, _predict_routed
from .study_stacking import ensure_base_oof

LOGGER = logging.getLogger(__name__)


def calibrate_specialist(source: Path, specialist_run: Path, output: Path, *, seed=42, raw_confirmation=None):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    original, raw = read(source / "report.json"), read(specialist_run / "report.json")
    config = load_study_config(source / "config.json")
    parent, frozen_raw = read(source / f"seed_{seed}/selection.json"), read(specialist_run / "selection.json")
    if original["status"] != "complete" or raw["status"] != "complete":
        raise ValueError("Finish base and raw specialist comparisons before final calibration.")
    if (object_digest(read(source / "config.json")) != original["config_sha256"] or
        object_digest(parent) != original["selections"][str(seed)]["sha256"] or
        object_digest(frozen_raw) != raw["selection_sha256"] or Path(raw["source_study"]).resolve() != source.resolve()):
        raise ValueError("Frozen source/specification changed.")
    candidate = frozen_raw["best_macro"]
    if candidate not in frozen_raw["validation_candidates"]:
        raise ValueError("Calibrate only a specialist frozen before its validation.")
    settings = frozen_raw["candidates"][candidate]
    if raw_confirmation is not None:
        confirmation = read(raw_confirmation / "report.json")
        if (confirmation["status"] != "complete" or confirmation["candidate"] != candidate or
            Path(confirmation["advanced_source"]).resolve() != specialist_run.resolve() or
            confirmation["advanced_source_selection_sha256"] != raw["selection_sha256"] or
            object_digest(read(raw_confirmation / "selection.json")) != confirmation["selection_sha256"]):
            raise ValueError("Raw specialist confirmation does not match the calibration source.")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "specialist_oof", "source_study": str(source.resolve()), "seed": seed,
              "raw_specialist_run": str(specialist_run.resolve()), "validation_results": [], "folds": [],
              "test_file_read": False, "upstream_feature_selection": original["upstream_feature_selection"],
              "scope": "Five-fold routed specialist OOF on A, with frozen B-selected group/settings. Optimization diagnostics, not nested performance estimates."}
    write_report(output / "report.json", report)
    started = time.perf_counter()
    base_p = reference_p = fused = None
    try:
        data = load_study_data(config)
        classes = data.encoder.classes_
        benign = list(classes).index("Benign" if config.num_classes == 8 else "BenignTraffic")
        group = np.asarray(frozen_raw["group_ids"])
        with np.load(source / f"seed_{seed}/splits.npz") as split:
            fitting, validation = split["fitting"].copy(), split["validation"].copy()
        name = frozen_raw["base_candidate"]
        for base in dict.fromkeys(["paper_fixed", name]):
            ensure_base_oof(data, fitting, parent["trials"][base], config, seed, output / "base_oof" / base)
        base_p = np.load(output / "base_oof" / name / "probabilities.npy", mmap_mode="r")
        reference_p = np.load(output / "base_oof/paper_fixed/probabilities.npy", mmap_mode="r")
        with np.load(output / "base_oof" / name / "rows.npz") as rows:
            np.testing.assert_array_equal(rows["row_ids"], fitting)
            fold_ids = rows["fold_ids"].copy()
        write_arrays(output / "rows.npz", row_ids=fitting, y=data.y[fitting], fold_ids=fold_ids,
                     classes=np.asarray(classes, dtype=str))
        fused = np.lib.format.open_memmap(output / "probabilities.npy", mode="w+", dtype="float32", shape=base_p.shape)
        fused[:] = np.nan
        for fold in range(1, 6):
            train, score = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            LOGGER.info("Frozen specialist OOF fold %s/5", fold)
            model, training = fit_specialist(data.x.iloc[fitting[train]], data.y[fitting[train]], group, config,
                seed=seed + fold, power=settings["weight_power"], rounds=settings["rounds"], depth=settings.get("depth", 6))
            fused[score] = _predict_routed(base_p[score], model, data.x.iloc[fitting[score]], group, config)[0]
            fused.flush()
            report["folds"].append({"fold": fold, "scoring_rows": len(score), **training})
            write_report(output / "report.json", report)
            del model
            gc.collect()
        validate_probabilities(fused)
        report["oof_probability_sha256"] = file_digest(output / "probabilities.npy")
        y = data.y[fitting]
        constraints = reference_constraints(y, reference_p.argmax(axis=1), config.num_classes,
            benign_id=benign, protected_ids=protected_ids(classes, config))
        grid = (-2., -1., -.5, -.25, 0., .25, .5, 1., 2.)
        report["status"] = "calibrating"
        write_report(output / "report.json", report)
        bias, diagnostics = tune_biases(y, fused, grid=grid)
        write_report(output / "bias_unrestricted.json", diagnostics)
        protected, protected_report = tune_constrained_biases(y, fused, constraints=constraints, grid=grid, starts=(bias,))
        write_report(output / "bias_protected.json", protected_report)
        joint, joint_report = tune_joint_biases(y, fused, constraints, coordinate_start(protected, protected_report), grid=grid)
        write_report(output / "bias_joint_protected.json", joint_report)
        conditions = {"specialist_argmax": np.zeros(config.num_classes), "specialist_oof_bias": bias}
        if protected is not None: conditions["specialist_oof_protected_bias"] = protected
        if joint is not None: conditions["specialist_oof_joint_bias"] = joint
        scores = {n: decision_statistics(y, predict_with_bias(fused, v), config.num_classes)["macro_f1"] for n, v in conditions.items()}
        guarded = [n for n in conditions if n in ("specialist_oof_protected_bias", "specialist_oof_joint_bias")]
        selected = {"base_candidate": name, "group_ids": group.tolist(),
            "candidates": {n: {**settings, "decision_bias": v.tolist()} for n, v in conditions.items()},
            "best_macro": max(scores, key=scores.get), "guarded_macro": max(guarded, key=scores.get) if guarded else None,
            "source_specialist_candidate": candidate, "source_specialist_run": str(specialist_run.resolve()),
            "source_specialist_selection_sha256": raw["selection_sha256"],
            "reuse_specialist_confirmation": str(raw_confirmation.resolve()) if raw_confirmation else None,
            "validation_used_for_selection": False, "test_file_read": False, "oof_scores": scores}
        write_report(output / "selection.json", selected)
        report.update(selection_sha256=object_digest(selected), status="validation")
        write_report(output / "report.json", report)
        for p in (base_p, reference_p, fused): close_memmap(p)
        base_p = reference_p = fused = None
        bundle = joblib.load(specialist_run / candidate / "model.joblib")
        x_validation, y_validation = data.x.iloc[validation], data.y[validation]
        probabilities = bundle.predict_proba(x_validation)
        with np.load(specialist_run / candidate / "validation/predictions.npz") as saved:
            np.testing.assert_array_equal(saved["y"], y_validation)
            np.testing.assert_array_equal(saved["predicted"], probabilities.argmax(axis=1))
        with np.load(source / f"seed_{seed}/finalists/paper_fixed/validation/predictions.npz") as saved:
            np.testing.assert_array_equal(saved["y"], y_validation)
            constraints = reference_constraints(y_validation, saved["predicted"], config.num_classes,
                benign_id=benign, protected_ids=protected_ids(classes, config))
        original_row = next(r for r in raw["validation_results"] if r["candidate"] == candidate)
        extra_fit = sum(r["training_seconds"] for r in report["folds"])
        for condition, vector in conditions.items():
            predicted = predict_with_bias(probabilities, vector)
            row = _score(output / condition / "validation", y_validation, predicted, classes, "validation", condition, seed, constraints)
            row.update(shared_probability_inference_us=original_row["inference_us_per_sample"],
                training_seconds=original_row["training_seconds"] + extra_fit,
                timing_note="Raw hierarchy timing reused; final decision overhead measured in the paired benchmark. No new paired latency claim here.")
            bundle.decision_bias = vector
            save_joblib_atomic(output / condition / "model.joblib", bundle)
            report["validation_results"].append(row)
        if object_digest(read(output / "selection.json")) != report["selection_sha256"]:
            raise ValueError("Specialist decision vectors changed during validation.")
        write_table(output / "validation_comparison.csv", pd.DataFrame(report["validation_results"]))
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for p in (base_p, reference_p, fused): close_memmap(p)
        report["seconds"] = time.perf_counter() - started
        write_report(output / "report.json", report)
    return report


def confirm_specialist_decisions(base_confirmation: Path, raw_confirmation: Path, calibration: Path, output: Path):
    """Score all frozen decisions with one shared probability pass per seed."""
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    base, raw, calibrated = (read(p / "report.json") for p in (base_confirmation, raw_confirmation, calibration))
    frozen = read(calibration / "selection.json")
    source = Path(base["source_study"])
    if any(r["status"] != "complete" for r in (base, raw, calibrated)):
        raise ValueError("Complete base, raw specialist and calibration stages first.")
    if (object_digest(frozen) != calibrated["selection_sha256"] or
        object_digest(read(base_confirmation / "selection.json")) != base["selection_sha256"] or
        object_digest(read(raw_confirmation / "selection.json")) != raw["selection_sha256"] or
        raw["advanced_source_selection_sha256"] != frozen["source_specialist_selection_sha256"] or
        raw["candidate"] != frozen["source_specialist_candidate"] or raw["seeds"] != base["seeds"] or
        Path(raw["source_study"]).resolve() != source.resolve() or Path(calibrated["source_study"]).resolve() != source.resolve()):
        raise ValueError("Frozen specialist/confirmation identity changed.")
    original = read(source / "report.json")
    if object_digest(read(source / "config.json")) != original["config_sha256"]:
        raise ValueError("Base configuration changed.")
    config = load_study_config(source / "config.json")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "confirming", "source_study": str(source.resolve()), "source_seed": base["source_seed"],
        "base_confirmation": str(base_confirmation.resolve()), "advanced_source": str(calibration.resolve()),
        "source_selection_sha256": base["source_selection_sha256"],
        "advanced_source_selection_sha256": calibrated["selection_sha256"], "seeds": base["seeds"],
        "validation_results": [], "results": [], "test_file_read": False, "advanced_kind": "specialist",
        "upstream_feature_selection": original["upstream_feature_selection"],
        "scope": "All OOF-frozen specialist decision vectors; raw fitted hierarchies reused. One shared probability pass per additional seed. No decision tuning on these validation rows."}
    write_report(output / "report.json", report)
    try:
        for row in calibrated["validation_results"]:
            raw_row = next(r for r in raw["results"] if r["seed"] == row["seed"])
            report["validation_results"].append({**row, "artifact_folder": str((calibration / row["candidate"]).resolve()),
                "inference_us_per_sample": raw_row["inference_us_per_sample"], "reused_source_result": True})
        for seed in base["seeds"]:
            if seed == base["source_seed"]: continue
            LOGGER.info("Shared specialist probability confirmation seed %s", seed)
            data = load_study_data(config)
            with np.load(base_confirmation / f"seed_{seed}/splits.npz") as split: validation = split["validation"].copy()
            x, y = data.x.iloc[validation], data.y[validation]
            raw_row = next(r for r in raw["results"] if r["seed"] == seed)
            raw_folder = Path(raw_row["artifact_folder"])
            bundle = joblib.load(raw_folder / "model.joblib")
            if list(bundle.encoder.classes_) != list(data.encoder.classes_) or not np.array_equal(bundle.group_ids, frozen["group_ids"]):
                raise ValueError("Cached raw specialist mapping changed.")
            probabilities = bundle.predict_proba(x)
            with np.load(raw_folder / "validation/predictions.npz") as saved:
                np.testing.assert_array_equal(saved["y"], y)
                np.testing.assert_array_equal(saved["predicted"], probabilities.argmax(axis=1))
            reference = next(r for r in base["results"] if r["seed"] == seed and r["candidate"] == "paper_fixed")
            with np.load(Path(reference["artifact_folder"]) / "validation/predictions.npz") as saved:
                np.testing.assert_array_equal(saved["y"], y)
                constraints = reference_constraints(y, saved["predicted"], config.num_classes,
                    benign_id=list(data.encoder.classes_).index("Benign" if config.num_classes == 8 else "BenignTraffic"),
                    protected_ids=protected_ids(data.encoder.classes_, config))
            for condition, settings in frozen["candidates"].items():
                vector = np.asarray(settings["decision_bias"])
                before = time.perf_counter(); predicted = predict_with_bias(probabilities, vector); seconds = time.perf_counter() - before
                destination = output / f"seed_{seed}" / condition
                row = _score(destination / "validation", y, predicted, data.encoder.classes_, "validation", condition, seed, constraints)
                row.update(artifact_folder=str(destination.resolve()), reused_source_result=False,
                    training_seconds=raw_row["training_seconds"], shared_probability_inference_us=raw_row["inference_us_per_sample"],
                    decision_us_per_sample=seconds * 1e6 / len(y),
                    inference_us_per_sample=raw_row["inference_us_per_sample"] + seconds * 1e6 / len(y),
                    timing_note="Raw complete-hierarchy cost plus separately measured new decision cost; conservative component estimate, not paired timing.")
                bundle.decision_bias = vector
                save_joblib_atomic(destination / "model.joblib", bundle)
                write_report(destination / "training.json", {"training_seconds": raw_row["training_seconds"], "reused_hierarchy": str(raw_folder)})
                report["validation_results"].append(row)
            write_report(output / "report.json", report)
            del bundle, probabilities, x, data
            gc.collect()
        summary = confirmation_summary(report["validation_results"])
        if not (summary.seed_count == len(base["seeds"])).all(): raise ValueError("Decision confirmation seed coverage is incomplete.")
        base_choice = read(base_confirmation / "selection.json")["candidate"]
        baseline_mean = np.mean([r["macro_f1"] for r in base["results"] if r["candidate"] == base_choice])
        feasible = summary.loc[summary.all_reference_constraints_pass & (summary.macro_f1_mean > baseline_mean)]
        accepted = not feasible.empty
        chosen = str((feasible if accepted else summary).sort_values("macro_f1_mean", ascending=False, kind="stable").iloc[0].candidate)
        final = {"candidate": chosen if accepted else base_choice, "model_kind": "specialist" if accepted else "simple",
            "advanced_confirmed": accepted, "advanced_source_selection": frozen, "seeds": base["seeds"],
            "base_confirmation": str(base_confirmation.resolve()), "advanced_source": str(calibration.resolve()),
            "source_selection_sha256": base["source_selection_sha256"], "test_file_read": False}
        write_report(output / "selection.json", final)
        report.update(status="complete", advanced_confirmed=accepted, selected_candidate=final["candidate"],
            candidate=chosen, selection_sha256=object_digest(final),
            results=[r for r in report["validation_results"] if r["candidate"] == chosen])
        write_table(output / "validation_comparison.csv", pd.DataFrame(report["validation_results"]))
        write_table(output / "validation_summary.csv", summary)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(output / "report.json", report)
    return report
