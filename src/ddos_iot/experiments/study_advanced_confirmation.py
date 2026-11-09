"""Confirm a promising, frozen specialist or stack without retuning its biases."""

from dataclasses import replace
import gc
import json
import logging
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle
from .recall_constraints import reference_constraints
from .storage import object_digest, write_arrays, write_report, write_table
from .study import _cpu_timing, _score
from .study_config import TrainingTrial, load_study_config
from .study_confirmation import confirmation_summary
from .study_data import load_study_data, protected_ids, study_split
from .study_models import SpecialistBundle
from .study_specialist import fit_specialist
from .study_stacking import run_stacking_study
from .study_training import refit_trial

LOGGER = logging.getLogger(__name__)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_stack_source(source, parent_selection, confirmation, seed, output):
    """Prepare only the frozen bases needed for confirmation; no trial search."""
    original = _read(source / "report.json")
    config = replace(load_study_config(source / "config.json"), output_path=output,
                     seeds=(seed,), latency_repeats=1, full_reference_fit=False)
    data = load_study_data(config)
    fitting, tuning, validation = study_split(data.y, config, seed)
    development = np.sort(np.concatenate([fitting, tuning]))
    output.mkdir(parents=True)
    write_report(output / "config.json", config.as_dict())
    write_report(output / "data_report.json", data.metadata)
    root = output / f"seed_{seed}"
    write_arrays(root / "splits.npz", fitting=fitting, tuning=tuning, validation=validation, y=data.y,
                 classes=np.asarray(data.encoder.classes_, dtype=str))
    names = [parent_selection["best_xgboost"], "catboost"]
    confirmed_rows = [r for r in _read(confirmation / "report.json")["results"] if r["seed"] == seed]
    cached = {r["candidate"]: r for r in confirmed_rows}
    with np.load(confirmation / f"seed_{seed}/splits.npz") as splits:
        np.testing.assert_array_equal(validation, splits["validation"])
    validation_rows = []
    constraints = None
    benign = list(data.encoder.classes_).index("Benign" if config.num_classes == 8 else "BenignTraffic")
    for name in list(dict.fromkeys(["paper_fixed", *names])):
        settings = parent_selection["trials"][name]
        trial = TrainingTrial(**settings["trial"])
        if name in names:
            LOGGER.info("Frozen stack confirmation seed %s: fit base %s on A", seed, name)
            bundle, training = refit_trial(data.x.iloc[fitting], data.y[fitting], data.encoder, trial, settings["rounds"], config, seed)
            save_joblib_atomic(root / "screening" / name / "model.joblib", bundle)
            write_report(root / "screening" / name / "training.json", training)
            del bundle
            gc.collect()
        destination = root / "finalists" / name
        if name in cached:
            artifact = Path(cached[name]["artifact_folder"])
            destination.mkdir(parents=True)
            shutil.copy2(artifact / "model.joblib", destination / "model.joblib")
            shutil.copy2(artifact / "training.json", destination / "training.json")
            (destination / "validation").mkdir()
            for file in (artifact / "validation").iterdir():
                if file.is_file(): shutil.copy2(file, destination / "validation" / file.name)
            row = cached[name]
            if constraints is None:
                with np.load(artifact / "validation/predictions.npz") as saved:
                    np.testing.assert_array_equal(saved["y"], data.y[validation])
                    constraints = reference_constraints(data.y[validation], saved["predicted"], config.num_classes,
                                                        benign_id=benign, protected_ids=protected_ids(data.encoder.classes_, config))
        else:
            LOGGER.info("Frozen stack confirmation seed %s: fit development base %s", seed, name)
            bundle, training = refit_trial(data.x.iloc[development], data.y[development], data.encoder,
                                           trial, settings["rounds"], config, seed)
            save_joblib_atomic(destination / "model.joblib", bundle)
            write_report(destination / "training.json", training)
            predicted, timing = _cpu_timing(bundle, data.x.iloc[validation], config)
            row = _score(destination / "validation", data.y[validation], predicted, data.encoder.classes_,
                         "validation", name, seed, constraints)
            row.update(**timing, training_seconds=training["training_seconds"])
            del bundle
            gc.collect()
        validation_rows.append(row)
    selection = {**parent_selection, "seed": seed, "confirmation_fixed_parameters": True,
                 "validation_used_for_selection": False}
    write_report(root / "selection.json", selection)
    write_report(output / "report.json", {"status": "complete", "config_sha256": object_digest(config.as_dict()),
                 "selections": {str(seed): {**selection, "sha256": object_digest(selection)}},
                 "upstream_feature_selection": original["upstream_feature_selection"],
                 "validation_results": validation_rows, "test_file_read": False, "results": [],
                 "protocol": "Frozen base settings for advanced confirmation; no repeated parameter search."})


def confirm_advanced_study(confirmation: Path, advanced: Path, output: Path, *, candidate_override=None, compare_only=False):
    base_report = _read(confirmation / "report.json")
    base_frozen = _read(confirmation / "selection.json")
    source = Path(base_report["source_study"])
    source_report = _read(source / "report.json")
    if object_digest(_read(source / "config.json")) != source_report["config_sha256"]:
        raise ValueError("Base study configuration changed.")
    source_seed = base_report["source_seed"]
    source_selection = _read(source / f"seed_{source_seed}/selection.json")
    result = _read(advanced / "report.json")
    if base_report["status"] != "complete" or object_digest(base_frozen) != base_report["selection_sha256"]:
        raise ValueError("Confirm advanced models only after base finalists have completed confirmation.")
    if object_digest(source_selection) != base_report["source_selection_sha256"]:
        raise ValueError("The underlying base study selection changed.")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "checking_evidence", "source_study": str(source.resolve()),
              "base_confirmation": str(confirmation.resolve()), "advanced_source": str(advanced.resolve()),
              "source_seed": source_seed, "seeds": base_report["seeds"], "results": [],
              "comparison_only": compare_only,
              "test_file_read": False, "source_selection_sha256": base_report["source_selection_sha256"],
              "upstream_feature_selection": source_report["upstream_feature_selection"]}
    if result["status"] == "skipped":
        report.update(status="skipped", reason="The original complexity gate skipped this method.")
        write_report(output / "report.json", report)
        return report
    if result["status"] != "complete":
        raise ValueError("Finish the advanced study before confirmation.")
    frozen = _read(advanced / "selection.json")
    if object_digest(frozen) != result["selection_sha256"]:
        raise ValueError("Advanced choices changed after freezing.")
    kind = "bias" if "single_model_conditions" in frozen else ("stack" if "conditions" in frozen else "specialist")
    if kind == "bias":
        candidate = frozen.get("guarded_macro") or ("xgb_oof_protected_bias" if "xgb_oof_protected_bias" in frozen["single_model_conditions"] else None)
    else:
        candidate = "stack_protected_bias" if kind == "stack" and "stack_protected_bias" in frozen["conditions"] else frozen.get("guarded_macro")
    if candidate_override is not None:
        pool = frozen["single_model_conditions"] if kind == "bias" else frozen["conditions"] if kind == "stack" else frozen["candidates"]
        if candidate_override not in pool:
            raise ValueError("Requested confirmation candidate was not frozen before validation.")
        candidate = candidate_override
        report["candidate_choice_role"] = "Explicit choice among frozen candidates; no recalibration. Validation is used to decide which candidates merit confirmation."
    reference = next(r for r in source_report["validation_results"] if r["seed"] == source_seed and r["candidate"] == "paper_fixed")
    observed = next((r for r in result["validation_results"] if r["candidate"] == candidate), None)
    if observed is None or (not compare_only and (not observed["reference_constraints_pass"] or observed["macro_f1"] <= reference["macro_f1"])):
        report.update(status="skipped", reason="No frozen protected advanced choice improved validation Macro-F1 while preserving reference recall/FPR.")
        write_report(output / "report.json", report)
        return report
    report.update(status="confirming", advanced_kind=kind, candidate=candidate,
                  advanced_source_selection_sha256=result["selection_sha256"])
    write_report(output / "report.json", report)
    started = time.perf_counter()
    try:
        config = replace(load_study_config(source / "config.json"), latency_repeats=1)
        reused = {**observed, "artifact_folder": str((advanced / candidate).resolve()), "reused_source_result": True}
        reused.setdefault("inference_us_per_sample", observed.get("shared_probability_inference_us", 0.) + observed.get("decision_us_per_sample", 0.))
        reused.setdefault("training_seconds", result["seconds"])
        report["results"].append(reused)
        for seed in base_report["seeds"]:
            if seed == source_seed: continue
            if kind == "stack":
                frozen_source = output / f"seed_{seed}/frozen_bases"
                _frozen_stack_source(source, source_selection, confirmation, seed, frozen_source)
                destination = output / f"seed_{seed}/stack"
                bias = frozen["conditions"][candidate]
                current = run_stacking_study(frozen_source, destination, seed=seed,
                                             fixed_conditions={candidate: bias}, meta_max_iter=frozen.get("meta_max_iter", 300),
                                             meta_solver=frozen.get("meta_solver", "lbfgs"))
                row = dict(current["validation_results"][0])
                row.update(artifact_folder=str((destination / candidate).resolve()),
                           inference_us_per_sample=row["shared_probability_inference_us"] + row["decision_us_per_sample"],
                           training_seconds=current["training_costs"]["total_fit_seconds"], timing_note="Sum of separately measured probability and decision costs.")
            else:
                data = load_study_data(config)
                with np.load(confirmation / f"seed_{seed}/splits.npz") as split:
                    development, validation = split["development"].copy(), split["validation"].copy()
                rows = [r for r in base_report["results"] if r["seed"] == seed]
                base_row = next(r for r in rows if r["candidate"] == frozen["base_candidate"])
                base = load_multiclass_bundle(Path(base_row["artifact_folder"]) / "model.joblib")
                specialist = None
                if kind == "bias":
                    base.decision_bias = np.asarray(frozen["single_model_conditions"][candidate])
                    bundle, training = base, {"training_seconds": 0., "bias_frozen_from_source_seed": True}
                else:
                    settings = frozen["candidates"][candidate]
                    group = np.asarray(frozen["group_ids"])
                    if frozen.get("reuse_specialist_confirmation"):
                        cached = _read(Path(frozen["reuse_specialist_confirmation"]) / "report.json")
                        if (cached["status"] != "complete" or cached["candidate"] != frozen["source_specialist_candidate"] or
                            cached["advanced_source_selection_sha256"] != frozen["source_specialist_selection_sha256"] or
                            Path(cached["source_study"]).resolve() != source.resolve()):
                            raise ValueError("Cached raw specialist confirmation changed.")
                        cached_row = next(r for r in cached["results"] if r["seed"] == seed)
                        bundle = joblib.load(Path(cached_row["artifact_folder"]) / "model.joblib")
                        if not isinstance(bundle, SpecialistBundle) or not np.array_equal(bundle.group_ids, group):
                            raise ValueError("Cached specialist routing group changed.")
                        base, specialist = bundle.base, bundle.specialist
                        training = {"training_seconds": max(0., cached_row["training_seconds"] - base_row["training_seconds"]),
                                    "reused_specialist_model": cached_row["artifact_folder"]}
                    else:
                        specialist, training = fit_specialist(data.x.iloc[development], data.y[development], group, config,
                                                              seed=seed, power=settings["weight_power"], rounds=settings["rounds"], depth=settings.get("depth", 6))
                        specialist.set_params(device="cpu")
                        bundle = SpecialistBundle(base, specialist, group)
                    bundle.decision_bias = np.asarray(settings["decision_bias"]) if settings.get("decision_bias") is not None else None
                destination = output / f"seed_{seed}" / candidate
                save_joblib_atomic(destination / "model.joblib", bundle)
                write_report(destination / "training.json", training)
                x_validation = data.x.iloc[validation]
                predicted = bundle.encoder.transform(bundle.predict(x_validation))
                before = time.perf_counter(); bundle.predict(x_validation); seconds = time.perf_counter() - before
                reference_row = next(r for r in rows if r["candidate"] == "paper_fixed")
                with np.load(Path(reference_row["artifact_folder"]) / "validation/predictions.npz") as saved:
                    np.testing.assert_array_equal(saved["y"], data.y[validation])
                    benign = list(data.encoder.classes_).index("Benign" if config.num_classes == 8 else "BenignTraffic")
                    constraints = reference_constraints(data.y[validation], saved["predicted"], config.num_classes,
                                                        benign_id=benign, protected_ids=protected_ids(data.encoder.classes_, config))
                row = _score(destination / "validation", data.y[validation], predicted, data.encoder.classes_,
                             "validation", candidate, seed, constraints)
                row.update(artifact_folder=str(destination.resolve()), inference_us_per_sample=seconds * 1e6 / len(validation),
                           training_seconds=base_row["training_seconds"] + training["training_seconds"],
                           full_cohort_warmup=True, inference_seconds_runs=[seconds])
                del bundle, base, specialist, data
                gc.collect()
            row["reused_source_result"] = False
            report["results"].append(row)
            write_report(output / "report.json", report)
        summary = confirmation_summary(report["results"])
        baseline_mean = np.mean([r["macro_f1"] for r in base_report["results"] if r["candidate"] == base_frozen["candidate"]])
        accepted = bool(not compare_only and summary.iloc[0].all_reference_constraints_pass and summary.iloc[0].macro_f1_mean > baseline_mean)
        write_table(output / "validation_comparison.csv", pd.DataFrame(report["results"]))
        write_table(output / "validation_summary.csv", summary)
        final = {"candidate": candidate if accepted else base_frozen["candidate"],
                 "model_kind": kind if accepted else "simple", "advanced_confirmed": accepted,
                 "advanced_source_selection": frozen, "seeds": base_report["seeds"],
                 "base_confirmation": str(confirmation.resolve()), "advanced_source": str(advanced.resolve()),
                 "comparison_only": compare_only, "compared_candidate": candidate,
                 "test_file_read": False, "source_selection_sha256": base_report["source_selection_sha256"]}
        write_report(output / "selection.json", final)
        report.update(status="complete", advanced_confirmed=accepted, selected_candidate=final["candidate"],
                      selection_sha256=object_digest(final))
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(output / "report.json", report)
    return report
