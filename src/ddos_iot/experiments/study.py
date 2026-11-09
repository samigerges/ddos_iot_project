"""Staged capacity/weight/diversity study; final tests are not an input."""

from dataclasses import asdict
import gc
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle, predict_probabilities
from .evaluation import save_scores
from .recall_constraints import constraint_statistics, decision_statistics, reference_constraints
from .storage import object_digest, write_arrays, write_report, write_table
from .study_config import TrainingTrial, weight_trials
from .study_data import load_study_data, protected_ids, study_split
from .study_training import fit_trial, refit_trial, trial_probabilities

LOGGER = logging.getLogger(__name__)


def error_diversity(y, reference, other, target_ids, benign_id):
    left, right = reference != y, other != y
    a, b = float(left.mean()), float(right.mean())
    denom = np.sqrt(a * (1 - a) * b * (1 - b))
    correlation = float((np.mean(left & right) - a * b) / denom) if denom else None
    target = np.isin(y, target_ids)
    rescued, spoiled = left & ~right, ~left & right
    return {"error_correlation": correlation, "rescued_samples": int(rescued.sum()),
            "spoiled_samples": int(spoiled.sum()), "target_rescues": int((rescued & target).sum()),
            "target_spoiled": int((spoiled & target).sum()),
            "target_to_benign_rescues": int(((reference == benign_id) & (other == y) & target).sum()),
            "disagreement_fraction": float(np.mean(reference != other)),
            "scope": "Training-only tuning cohort, not independent performance evidence.",
            "complementarity_present": bool((rescued & target).any())}


def _cpu_timing(bundle, x, config):
    def predict():
        return predict_probabilities(bundle.name, bundle.model, x, config.batch_size, device="cpu").argmax(axis=1)
    warmed = predict()
    times = []
    for _ in range(config.latency_repeats):
        started = time.perf_counter(); predicted = predict(); times.append(time.perf_counter() - started)
        if not np.array_equal(warmed, predicted):
            raise RuntimeError("Repeated CPU predictions changed.")
    return warmed, {"inference_us_per_sample": float(np.median(times) * 1e6 / len(x)),
                    "inference_seconds_runs": times, "inference_device": "cpu", "full_cohort_warmup": True,
                    "latency_scope": "Batched prediction on published selected/scaled features; excludes raw feature extraction/loading.",
                    "single_flow_latency_measured": False}


def _score(folder, y, predicted, classes, phase, candidate, seed, constraints=None):
    metrics = save_scores(folder, classes, y, predicted, split=phase,
                          role="Training-only candidate/round selection diagnostics." if phase == "tuning" else
                          "Validation after training-only choices were frozen; upstream published feature selection is not fold-local.")
    stats = decision_statistics(y, predicted, len(classes))
    row = {**metrics, "candidate": candidate, "seed": seed, "recall_by_class": dict(zip(classes, stats["recall"]))}
    if constraints is not None:
        row["reference_constraints_pass"] = constraint_statistics(stats, constraints)["feasible"]
    return row


def run_training_study(config):
    config.output_path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_report(config.output_path / "config.json", config.as_dict())
    report = {"status": "loading", "config_sha256": object_digest(config.as_dict()), "num_classes": config.num_classes,
              "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost")},
              "results": [], "validation_results": [], "failures": [], "selections": {},
              "test_file_read": False, "test_used_for_selection": False,
              "protocol": "Verified published training subset; disjoint fit/tune/validation; choices freeze before validation.",
              "upstream_feature_selection": "Author-selected/scaled 23 features; cannot establish nested preprocessing or exact held-out paper reproduction.",
              "paper_reference_evaluation": "Pending a separately frozen final-test protocol; development scores are not the paper test score."}
    report_path = config.output_path / "report.json"
    write_report(report_path, report)
    try:
        data = load_study_data(config)
        write_report(config.output_path / "data_report.json", data.metadata)
        classes = data.encoder.classes_
        target = protected_ids(classes, config)
        benign = list(classes).index("Benign" if config.num_classes == 8 else "BenignTraffic")
        for seed in config.seeds:
            root = config.output_path / f"seed_{seed}"
            root.mkdir()
            fitting, tuning, validation = study_split(data.y, config, seed)
            write_arrays(root / "splits.npz", fitting=fitting, tuning=tuning, validation=validation,
                         y=data.y, classes=np.asarray(classes, dtype=str))
            x_fit, y_fit = data.x.iloc[fitting], data.y[fitting]
            x_tune, y_tune = data.x.iloc[tuning], data.y[tuning]
            trials, rows, cached_ids, fitted_rounds = {}, [], {}, {}
            constraints = None

            def run_trial(trial):
                nonlocal constraints
                destination = root / "screening" / trial.name
                destination.mkdir(parents=True)
                LOGGER.info("%s classes seed %s: screening %s", config.num_classes, seed, trial.name)
                bundle = None
                try:
                    bundle, training = fit_trial(x_fit, y_fit, data.encoder, trial, config, seed, tuning=(x_tune, y_tune))
                    save_joblib_atomic(destination / "model.joblib", bundle)
                    write_report(destination / "training.json", training)
                    p = trial_probabilities(bundle, x_tune, config)
                    predicted = p.argmax(axis=1)
                    if constraints is None:
                        constraints = reference_constraints(y_tune, predicted, config.num_classes,
                                                            benign_id=benign, protected_ids=target)
                    row = _score(destination / "tuning", y_tune, predicted, classes, "tuning", trial.name, seed, constraints)
                    row.update(training_seconds=training["training_seconds"], rounds=training["rounds"],
                               trial=asdict(trial))
                    rows.append(row); trials[trial.name] = trial; fitted_rounds[trial.name] = training["rounds"]
                    cached_ids[trial.name] = predicted
                    report["results"].append(row)
                    LOGGER.info("%s: tuning Macro-F1 %.5f; protected recalls/FPR pass=%s; rounds=%s",
                                trial.name, row["macro_f1"], row["reference_constraints_pass"], training["rounds"])
                except Exception as error:
                    failure = {"candidate": trial.name, "seed": seed, "error": f"{type(error).__name__}: {error}"}
                    write_report(destination / "failure.json", failure)
                    report["failures"].append(failure)
                    LOGGER.exception("Trial failed: %s", trial.name)
                    if trial.name == "paper_fixed":
                        raise
                finally:
                    del bundle
                    gc.collect()
                    write_report(report_path, report)

            report["status"] = "screening"
            for trial in config.capacity:
                run_trial(trial)
            capacity_names = {t.name for t in config.capacity}
            best_capacity = max((r for r in rows if r["candidate"] in capacity_names), key=lambda r: r["macro_f1"])
            for trial in weight_trials(trials[best_capacity["candidate"]], config.weight_powers):
                run_trial(trial)
            if config.catboost:
                run_trial(TrainingTrial("catboost", "catboost", depth=8, early_stopping=True))
            best = max(rows, key=lambda r: r["macro_f1"])
            guarded = max((r for r in rows if r["reference_constraints_pass"]), key=lambda r: r["macro_f1"])
            best_xgb = max((r for r in rows if trials[r["candidate"]].model == "xgboost"), key=lambda r: r["macro_f1"])
            diversity = None
            if "catboost" in cached_ids:
                diversity = error_diversity(y_tune, cached_ids[best_xgb["candidate"]], cached_ids["catboost"], target, benign)
                write_report(root / "diversity.json", diversity)
            errors = y_tune != cached_ids[best_xgb["candidate"]]
            target_errors = errors & np.isin(y_tune, target)
            specialist_evidence = {"target_errors": int(target_errors.sum()),
                                   "target_errors_to_benign": int((target_errors & (cached_ids[best_xgb["candidate"]] == benign)).sum()),
                                   "scope": "Training-only tuning cohort. Routing and within-family errors remain part of full-system evaluation."}
            write_report(root / "specialist_evidence.json", specialist_evidence)
            selected_names = list(dict.fromkeys(["paper_fixed", best["candidate"], guarded["candidate"], best_xgb["candidate"]]))
            if "catboost" in trials:
                selected_names.append("catboost")
            selected_names = list(dict.fromkeys(selected_names))
            selection = {"seed": seed, "best_macro": best["candidate"], "guarded_macro": guarded["candidate"],
                         "best_xgboost": best_xgb["candidate"], "validation_candidates": selected_names,
                         "constraints": constraints.as_dict(), "protected_labels": [classes[i] for i in target],
                         "trials": {name: {"trial": asdict(trials[name]), "rounds": fitted_rounds[name]} for name in selected_names},
                         "rule": "Training-only tuning Macro-F1; report unrestricted and all-target-recall/FPR guarded choices separately.",
                         "test_file_read": False, "validation_used_for_selection": False,
                         "complexity_evidence": {"diversity": diversity, "specialist": specialist_evidence}}
            write_report(root / "selection.json", selection)
            report["selections"][str(seed)] = {"sha256": object_digest(selection), **selection}
            write_report(report_path, report)
            # Validation features/labels are first materialized after the freeze.
            development = np.sort(np.concatenate([fitting, tuning]))
            del x_fit, x_tune, cached_ids
            gc.collect()
            x_development, y_development = data.x.iloc[development], data.y[development]
            x_validation, y_validation = data.x.iloc[validation], data.y[validation]
            validation_constraints = None
            for name in selected_names:
                LOGGER.info("%s classes seed %s: development refit/validation %s", config.num_classes, seed, name)
                bundle, training = refit_trial(x_development, y_development, data.encoder,
                                               trials[name], fitted_rounds[name], config, seed)
                destination = root / "finalists" / name
                save_joblib_atomic(destination / "model.joblib", bundle)
                write_report(destination / "training.json", training)
                predicted, timing = _cpu_timing(bundle, x_validation, config)
                if validation_constraints is None:
                    validation_constraints = reference_constraints(y_validation, predicted, config.num_classes,
                                                                       benign_id=benign, protected_ids=target)
                row = _score(destination / "validation", y_validation, predicted, classes, "validation", name, seed,
                             validation_constraints)
                row.update(**timing, training_seconds=training["training_seconds"], selected_guarded=name == guarded["candidate"],
                           selected_unrestricted=name == best["candidate"])
                report["validation_results"].append(row)
                write_report(destination / "validation/metrics.json", row)
                write_report(report_path, report)
                del bundle
                gc.collect()
            if object_digest(json.loads((root / "selection.json").read_text())) != report["selections"][str(seed)]["sha256"]:
                raise RuntimeError("Training-only selection changed during validation.")
            del x_development, x_validation
            gc.collect()
            if config.full_reference_fit:
                LOGGER.info("%s classes: full published-training fixed reference; no test access", config.num_classes)
                bundle, training = fit_trial(data.x, data.y, data.encoder, config.capacity[0], config, seed)
                destination = root / "paper_reference_full_training"
                save_joblib_atomic(destination / "model.joblib", bundle)
                write_report(destination / "training.json", {**training, "test_file_read": False,
                             "purpose": "Paper-parameter reference fitted to all published training rows; final test evaluation pending."})
                del bundle
                gc.collect()
        write_table(config.output_path / "tuning_comparison.csv", pd.DataFrame(report["results"]))
        write_table(config.output_path / "validation_comparison.csv", pd.DataFrame(report["validation_results"]))
        report["status"] = "partial_failure" if report["failures"] else "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(report_path, report)
    return report
