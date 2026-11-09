"""Five-fold base OOF stacking and separately cross-fitted meta calibration."""

from contextlib import ExitStack
import gc
import json
import logging
from pathlib import Path
import time
import os
import shutil
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle
from .decisions import predict_with_bias, tune_biases, validate_probabilities
from .joint_bias import coordinate_start, tune_joint_biases
from .recall_constraints import reference_constraints, tune_constrained_biases
from .storage import close_memmap, file_digest, object_digest, write_arrays, write_report, write_table
from .study import _score
from .study_config import TrainingTrial, load_study_config
from .study_data import load_study_data, protected_ids
from .study_models import StackBundle, log_probability_features
from .study_training import refit_trial, trial_probabilities

LOGGER = logging.getLogger(__name__)


def fit_meta(a, ya, b, yb, *, seed, cap=10_000, threads=2, max_iter=2000, solver="lbfgs"):
    """Cap each class uniformly; balanced loss preserves equal class emphasis."""
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("The meta optimizer needs a positive integer iteration budget.")
    if solver not in ("lbfgs", "newton-cg"):
        raise ValueError("Use an explicitly supported multinomial L2 solver.")
    started = time.perf_counter()
    y = np.concatenate([ya, yb])
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(ids, min(cap, len(ids)), replace=False)
                             for c in np.unique(y) if len(ids := np.flatnonzero(y == c))])
    from_a = chosen < len(ya)
    x = np.empty((len(chosen), a.shape[1]), dtype=np.float64)
    x[from_a] = a[chosen[from_a]]
    x[~from_a] = b[chosen[~from_a] - len(ya)]
    model = LogisticRegression(C=1., max_iter=max_iter, class_weight="balanced", solver=solver)
    with threadpool_limits(limits=threads):
        model.fit(x, y[chosen])
    return model, {"sample_rows": len(chosen), "class_cap": cap, "seed": seed,
                   "class_counts": np.bincount(y[chosen]).tolist(), "iterations": model.n_iter_.tolist(),
                   "maximum_iterations": model.max_iter, "hit_iteration_limit": bool((model.n_iter_ >= model.max_iter).any()),
                   "solver": solver, "tolerance": model.tol,
                   "training_seconds": time.perf_counter() - started}


def crossfit_meta(a, ya, b, yb, *, seed, folder, threads=2, folds=5, max_iter=2000, solver="lbfgs"):
    """Score B folds excluded from all base fits and their own meta fit.

    A contains base OOF features, B base-held-out features. Every scoring B row
    is absent from all base gradient fits and its meta gradient fit. Base
    hyperparameters/rounds were selected using B, so these are optimization
    diagnostics rather than nested generalization estimates.
    """
    folder.mkdir(parents=True)
    p = np.lib.format.open_memmap(folder / "probabilities.npy", mode="w+", dtype="float32", shape=(len(yb), len(np.unique(ya))))
    p[:] = np.nan
    fold_ids = np.full(len(yb), -1, dtype=np.int16)
    report = {"status": "running", "folds": [], "validation_used": False, "test_used": False,
              "base_hyperparameter_selection_scope": "B labels selected base parameters/rounds; this is not nested performance estimation.",
              "score_role": "Training-only bias optimization diagnostics."}
    try:
        for fold, (train, score) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(b, yb), 1):
            LOGGER.info("Meta calibration fold %s/%s", fold, folds)
            meta, training = fit_meta(a, ya, b[train], yb[train], seed=seed + fold, threads=threads, max_iter=max_iter, solver=solver)
            with threadpool_limits(limits=threads):
                p[score] = meta.predict_proba(b[score]).astype("float32")
            fold_ids[score] = fold
            report["folds"].append({"fold": fold, "scoring_rows": len(score), **training})
            p.flush()
            del meta
        validate_probabilities(p)
        if (fold_ids < 1).any():
            raise RuntimeError("Incomplete meta OOF coverage.")
        write_arrays(folder / "rows.npz", y=yb, fold_ids=fold_ids)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        close_memmap(p)
        raise
    finally:
        write_report(folder / "report.json", report)
    return p


def _base_oof(data, fitting, settings, config, seed, folder):
    folder.mkdir(parents=True)
    trial = TrainingTrial(**settings["trial"])
    y = data.y[fitting]
    p = np.lib.format.open_memmap(folder / "probabilities.npy", mode="w+", dtype="float32", shape=(len(y), config.num_classes))
    p[:] = np.nan
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    report = {"status": "running", "folds": [], "scoring_rows": len(y), "trial": settings,
              "source_sha256": data.metadata["source_sha256"], "seed": seed,
              "model_library_version": importlib.metadata.version(trial.model),
              "sklearn_version": importlib.metadata.version("scikit-learn"),
              "fitting_implementation_sha256": file_digest(Path(__file__).with_name("study_training.py")),
              "preprocessing": "Published selected/scaled input; upstream selection is not fold-local.",
              "early_stopping": "None within OOF fits; fixed rounds selected on disjoint B tuning partition."}
    try:
        for fold, (train, score) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fitting, y), 1):
            LOGGER.info("%s base OOF fold %s/5", trial.name, fold)
            bundle, training = refit_trial(data.x.iloc[fitting[train]], y[train], data.encoder,
                                           trial, settings["rounds"], config, seed + fold)
            p[score] = trial_probabilities(bundle, data.x.iloc[fitting[score]], config)
            fold_ids[score] = fold
            report["folds"].append({"fold": fold, **training})
            p.flush()
            write_arrays(folder / "rows.npz", row_ids=fitting, y=y, fold_ids=fold_ids)
            write_report(folder / "report.json", report)
            del bundle
            gc.collect()
        validate_probabilities(p)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close_memmap(p)
        if report["status"] == "complete":
            report["probability_sha256"] = file_digest(folder / "probabilities.npy")
            report["rows_sha256"] = file_digest(folder / "rows.npz")
        write_report(folder / "report.json", report)


def ensure_base_oof(data, fitting, settings, config, seed, destination):
    """Share immutable complete caches between single-model bias and stacking."""
    cache = config.output_path / "oof_cache" / f"seed_{seed}" / settings["trial"]["name"]
    if not cache.exists():
        _base_oof(data, fitting, settings, config, seed, cache)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    report = read(cache / "report.json")
    if report["status"] != "complete" or report["trial"] != settings or report["source_sha256"] != data.metadata["source_sha256"] or report["seed"] != seed:
        raise ValueError("The shared OOF cache is incomplete or belongs to another study specification.")
    if (report["model_library_version"] != importlib.metadata.version(settings["trial"]["model"]) or
            report["sklearn_version"] != importlib.metadata.version("scikit-learn") or
            report["fitting_implementation_sha256"] != file_digest(Path(__file__).with_name("study_training.py"))):
        raise ValueError("Shared OOF software/implementation changed; create a fresh study cache.")
    if file_digest(cache / "probabilities.npy") != report["probability_sha256"] or file_digest(cache / "rows.npz") != report["rows_sha256"]:
        raise ValueError("Shared OOF array or row manifest was modified.")
    with np.load(cache / "rows.npz") as rows:
        if not np.array_equal(rows["row_ids"], fitting) or not np.array_equal(rows["y"], data.y[fitting]) or set(rows["fold_ids"]) != set(range(1, 6)):
            raise ValueError("Shared OOF cache row/fold alignment changed.")
    p = np.load(cache / "probabilities.npy", mmap_mode="r")
    try:
        if p.shape != (len(fitting), config.num_classes): raise ValueError("Shared OOF shape mismatch.")
        validate_probabilities(p)
    finally:
        close_memmap(p)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("probabilities.npy", "rows.npz"):
        try: os.link(cache / name, destination / name)
        except OSError: shutil.copy2(cache / name, destination / name)
    write_report(destination / "report.json", {**report, "shared_cache": str(cache.resolve()), "immutable_after_completion": True})


def run_stacking_study(source: Path, output: Path, *, seed=42, max_latency_ratio=5., fixed_conditions=None, meta_max_iter=2000, meta_solver="lbfgs"):
    if type(meta_max_iter) is not int or meta_max_iter < 1:
        raise ValueError("The meta optimizer needs a positive integer iteration budget.")
    if meta_solver not in ("lbfgs", "newton-cg"):
        raise ValueError("Use an explicitly supported multinomial L2 solver.")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    original = read(source / "report.json")
    if original["status"] != "complete":
        raise ValueError("Finish all base trials before deciding whether to stack.")
    config = load_study_config(source / "config.json")
    if object_digest(read(source / "config.json")) != original["config_sha256"]:
        raise ValueError("Base study configuration changed.")
    selection = read(source / f"seed_{seed}/selection.json")
    if object_digest(selection) != original["selections"][str(seed)]["sha256"]:
        raise ValueError("Base selection changed after freezing.")
    diversity = selection["complexity_evidence"]["diversity"]
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "checking_evidence", "source_study": str(source.resolve()), "seed": seed,
              "diversity": diversity, "test_file_read": False, "validation_results": [],
              "upstream_feature_selection": original["upstream_feature_selection"]}
    names = [selection["best_xgboost"], "catboost"]
    costs = {r["candidate"]: r["inference_us_per_sample"] for r in original["validation_results"] if r["seed"] == seed}
    ratio = (sum(costs[n] for n in names) / costs[names[0]]) if all(n in costs for n in names) else None
    report["latency_gate"] = {"ratio_estimate": ratio, "maximum": max_latency_ratio,
                               "scope": "Sum of measured base throughput costs, used only for compute-cost gating."}
    if fixed_conditions is None and (not diversity or not diversity["complementarity_present"] or ratio is None or ratio > max_latency_ratio):
        report.update(status="skipped", reason="Complementary target corrections or the declared base latency budget were not demonstrated.")
        write_report(output / "report.json", report)
        return report
    write_report(output / "report.json", report)
    started = time.perf_counter()
    mappings = []
    try:
        data = load_study_data(config)
        classes = data.encoder.classes_
        benign = list(classes).index("Benign" if config.num_classes == 8 else "BenignTraffic")
        with np.load(source / f"seed_{seed}/splits.npz") as split:
            fitting, tuning, validation = (split[k].copy() for k in ("fitting", "tuning", "validation"))
        for name in names:
            ensure_base_oof(data, fitting, selection["trials"][name], config, seed, output / "base_oof" / name)
        # Construct meta features in bounded blocks to avoid N x 2K copies.
        a = np.lib.format.open_memmap(output / "meta_fit_features.npy", mode="w+", dtype="float32",
                                      shape=(len(fitting), 2 * config.num_classes)); mappings.append(a)
        base_oof = [np.load(output / "base_oof" / n / "probabilities.npy", mmap_mode="r") for n in names]
        try:
            for begin in range(0, len(fitting), config.batch_size):
                end = min(len(fitting), begin + config.batch_size)
                a[begin:end] = log_probability_features([p[begin:end] for p in base_oof])
        finally:
            for p in base_oof: close_memmap(p)
        b_probabilities = []
        for name in names:
            base = load_multiclass_bundle(source / f"seed_{seed}/screening" / name / "model.joblib")
            b_probabilities.append(trial_probabilities(base, data.x.iloc[tuning], config))
            del base
            gc.collect()
        b = log_probability_features(b_probabilities)
        del b_probabilities
        ya, yb = data.y[fitting], data.y[tuning]
        meta_oof = None
        if fixed_conditions is None:
            meta_oof = crossfit_meta(a, ya, b, yb, seed=seed, folder=output / "meta_oof", threads=config.n_jobs, max_iter=meta_max_iter, solver=meta_solver)
            mappings.append(meta_oof)
            with np.load(source / f"seed_{seed}/screening/paper_fixed/tuning/predictions.npz") as reference:
                np.testing.assert_array_equal(reference["y"], yb)
                constraints = reference_constraints(yb, reference["predicted"], config.num_classes,
                                                    benign_id=benign, protected_ids=protected_ids(classes, config))
            bias, diagnostics = tune_biases(yb, meta_oof, grid=(-2., -1., -.5, 0., .5, 1., 2.))
            constrained, protected_diagnostics = tune_constrained_biases(yb, meta_oof, constraints=constraints,
                grid=(-3., -2., -1., -.5, 0., .5, 1., 2., 3.), starts=(bias,))
            joint, joint_report = tune_joint_biases(yb, meta_oof, constraints,
                coordinate_start(constrained, protected_diagnostics), grid=(-3., -2., -1., -.5, 0., .5, 1., 2., 3.))
            write_report(output / "bias_unrestricted.json", diagnostics)
            write_report(output / "bias_protected.json", protected_diagnostics)
            write_report(output / "bias_joint_protected.json", joint_report)
            if joint is not None and (constrained is None or joint_report["tuned"]["macro_f1"] > protected_diagnostics["tuned"]["macro_f1"] + 1e-12):
                constrained = joint
                report["protected_bias_search"] = "joint"
            else:
                report["protected_bias_search"] = "coordinate"
            conditions = {"stack_argmax": np.zeros(config.num_classes), "stack_bias": bias}
            if constrained is not None: conditions["stack_protected_bias"] = constrained
        else:
            import re
            conditions = {name: np.asarray(value, dtype=float) for name, value in fixed_conditions.items()}
            if not conditions or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", n) or v.shape != (config.num_classes,) or not np.isfinite(v).all() for n, v in conditions.items()):
                raise ValueError("Invalid frozen confirmation decisions.")
            report["confirmation_reuses_frozen_gate_and_biases"] = True
        frozen = {"base_names": names, "conditions": {n: v.tolist() for n, v in conditions.items()},
                  "test_file_read": False, "validation_used_for_tuning": False,
                  "bias_tuning_scope": "Cross-fitted meta predictions on training-only B, with B rows excluded from base fitting; base parameter selection used B labels.",
                  "meta_class_cap": 10_000, "meta_C": 1., "meta_max_iter": meta_max_iter, "meta_solver": meta_solver}
        write_report(output / "selection.json", frozen)
        report["selection_sha256"] = object_digest(frozen)
        meta, meta_report = fit_meta(a, ya, b, yb, seed=seed, threads=config.n_jobs, max_iter=meta_max_iter, solver=meta_solver)
        write_report(output / "meta_training.json", meta_report)
        costs = {
            "base_oof_fit_seconds": sum(f["training_seconds"] for name in names for f in read(output / "base_oof" / name / "report.json")["folds"]),
            "base_development_fit_seconds": sum(read(source / f"seed_{seed}/finalists" / name / "training.json")["training_seconds"] for name in names),
            "meta_final_fit_seconds": meta_report["training_seconds"],
            "meta_calibration_fit_seconds": sum(f["training_seconds"] for f in read(output / "meta_oof/report.json")["folds"]) if fixed_conditions is None else 0.}
        costs["total_fit_seconds"] = sum(costs.values())
        report["training_costs"] = costs
        for p in mappings: close_memmap(p)
        mappings.clear()
        del a, b, meta_oof
        gc.collect()
        bases = tuple(load_multiclass_bundle(source / f"seed_{seed}/finalists" / name / "model.joblib") for name in names)
        bundle = StackBundle(bases, meta, n_jobs=config.n_jobs)
        x_validation, y_validation = data.x.iloc[validation], data.y[validation]
        p_validation = bundle.predict_proba(x_validation)
        runs = []
        for _ in range(config.latency_repeats):
            before = time.perf_counter(); bundle.predict_proba(x_validation); runs.append(time.perf_counter() - before)
        report["probability_inference_timing"] = {"seconds_runs": runs, "median_us_per_sample": float(np.median(runs) * 1e6 / len(x_validation)),
                                                  "full_cohort_warmup": True, "inference_device": "cpu",
                                                  "scope": "All base predictions plus log features and meta predict_proba; decision timed separately per condition."}
        with np.load(source / f"seed_{seed}/finalists/paper_fixed/validation/predictions.npz") as reference:
            np.testing.assert_array_equal(reference["y"], y_validation)
            constraints = reference_constraints(y_validation, reference["predicted"], config.num_classes,
                                                benign_id=benign, protected_ids=protected_ids(classes, config))
        for name, decision_bias in conditions.items():
            predicted = predict_with_bias(p_validation, decision_bias)
            row = _score(output / name / "validation", y_validation, predicted, classes, "validation", name, seed, constraints)
            times = []
            for _ in range(config.latency_repeats):
                before = time.perf_counter(); predict_with_bias(p_validation, decision_bias); times.append(time.perf_counter() - before)
            row.update(decision_us_per_sample=float(np.median(times) * 1e6 / len(x_validation)),
                       shared_probability_inference_us=report["probability_inference_timing"]["median_us_per_sample"],
                       timing_note="Probability and decision timings are separate measurements, not a paired end-to-end latency measurement.",
                       training_seconds=costs["total_fit_seconds"])
            report["validation_results"].append(row)
            bundle.decision_bias = decision_bias
            save_joblib_atomic(output / name / "model.joblib", bundle)
        if object_digest(read(output / "selection.json")) != report["selection_sha256"]:
            raise RuntimeError("Stack biases changed during validation.")
        write_table(output / "validation_comparison.csv", pd.DataFrame(report["validation_results"]))
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for p in mappings: close_memmap(p)
        report["seconds"] = time.perf_counter() - started
        write_report(output / "report.json", report)
    return report
