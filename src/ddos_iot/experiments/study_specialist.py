"""Evidence-gated Benign/minority specialist, selected using tuning data only."""

from dataclasses import asdict
import gc
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.attack_families import RAW_TO_FAMILY
from ddos_iot.multiclass import load_multiclass_bundle, predict_probabilities
from ddos_iot.weighting import balanced_class_weights
from .recall_constraints import reference_constraints
from .storage import object_digest, write_report, write_table
from .study import _score
from .study_config import load_study_config
from .study_data import load_study_data, protected_ids
from .study_models import SpecialistBundle, route_specialist

LOGGER = logging.getLogger(__name__)


def specialist_group(classes, y, predicted, targets, benign, *, expand=False, minimum_extra_fraction=.1):
    """Expand only from cached training-tuning mistakes, never validation rows."""
    if not np.isfinite(minimum_extra_fraction) or not 0 < minimum_extra_fraction <= 1:
        raise ValueError("Declare an extra-family error fraction in (0, 1].")
    classes = list(classes)
    group = {benign, *targets}
    errors = (y != predicted) & np.isin(y, targets)
    total = int(errors.sum())
    families = [RAW_TO_FAMILY.get(label, label) for label in classes]
    outside = errors & ~np.isin(predicted, sorted(group))
    counts = {}
    for label_id in predicted[outside]:
        family = families[int(label_id)]
        counts[family] = counts.get(family, 0) + 1
    added = sorted(f for f, n in counts.items() if expand and n / max(1, total) >= minimum_extra_fraction)
    group.update(i for i, family in enumerate(families) if family in added)
    ids = np.asarray(sorted(group))
    return ids, {"target_errors": total, "outside_group_error_counts_by_family": counts,
                 "added_families": added, "minimum_extra_error_fraction": minimum_extra_fraction,
                 "target_errors_inside_routing_group": int((errors & np.isin(predicted, ids)).sum()),
                 "selection_scope": "Training-only tuning predictions; validation/test labels do not select the routing group."}


def fit_specialist(x, y, group, config, *, seed, power, rounds=300, depth=6, tuning=None):
    group = np.asarray(group, dtype=int)
    mapping = np.full(config.num_classes, -1, dtype=int); mapping[group] = np.arange(len(group))
    mask = mapping[y] >= 0
    local_y = mapping[y[mask]]
    weights = balanced_class_weights(local_y, len(group), power)
    model = XGBClassifier(n_estimators=rounds, max_depth=depth, learning_rate=.1, objective="multi:softprob",
                          num_class=len(group), eval_metric="mlogloss", tree_method="hist", max_bin=256,
                          device="cuda:0" if config.device == "gpu" else "cpu", n_jobs=config.n_jobs,
                          random_state=seed, **({"early_stopping_rounds": config.early_stopping_rounds} if tuning is not None else {}))
    kwargs = {"sample_weight": weights[local_y], "verbose": False}
    if tuning is not None:
        tx, ty = tuning
        selected = mapping[ty] >= 0
        kwargs.update(eval_set=[(tx.iloc[np.flatnonzero(selected)], mapping[ty[selected]])],
                      sample_weight_eval_set=[weights[mapping[ty[selected]]]])
    started = time.perf_counter()
    model.fit(x.iloc[np.flatnonzero(mask)], local_y, **kwargs)
    fitted_rounds = int(model.best_iteration) + 1 if tuning is not None else rounds
    return model, {"training_rows": int(mask.sum()), "rounds": fitted_rounds,
                   "weight_power": power, "depth": depth, "group_ids": group.tolist(),
                   "training_seconds": time.perf_counter() - started}


def _predict_routed(base_p, specialist, x, group, config):
    routed = np.isin(base_p.argmax(axis=1), group)
    local = predict_probabilities("xgboost", specialist, x.iloc[np.flatnonzero(routed)], config.batch_size,
                                  device=config.device) if routed.any() else np.empty((0, len(group)))
    return route_specialist(base_p, local, group)


def run_specialist_study(source: Path, output: Path, *, seed=42, minimum_error_fraction=.5, expanded_fixed=False):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    original = read(source / "report.json")
    if original["status"] != "complete":
        raise ValueError("Finish the base study before assessing specialist evidence.")
    config = load_study_config(source / "config.json")
    if object_digest(read(source / "config.json")) != original["config_sha256"]:
        raise ValueError("Base study configuration changed.")
    selection = read(source / f"seed_{seed}/selection.json")
    if object_digest(selection) != original["selections"][str(seed)]["sha256"]:
        raise ValueError("Base selection changed.")
    if not np.isfinite(minimum_error_fraction) or not 0 <= minimum_error_fraction <= 1:
        raise ValueError("Declare a reachable-error fraction in [0, 1].")
    evidence = dict(selection["complexity_evidence"]["specialist"])
    declared_classes = read(source / "data_report.json")["classes"]
    declared_benign = declared_classes.index("Benign" if config.num_classes == 8 else "BenignTraffic")
    declared_targets = [declared_classes.index(c) for c in selection["protected_labels"]]
    with np.load(source / f"seed_{seed}/screening" / selection["best_xgboost"] / "tuning/predictions.npz") as cached:
        declared_group, group_report = specialist_group(declared_classes, cached["y"], cached["predicted"],
            declared_targets, declared_benign, expand=expanded_fixed)
        evidence.update(group_report)
    fraction = evidence["target_errors_inside_routing_group"] / max(1, evidence["target_errors"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "checking_evidence", "source_study": str(source.resolve()), "seed": seed,
              "gate": {"minimum_fraction": minimum_error_fraction, "observed_fraction": fraction, "evidence": evidence,
                       "rule": "At least half of target errors must be reachable by the Benign/target routing group; minority-to-Benign and within-target confusions both count."},
              "tuning_results": [], "validation_results": [], "test_file_read": False,
              "variant": "expanded group plus reference capacity (joint variant)" if expanded_fixed else "original group, shallow early stopping",
              "upstream_feature_selection": original["upstream_feature_selection"]}
    if not evidence["target_errors_inside_routing_group"] or fraction < minimum_error_fraction:
        report.update(status="skipped", reason="Target errors do not meet the predeclared reachable-error gate.")
        write_report(output / "report.json", report)
        return report
    write_report(output / "report.json", report)
    started = time.perf_counter()
    try:
        data = load_study_data(config)
        classes = data.encoder.classes_
        benign = list(classes).index("Benign" if config.num_classes == 8 else "BenignTraffic")
        if list(classes) != declared_classes:
            raise ValueError("Specialist class order changed since the group was frozen.")
        group = declared_group
        with np.load(source / f"seed_{seed}/splits.npz") as split:
            fitting, tuning, validation = (split[k].copy() for k in ("fitting", "tuning", "validation"))
        x_fit, y_fit = data.x.iloc[fitting], data.y[fitting]
        x_tune, y_tune = data.x.iloc[tuning], data.y[tuning]
        name = selection["best_xgboost"]
        base = load_multiclass_bundle(source / f"seed_{seed}/screening" / name / "model.joblib")
        base_p = predict_probabilities(base.name, base.model, x_tune, config.batch_size, device=config.device)
        with np.load(source / f"seed_{seed}/screening/paper_fixed/tuning/predictions.npz") as reference:
            np.testing.assert_array_equal(reference["y"], y_tune)
            constraints = reference_constraints(y_tune, reference["predicted"], config.num_classes,
                                                benign_id=benign, protected_ids=protected_ids(classes, config))
        rounds = {}
        for power in (.75, 1., 1.25):
            candidate = f"specialist_{round(100 * power):03d}"
            LOGGER.info("Fitting %s on %s-class specialist group", candidate, len(group))
            specialist, training = fit_specialist(x_fit, y_fit, group, config, seed=seed, power=power,
                depth=10 if expanded_fixed else 6, rounds=500 if expanded_fixed else 300,
                tuning=None if expanded_fixed else (x_tune, y_tune))
            p, routed = _predict_routed(base_p, specialist, x_tune, group, config)
            row = _score(output / candidate / "tuning", y_tune, p.argmax(axis=1), classes, "tuning", candidate, seed, constraints)
            row.update(weight_power=power, training_seconds=training["training_seconds"],
                       routed_rows=int(routed.sum()),
                       missed_true_target_rows_by_gate=int((np.isin(y_tune, declared_targets) & ~routed).sum()),
                       routed_rows_from_outside_group=int((~np.isin(y_tune, group) & routed).sum()))
            report["tuning_results"].append(row); rounds[candidate] = training["rounds"]
            write_report(output / candidate / "training.json", training)
            del specialist, p
            gc.collect()
        best = max(report["tuning_results"], key=lambda r: r["macro_f1"])
        guarded_rows = [r for r in report["tuning_results"] if r["reference_constraints_pass"]]
        guarded = max(guarded_rows, key=lambda r: r["macro_f1"])["candidate"] if guarded_rows else None
        names = list(dict.fromkeys([best["candidate"], *([guarded] if guarded else [])]))
        frozen = {"base_candidate": name, "best_macro": best["candidate"], "guarded_macro": guarded,
                  "candidates": {r["candidate"]: {"weight_power": r["weight_power"], "rounds": rounds[r["candidate"]],
                                  "depth": 10 if expanded_fixed else 6} for r in report["tuning_results"]},
                  "validation_candidates": names, "group_ids": group.tolist(), "validation_used_for_selection": False,
                  "group_selection": group_report, "expanded_fixed": expanded_fixed,
                  "test_file_read": False}
        write_report(output / "selection.json", frozen)
        report["selection_sha256"] = object_digest(frozen)
        del base, base_p, x_fit, x_tune
        gc.collect()
        development = np.sort(np.concatenate([fitting, tuning]))
        x_train, y_train = data.x.iloc[development], data.y[development]
        x_validation, y_validation = data.x.iloc[validation], data.y[validation]
        with np.load(source / f"seed_{seed}/finalists/paper_fixed/validation/predictions.npz") as reference:
            np.testing.assert_array_equal(reference["y"], y_validation)
            constraints = reference_constraints(y_validation, reference["predicted"], config.num_classes,
                                                benign_id=benign, protected_ids=protected_ids(classes, config))
        for candidate in names:
            settings = frozen["candidates"][candidate]
            base = load_multiclass_bundle(source / f"seed_{seed}/finalists" / name / "model.joblib")
            specialist, training = fit_specialist(x_train, y_train, group, config, seed=seed,
                                                   power=settings["weight_power"], rounds=settings["rounds"], depth=settings["depth"])
            specialist.set_params(device="cpu")
            bundle = SpecialistBundle(base, specialist, group)
            save_joblib_atomic(output / candidate / "model.joblib", bundle)
            write_report(output / candidate / "refit_training.json", training)
            predicted = bundle.encoder.transform(bundle.predict(x_validation))
            times = []
            for _ in range(config.latency_repeats):
                before = time.perf_counter(); bundle.predict(x_validation); times.append(time.perf_counter() - before)
            row = _score(output / candidate / "validation", y_validation, predicted, classes, "validation", candidate, seed, constraints)
            row.update(inference_us_per_sample=float(np.median(times) * 1e6 / len(x_validation)),
                       inference_seconds_runs=times, full_cohort_warmup=True, inference_device="cpu",
                       training_seconds=training["training_seconds"] + read(source / f"seed_{seed}/finalists" / name / "training.json")["training_seconds"],
                       specialist_training_seconds=training["training_seconds"],
                       latency_scope="Complete base prediction and routing, specialist prediction and decision; published features, batched CPU.")
            report["validation_results"].append(row)
            del bundle, base, specialist
            gc.collect()
        if object_digest(read(output / "selection.json")) != report["selection_sha256"]:
            raise RuntimeError("Specialist choices changed during validation.")
        write_table(output / "tuning_comparison.csv", pd.DataFrame(report["tuning_results"]))
        write_table(output / "validation_comparison.csv", pd.DataFrame(report["validation_results"]))
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(output / "report.json", report)
    return report
