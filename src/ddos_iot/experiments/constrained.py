"""Retune completed experiments from cached OOF predictions, then validate once."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle, predict_probabilities, read_schema
from .config import config_from_dict
from .decisions import predict_with_bias, validate_probabilities
from .evaluation import save_scores
from .recall_constraints import (
    constraint_statistics, decision_statistics, reference_constraints, tune_constrained_biases,
)
from .storage import close_memmap, file_digest, object_digest, write_report, write_table

LOGGER = logging.getLogger(__name__)
DEFAULT_GRID = (-4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4.)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _oof(folder: Path, expected_y: np.ndarray, expected_ids: np.ndarray,
         expected_folds: np.ndarray | None, folds: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    report = _read(folder / "report.json")
    if report["status"] != "complete" or report["row_count"] != len(expected_y):
        raise ValueError(f"Incomplete OOF cache: {folder}")
    with np.load(folder / "rows.npz", allow_pickle=False) as rows:
        if not np.array_equal(rows["row_ids"], expected_ids) or not np.array_equal(rows["y"], expected_y):
            raise ValueError(f"OOF row/label alignment mismatch: {folder}")
        fold_ids = rows["fold_ids"].copy()
    if set(np.unique(fold_ids)) != set(range(1, folds + 1)):
        raise ValueError(f"Incomplete OOF fold coverage: {folder}")
    if expected_folds is not None and not np.array_equal(fold_ids, expected_folds):
        raise ValueError(f"OOF folds differ from reference: {folder}")
    p = np.load(folder / "probabilities.npy", mmap_mode="r", allow_pickle=False)
    if p.shape != (len(expected_y), k):
        close_memmap(p)
        raise ValueError(f"OOF probability shape mismatch: {folder}")
    return p, fold_ids


def _validation_features(config, row_ids: np.ndarray, count: int) -> pd.DataFrame:
    if len(row_ids) == 0 or np.any(np.diff(row_ids) <= 0) or row_ids[0] < 0 or row_ids[-1] >= count:
        raise ValueError("Invalid saved validation row IDs.")
    membership = np.zeros(count, dtype=bool)
    membership[row_ids] = True
    features = read_schema(config.train_input_path, config.label_column)
    parts = []
    for chunk in pd.read_csv(config.train_input_path, usecols=features, dtype="float32", chunksize=config.chunk_size):
        part = chunk.loc[membership[chunk.index.to_numpy()]]
        if len(part):
            parts.append(part)
    x = pd.concat(parts)
    if not np.array_equal(x.index.to_numpy(), row_ids):
        raise ValueError("Validation cohort differs from saved row IDs.")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Saved validation rows contain invalid features.")
    return x


def run_constrained_tuning(run_path: Path, output_path: Path, *,
                           protected_classes: tuple[str, ...] = ("WebBased", "BruteForce"),
                           grid: tuple[float, ...] = DEFAULT_GRID, passes: int = 3) -> dict:
    """Preserve reference OOF recalls/FPR; choose by OOF, then assess validation.

    All candidate biases and the chosen candidate are frozen before reading any
    validation labels/features. Existing experiment artifacts remain unchanged.
    """
    payload = _read(run_path / "config.json")
    config = config_from_dict(payload)
    original = _read(run_path / "experiment_report.json")
    original_selection = _read(run_path / "selection.json")
    if original["status"] != "complete":
        raise ValueError("Constrained tuning requires a completed experiment without failed candidates.")
    if object_digest(payload) != original["config_sha256"] or object_digest(original_selection) != original["selection_sha256"]:
        raise ValueError("The original frozen configuration or selection was modified.")
    mapping = _read(run_path / "label_mapping.json")
    classes = [mapping[str(i)] for i in range(config.num_classes)]
    if not protected_classes or len(set(protected_classes)) != len(protected_classes):
        raise ValueError("Provide distinct protected attack classes.")
    if any(c not in classes for c in protected_classes):
        raise ValueError(f"Protected class names must occur in {classes}.")
    benign = "Benign" if config.num_classes == 8 else "BenignTraffic"
    protected_ids = tuple(classes.index(c) for c in protected_classes)
    benign_id = classes.index(benign)
    if benign_id in protected_ids:
        raise ValueError("Benign cannot be a protected attack class.")
    if not grid or not np.isfinite(grid).all() or type(passes) is not int or passes < 1:
        raise ValueError("Invalid search grid/passes.")
    output_path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "tuning_oof", "source_run": str(run_path.resolve()),
              "config_sha256": original["config_sha256"],
              "original_selection_sha256": original["selection_sha256"],
              "training_source_sha256": original_selection["training_source_sha256"],
              "reference_candidate": config.reference_name, "reference_decision": "argmax",
              "protected_classes": protected_classes, "classes": classes,
              "policy": "No recall loss or Benign FPR increase relative to reference on the same OOF rows.",
              "grid": grid, "passes": passes, "constraints_by_seed": {}, "oof_results": [],
              "validation_results": [], "test_file_read": False, "models_trained": 0,
              "validation_used_for_tuning": False, "validation_used_for_selection": False,
              "selection_rule": "Highest mean constrained OOF Macro-F1 across all seeds; declaration order breaks ties.",
              "score_warning": "OOF scores optimized on these labels are diagnostics. Validation measures transfer; it does not guarantee constraints."}
    write_report(output_path / "report.json", report)
    try:
        # Complete the entire OOF stage for every seed before validation access.
        for seed in config.seeds:
            source = run_path / f"seed_{seed}"
            with np.load(source / "splits.npz", allow_pickle=False) as rows:
                y, row_ids = rows["development_y"].copy(), rows["development_row_ids"].copy()
            p, fold_ids = _oof(source / config.reference_name / "oof", y, row_ids, None, config.folds, config.num_classes)
            try:
                validate_probabilities(p)
                constraints = reference_constraints(y, p.argmax(axis=1), config.num_classes,
                                                    benign_id=benign_id, protected_ids=protected_ids)
            finally:
                close_memmap(p)
            report["constraints_by_seed"][str(seed)] = constraints.as_dict()
            for candidate in config.candidates:
                LOGGER.info("%s seed %s: recall-constrained OOF search", candidate.name, seed)
                folder = source / candidate.name
                destination = output_path / f"seed_{seed}" / candidate.name
                destination.mkdir(parents=True)
                p, _ = _oof(folder / "oof", y, row_ids, fold_ids, config.folds, config.num_classes)
                tuning_started = time.perf_counter()
                try:
                    historical = np.asarray(_read(folder / "bias.json")["bias"])
                    bias, diagnostics = tune_constrained_biases(y, p, constraints=constraints,
                                                                grid=grid, passes=passes, starts=(historical,))
                    diagnostics.update(seconds=time.perf_counter() - tuning_started,
                                       probability_sha256=file_digest(folder / "oof" / "probabilities.npy"),
                                       row_manifest_sha256=file_digest(folder / "oof" / "rows.npz"),
                                       classes=classes)
                    write_report(destination / "bias.json", diagnostics)
                    row = {"candidate": candidate.name, "seed": seed, "feasible": bias is not None,
                           "macro_f1": diagnostics["tuned"]["macro_f1"],
                           "benign_false_positive_rate": diagnostics["tuned"]["benign_false_positive_rate"],
                           "seconds": diagnostics["seconds"]}
                    row.update({f"{name}_recall": diagnostics["tuned"]["recall"][cid]
                                for name, cid in zip(protected_classes, protected_ids)})
                    report["oof_results"].append(row)
                    LOGGER.info("%s: feasible=%s OOF Macro-F1=%.4f", candidate.name, row["feasible"], row["macro_f1"])
                finally:
                    close_memmap(p)
                    gc.collect()
                write_report(output_path / "report.json", report)
        oof = pd.DataFrame(report["oof_results"])
        eligible = []
        for candidate in config.candidates:
            rows = oof.loc[oof.candidate == candidate.name]
            if set(rows.seed) == set(config.seeds) and rows.feasible.all():
                eligible.append((float(rows.macro_f1.mean()), candidate))
        if not eligible:
            raise RuntimeError("No candidate meets constraints in all seeds, including the reference fallback.")
        # max preserves candidate declaration order for exact ties.
        score, winner = max(eligible, key=lambda pair: pair[0])
        selection = {"candidate": asdict(winner), "decision": "constrained_bias", "oof_macro_f1_mean": score,
                     "selection_rule": report["selection_rule"], "config_sha256": report["config_sha256"],
                     "validation_used_for_selection": False, "test_file_read": False,
                     "stage": "OOF-selected research candidate; validation and multiseed confirmation required",
                     "seeds": {str(seed): {"bias": _read(output_path / f"seed_{seed}" / winner.name / "bias.json")["bias"],
                                            "constraints": report["constraints_by_seed"][str(seed)]}
                               for seed in config.seeds}}
        write_report(output_path / "selection.json", selection)
        write_table(output_path / "oof_comparison.csv", oof)
        report.update(status="validating_frozen_biases", selected_candidate=winner.name,
                      selection_sha256=object_digest(selection))
        write_report(output_path / "report.json", report)
        LOGGER.info("Frozen %s using OOF only; beginning validation", winner.name)
        if file_digest(config.train_input_path) != report["training_source_sha256"]:
            raise ValueError("Training source changed since the original experiment.")
        for seed in config.seeds:
            source = run_path / f"seed_{seed}"
            with np.load(source / "splits.npz", allow_pickle=False) as rows:
                row_ids, y = rows["validation_row_ids"].copy(), rows["validation_y"].copy()
            x = _validation_features(config, row_ids, _read(run_path / "data_report.json")["raw_rows"])
            with np.load(source / config.reference_name / "validation" / "argmax" / "predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["y"], y):
                    raise ValueError("Reference validation labels differ from the frozen split.")
                validation_constraints = reference_constraints(y, saved["predicted"], config.num_classes,
                                                               benign_id=benign_id, protected_ids=protected_ids)
            for candidate in config.candidates:
                destination = output_path / f"seed_{seed}" / candidate.name
                diagnostics = _read(destination / "bias.json")
                if diagnostics["bias"] is None:
                    continue
                LOGGER.info("%s seed %s: validating frozen constrained bias", candidate.name, seed)
                bundle = load_multiclass_bundle(source / candidate.name / "model.joblib")
                if list(bundle.encoder.classes_) != classes:
                    raise ValueError("Model class mapping differs from OOF probability columns.")
                p = predict_probabilities(bundle.name, bundle.model, bundle.features.transform(x), config.prediction_batch_size)
                with np.load(source / candidate.name / "validation" / "argmax" / "predictions.npz", allow_pickle=False) as saved:
                    if not np.array_equal(saved["y"], y) or not np.array_equal(p.argmax(axis=1), saved["predicted"]):
                        raise ValueError("Reloaded model does not reproduce original validation predictions.")
                predicted = predict_with_bias(p, np.asarray(diagnostics["bias"]))
                metrics = save_scores(destination / "validation", np.asarray(classes), y, predicted,
                                      split="validation", role="Evaluation after OOF-only constraints, bias tuning, and candidate selection were frozen.")
                stats = decision_statistics(y, predicted, config.num_classes)
                checked = constraint_statistics(stats, validation_constraints)
                row = {**metrics, "candidate": candidate.name, "seed": seed,
                       "decision": "constrained_bias", "selected_by_oof": candidate.name == winner.name,
                       "validation_reference_constraints_pass": checked["feasible"],
                       "validation_recall_constraints_pass": checked["recall_constraints_pass"],
                       "validation_fpr_constraint_pass": checked["fpr_constraint_pass"]}
                row.update({f"{name}_recall": stats["recall"][cid]
                            for name, cid in zip(protected_classes, protected_ids)})
                write_report(destination / "validation" / "metrics.json", row)
                report["validation_results"].append(row)
                del bundle, p, predicted
                gc.collect()
                write_report(output_path / "report.json", report)
            del x
        write_table(output_path / "validation_comparison.csv", pd.DataFrame(report["validation_results"]))
        selected_rows = [r for r in report["validation_results"] if r["selected_by_oof"]]
        if {r["seed"] for r in selected_rows} != set(config.seeds):
            raise RuntimeError("Selected validation results are incomplete.")
        report.update(status="complete", selected_validation_constraints_pass=all(
            r["validation_reference_constraints_pass"] for r in selected_rows))
        if object_digest(_read(output_path / "selection.json")) != report["selection_sha256"]:
            raise ValueError("OOF selection changed during validation.")
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(output_path / "report.json", report)
    return report


def export_constrained_choice(output_path: Path) -> Path:
    """Export the OOF choice as a reusable development bundle, without refitting."""
    report = _read(output_path / "report.json")
    selection = _read(output_path / "selection.json")
    if report["status"] != "complete" or object_digest(selection) != report["selection_sha256"]:
        raise ValueError("A complete, unchanged OOF selection is required for export.")
    destination = output_path / "benchmark_models"
    destination.mkdir(exist_ok=False)
    source = Path(report["source_run"])
    for seed, settings in selection["seeds"].items():
        original_model = source / f"seed_{seed}" / selection["candidate"]["name"] / "model.joblib"
        bundle = load_multiclass_bundle(original_model)
        bundle.decision_bias = np.asarray(settings["bias"], dtype=float)
        folder = destination / f"seed_{seed}"
        save_joblib_atomic(folder / "model.joblib", bundle)
        write_report(folder / "metadata.json", {
            "source_model": str(original_model.resolve()), "source_model_sha256": file_digest(original_model),
            "selection_sha256": report["selection_sha256"], "candidate": selection["candidate"],
            "decision_bias": settings["bias"], "seed": int(seed), "classes": report["classes"],
            "training_scope": "Original development refit; not trained on full source or test data.",
            "stage": "Research operating point; paper-baseline reconciliation and multiseed confirmation pending.",
            "selected_validation_constraints_pass": report["selected_validation_constraints_pass"],
            "test_file_read": False, "models_trained": 0})
    return destination
