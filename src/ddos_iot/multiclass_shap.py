"""Post-hoc multiclass SHAP for saved bundles, without training or CV.

Native TreeSHAP outputs are normalized to (row, feature, class). The last
native contribution is the baseline, not a feature. Native explanations use
raw class scores; the optional permutation fallback explains probabilities.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ddos_iot.artifacts import atomic_output
from ddos_iot.multiclass import MulticlassBundle, load_multiclass_bundle

LOGGER = logging.getLogger(__name__)


def save_report(path: Path, report: dict[str, Any]) -> None:
    with atomic_output(path) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False).encode("utf-8"))


def sample_raw_features(path: Path, features: list[str], size: int, *,
                        max_scan_rows: int = 50_000, seed: int = 42,
                        chunk_size: int = 10_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Uniform priority sample of valid rows in the scanned prefix (or full CSV).

    Memory is bounded by a chunk plus the requested sample. Original zero-based
    data-row positions are preserved, even when invalid rows are excluded.
    """
    if size <= 0 or max_scan_rows < 0 or chunk_size <= 0:
        raise ValueError("Sample/chunk sizes must be positive; scan limit must be nonnegative.")
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(features) - set(header.columns))
    if missing:
        raise ValueError(f"Missing raw model features: {missing}")
    rng = np.random.default_rng(seed)
    kept = pd.DataFrame(columns=features, dtype="float32")
    keys = np.empty(0)
    scanned = invalid = 0
    with pd.read_csv(path, usecols=features, chunksize=chunk_size,
                     nrows=max_scan_rows or None) as reader:
        for chunk in reader:
            chunk = chunk[features].apply(pd.to_numeric, errors="coerce").astype("float32")
            chunk.index = np.arange(scanned, scanned + len(chunk), dtype=np.int64)
            scanned += len(chunk)
            valid = np.isfinite(chunk.to_numpy()).all(axis=1)
            invalid += int((~valid).sum())
            chunk = chunk.loc[valid]
            if chunk.empty:
                continue
            priorities = rng.random(len(chunk))
            joined = pd.concat([kept, chunk])
            all_keys = np.concatenate([keys, priorities])
            take = np.argsort(all_keys, kind="stable")[:size]
            kept, keys = joined.iloc[take], all_keys[take]
    if kept.empty:
        raise ValueError(f"No valid numeric rows found in {path} within the scan limit.")
    kept = kept.sort_index()
    return kept, {"path": str(path.resolve()), "requested_rows": size, "scanned_rows": scanned,
                  "invalid_rows_skipped": invalid, "sampled_rows": len(kept), "seed": seed,
                  "max_scan_rows": max_scan_rows,
                  "sampling_scope": "full_file" if not max_scan_rows or scanned < max_scan_rows else "prefix",
                  "note": "Random sample of valid scanned rows; not stratified and not guaranteed to cover rare classes."}


def class_probabilities(bundle: MulticlassBundle, data: pd.DataFrame, n_jobs: int) -> np.ndarray:
    if bundle.name == "catboost":
        value = bundle.model.predict_proba(data, task_type="CPU", thread_count=n_jobs)
    elif bundle.name == "lightgbm":
        value = bundle.model.predict_proba(data, num_threads=n_jobs)
    else:
        value = bundle.model.predict_proba(data)
    return np.asarray(value, dtype=float)


def native_contributions(bundle: MulticlassBundle, data: pd.DataFrame,
                         n_jobs: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all class contributions, baselines, and corresponding raw scores."""
    rows, features, classes = len(data), len(data.columns), len(bundle.encoder.classes_)
    if bundle.name == "xgboost":
        from xgboost import DMatrix

        booster = bundle.model.get_booster()
        matrix = DMatrix(data, nthread=n_jobs)
        best = getattr(bundle.model, "best_iteration", None)
        iterations = (0, best + 1) if best is not None else (0, 0)
        options = {"strict_shape": True, "iteration_range": iterations}
        contributions = booster.predict(matrix, pred_contribs=True, **options)
        raw = booster.predict(matrix, output_margin=True, **options)
    elif bundle.name == "lightgbm":
        booster = bundle.model.booster_
        contributions = booster.predict(data, pred_contrib=True, num_threads=n_jobs)
        # Dense input returns class blocks, each ending with its own baseline.
        contributions = np.asarray(contributions).reshape(rows, classes, features + 1)
        raw = booster.predict(data, raw_score=True, num_threads=n_jobs)
    elif bundle.name == "catboost":
        from catboost import Pool

        contributions = bundle.model.get_feature_importance(
            Pool(data), type="ShapValues", thread_count=n_jobs)
        raw = bundle.model.predict(data, prediction_type="RawFormulaVal", task_type="CPU", thread_count=n_jobs)
    else:
        raise ValueError(f"No native SHAP implementation for {bundle.name}.")
    contributions = np.asarray(contributions, dtype=float)
    if contributions.shape != (rows, classes, features + 1):
        raise ValueError(f"Unexpected native SHAP shape: {contributions.shape}")
    return contributions[:, :, :-1].transpose(0, 2, 1), contributions[:, :, -1], np.asarray(raw, dtype=float)


def check_additivity(values: np.ndarray, bases: np.ndarray, outputs: np.ndarray,
                      rows: int, features: int, classes: int) -> float:
    if values.shape != (rows, features, classes) or bases.shape != (rows, classes) or outputs.shape != (rows, classes):
        raise ValueError("SHAP class/sample/feature axes do not match the model.")
    if not all(np.isfinite(a).all() for a in (values, bases, outputs)):
        raise ValueError("SHAP produced nonfinite values.")
    reconstructed = bases + values.sum(axis=1)
    error = float(np.max(np.abs(reconstructed - outputs)))
    if not np.allclose(reconstructed, outputs, rtol=1e-4, atol=1e-4):
        raise ValueError(f"SHAP additivity check failed (maximum error {error:g}).")
    return error


def explain_bundle(bundle: MulticlassBundle, raw: pd.DataFrame, *, method: str = "auto",
                   background: pd.DataFrame | None = None, fallback_size: int = 20,
                   seed: int = 42, n_jobs: int = 2) -> dict[str, Any]:
    """Never call fit: reuse the saved feature transform and encoded class order."""
    if method not in ("auto", "native", "permutation") or fallback_size <= 0 or n_jobs <= 0:
        raise ValueError("Invalid explanation method, fallback size or CPU thread count.")
    classes = len(bundle.encoder.classes_)
    if not np.array_equal(bundle.model.classes_, np.arange(classes)):
        raise ValueError("Estimator class order does not match the saved label encoder.")
    if bundle.name == "xgboost":
        # Only this newly loaded in-memory estimator is changed, never its file.
        bundle.model.set_params(device="cpu", n_jobs=n_jobs)
    raw = raw[bundle.features.columns]
    transformed = bundle.features.transform(raw)
    reason = None
    if method != "permutation":
        try:
            values, bases, outputs = native_contributions(bundle, transformed, n_jobs)
            error = check_additivity(values, bases, outputs, len(raw), len(raw.columns), classes)
            shifted = outputs - outputs.max(axis=1, keepdims=True)
            probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            actual = class_probabilities(bundle, transformed, n_jobs)
            if actual.shape != probability.shape or not np.allclose(probability, actual, atol=1e-5, rtol=1e-4):
                raise ValueError("SHAP raw scores do not match the saved model's class probabilities.")
            return {"values": values, "base_values": bases, "outputs": outputs,
                    "probabilities": actual, "raw": raw, "transformed": transformed,
                    "method": f"{bundle.name}_native_tree_shap", "output_space": "raw_margin",
                    "max_additivity_error": error, "fallback_reason": None, "background_rows": 0}
        except Exception as exc:
            if method == "native" or background is None:
                raise RuntimeError("Native SHAP failed. Supply a separate training-background CSV to enable "
                                   f"the capped permutation fallback. Cause: {exc}") from exc
            reason = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Native SHAP failed; using bounded probability SHAP: %s", reason)
    if background is None:
        raise ValueError("Permutation SHAP requires an explicit training-background CSV.")
    import shap

    raw = raw.sample(n=min(len(raw), fallback_size), random_state=seed).sort_index()
    transformed = bundle.features.transform(raw)
    reference = bundle.features.transform(background)

    def predict(data: np.ndarray) -> np.ndarray:
        return class_probabilities(bundle, pd.DataFrame(data, columns=transformed.columns), n_jobs)

    masker = shap.maskers.Independent(reference, max_samples=len(reference))
    explainer = shap.Explainer(predict, masker, algorithm="permutation", seed=seed,
                               output_names=list(bundle.encoder.classes_))
    explanation = explainer(transformed, max_evals=2 * (2 * len(transformed.columns) + 1), silent=True)
    values = np.asarray(explanation.values, dtype=float)
    bases = np.asarray(explanation.base_values, dtype=float)
    outputs = predict(transformed.to_numpy())
    error = check_additivity(values, bases, outputs, len(raw), len(raw.columns), classes)
    return {"values": values, "base_values": bases, "outputs": outputs, "probabilities": outputs,
            "raw": raw, "transformed": transformed, "method": "permutation", "output_space": "probability",
            "max_additivity_error": error, "fallback_reason": reason, "background_rows": len(reference)}


def run_multiclass_shap(model_path: Path, input_file: Path, output_path: Path | None = None, *,
                        sample_size: int = 200, max_scan_rows: int = 50_000,
                        background_file: Path | None = None, background_size: int = 50,
                        fallback_size: int = 20, method: str = "auto", local_count: int = 5,
                        max_display: int = 15, class_label: str | None = None,
                        random_state: int = 42, n_jobs: int = 2, no_plots: bool = False,
                        expected_num_classes: int | None = None) -> dict[str, Any]:
    if min(sample_size, background_size, fallback_size, max_display, n_jobs) <= 0 or min(max_scan_rows, local_count) < 0:
        raise ValueError("Sizes/threads must be positive; scan limit and local count must be nonnegative.")
    if method not in ("auto", "native", "permutation"):
        raise ValueError("Method must be auto, native or permutation.")
    if method == "permutation" and background_file is None:
        raise ValueError("Permutation SHAP requires --background-file from training data.")
    if background_file is not None and background_file.resolve() == input_file.resolve():
        raise ValueError("Background must be a separate training CSV, not the explained CSV.")
    bundle = load_multiclass_bundle(model_path)
    classes = list(map(str, bundle.encoder.classes_))
    if expected_num_classes is not None and len(classes) != expected_num_classes:
        raise ValueError("Saved model does not match the selected classification task.")
    if class_label is not None and class_label not in classes:
        raise ValueError(f"Unknown class {class_label!r}; expected one of {classes}.")
    output = output_path or model_path.parent / "shap" / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    # A new directory is mandatory: no overwriting models, data, or past explanations.
    output.mkdir(parents=True, exist_ok=False)
    with model_path.open("rb") as model_file:
        fingerprint = hashlib.file_digest(model_file, "sha256").hexdigest()
    report: dict[str, Any] = {"status": "running", "model": bundle.name, "class_count": len(classes),
        "classes": classes, "model_path": str(model_path.resolve()), "input_file": str(input_file.resolve()),
        "output_path": str(output.resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "n_jobs": n_jobs, "training_performed": False, "requested_method": method,
        "model_sha256": fingerprint}
    started = time.perf_counter()
    try:
        save_report(output / "shap_report.json", report)
        LOGGER.info("Explaining saved %s (%s classes); no training or CV", bundle.name, len(classes))
        raw, sampling = sample_raw_features(input_file, bundle.features.columns, sample_size,
                                            max_scan_rows=max_scan_rows, seed=random_state)
        background, background_sampling = None, None
        if background_file is not None:
            background, background_sampling = sample_raw_features(background_file, bundle.features.columns,
                background_size, max_scan_rows=max_scan_rows, seed=random_state + 1)
        result = explain_bundle(bundle, raw, method=method, background=background, fallback_size=fallback_size,
                                seed=random_state, n_jobs=n_jobs)
        raw = result["raw"]
        values, bases, probabilities = result["values"], result["base_values"], result["probabilities"]
        predicted = probabilities.argmax(axis=1)
        with atomic_output(output / "shap_values.npz") as stream:
            np.savez_compressed(stream, values=values, base_values=bases, outputs=result["outputs"],
                probabilities=probabilities, raw_values=raw.to_numpy(),
                transformed_values=result["transformed"].to_numpy(), row_indices=raw.index.to_numpy(dtype=np.int64),
                feature_names=np.asarray(raw.columns, dtype=str), classes=np.asarray(classes, dtype=str))
        global_frame = pd.DataFrame({"feature": raw.columns, "mean_abs_shap": np.abs(values).mean(axis=(0, 2))})
        global_frame.sort_values("mean_abs_shap", ascending=False).to_csv(output / "shap_global_importance.csv", index=False)
        per_class = []
        local = []
        for k, label in enumerate(classes):
            for j, feature in enumerate(raw.columns):
                per_class.append({"class_id": k, "class": label, "feature": feature,
                                  "mean_abs_shap": float(np.abs(values[:, j, k]).mean()),
                                  "mean_shap": float(values[:, j, k].mean())})
            for i in range(min(local_count, len(raw))):
                for j in np.argsort(np.abs(values[i, :, k]))[::-1][:max_display]:
                    local.append({"sample_row": int(raw.index[i]), "class_id": k, "class": label,
                        "predicted_label": classes[predicted[i]], "class_probability": float(probabilities[i, k]),
                        "base_value": float(bases[i, k]), "model_output": float(result["outputs"][i, k]),
                        "feature": raw.columns[j], "raw_feature_value": float(raw.iloc[i, j]),
                        "model_feature_value": float(result["transformed"].iloc[i, j]),
                        "shap_value": float(values[i, j, k]), "output_space": result["output_space"]})
        pd.DataFrame(per_class).to_csv(output / "shap_class_importance.csv", index=False)
        if local:
            pd.DataFrame(local).to_csv(output / "shap_local_top_features.csv", index=False)
        sample = pd.DataFrame({"sample_row": raw.index, "predicted_label": np.asarray(classes)[predicted],
                               "predicted_probability": probabilities[np.arange(len(raw)), predicted]})
        sample.to_csv(output / "explained_samples.csv", index=False)
        plot_class = classes.index(class_label) if class_label is not None else int(np.bincount(predicted, minlength=len(classes)).argmax())
        report.update({key: result[key] for key in ("method", "output_space", "max_additivity_error", "fallback_reason", "background_rows")})
        report.update(sampling=sampling, background_sampling=background_sampling, explained_rows=len(raw),
                      feature_count=len(raw.columns), feature_names=list(raw.columns), local_rows=min(local_count, len(raw)),
                      plot_class=classes[plot_class], random_state=random_state,
                      reference="training tree statistics" if result["output_space"] == "raw_margin" else "supplied training background",
                      global_aggregation="Mean absolute SHAP over sampled rows and all output classes; not a balanced population estimate.",
                      feature_units="Model receives saved standardized features; plots display original feature units.",
                      versions={name: importlib.metadata.version(name) for name in (bundle.name, "numpy", "shap")})
        if not no_plots:
            from ddos_iot.multiclass_shap_plots import save_plots

            save_report(output / "shap_report.json", report)
            save_plots(output, plot_class, local_count, max_display)
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        save_report(output / "shap_report.json", report)
        LOGGER.info("Saved multiclass SHAP to %s", output)
        return report
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        save_report(output / "shap_report.json", report)
        raise
