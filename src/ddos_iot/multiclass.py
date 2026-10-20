"""34-label and eight-family XGBoost, LightGBM, and CatBoost experiments.

This module deliberately consumes raw labels, not binary processed data. Model
artifacts contain the fitted feature transformation and the shared label encoder.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import logging
import time
from collections import Counter
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ddos_iot.paths import configure_runtime_environment
from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.attack_families import (
    FAMILY_NAMES, RAW_TO_FAMILY, generated_split_report, map_family_labels,
    prepare_paper_split, proportional_quotas, scan_source_pool, validate_source_paths,
)
from ddos_iot.utils import write_json

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = ("xgboost", "lightgbm", "catboost")
NUM_CLASSES = 34
NUM_FEATURES = 46
GRID_PARAMETERS = ("n_estimators", "learning_rate", "max_depth")


def read_schema(path: Path, label_column: str) -> list[str]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    if label_column not in columns:
        raise ValueError(f"{path} has no {label_column!r} column. Use the original raw CSV.")
    features = [c for c in columns if c != label_column]
    if not features:
        raise ValueError(f"No feature columns in {path}")
    return features


def count_labels(path: Path, label_column: str, chunk_size: int, num_classes: int = NUM_CLASSES) -> Counter:
    counts: Counter = Counter()
    for chunk in pd.read_csv(path, usecols=[label_column], dtype={label_column: "string"},
                             chunksize=chunk_size):
        labels = chunk[label_column].str.strip()
        if num_classes == 8:
            labels = map_family_labels(labels)
        counts.update(labels[labels.notna() & labels.ne("")])
    return counts


def load_raw_split(path: Path, label_column: str, cap: int, chunk_size: int,
                   seed: int, num_classes: int = NUM_CLASSES) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read numeric flows; random priorities give an unbiased per-class reservoir.

The scan always covers the entire CSV, including late rare classes. Missing or
infinite feature values are removed; counts before/after cleaning are preserved.
Duplicate flows are retained (the existing split is kept unchanged).
"""
    features = read_schema(path, label_column)
    dtypes = {c: "float32" for c in features}
    dtypes[label_column] = "string"
    rng = np.random.default_rng(seed)
    raw_counts: Counter = Counter()
    valid_counts: Counter = Counter()
    reservoirs: dict[str, pd.DataFrame] = {}
    parts = []
    total = invalid = 0
    priority_column = "__sample_priority__"
    if priority_column in features:
        raise ValueError(f"Reserved input column: {priority_column}")
    LOGGER.info("Reading %s (sample cap per class: %s)", path, cap or "all")
    for index, chunk in enumerate(pd.read_csv(path, dtype=dtypes, chunksize=chunk_size)):
        total += len(chunk)
        labels = chunk[label_column].str.strip()
        if num_classes == 8:
            labels = map_family_labels(labels)
        chunk[label_column] = labels
        has_label = labels.notna() & labels.ne("")
        raw_counts.update(labels[has_label])
        valid = has_label.to_numpy(dtype=bool, na_value=False) & np.isfinite(chunk[features]).all(axis=1)
        invalid += int((~valid).sum())
        chunk = chunk.loc[valid].copy()
        valid_counts.update(chunk[label_column])
        if cap:
            chunk[priority_column] = rng.random(len(chunk))
            for label, group in chunk.groupby(label_column, sort=False):
                previous = reservoirs.get(label)
                candidates = group if previous is None else pd.concat([previous, group], ignore_index=True)
                reservoirs[label] = candidates.nsmallest(cap, priority_column).copy()
        elif len(chunk):
            parts.append(chunk)
        if (index + 1) % 4 == 0:
            LOGGER.info("Scanned %s rows; %s valid classes", f"{total:,}", len(valid_counts))
    if not valid_counts:
        raise ValueError(f"No valid numeric labeled rows in {path}")
    if cap:
        parts = [reservoirs[label].drop(columns=priority_column) for label in sorted(reservoirs)]
    data = pd.concat(parts, ignore_index=True)
    return data, {"path": str(path.resolve()), "raw_rows": total, "removed_invalid_rows": invalid,
                  "valid_rows": total - invalid, "selected_rows": len(data),
                  "raw_counts": dict(raw_counts), "valid_counts": dict(valid_counts),
                  "selected_counts": data[label_column].value_counts().to_dict(),
                  "max_rows_per_class": cap, "duplicates": "retained"}


def encode_labels(train_labels: pd.Series, test_labels: pd.Series,
                  num_classes: int = NUM_CLASSES) -> tuple[LabelEncoder, np.ndarray, np.ndarray]:
    encoder = LabelEncoder().fit(train_labels)
    if len(encoder.classes_) != num_classes:
        raise ValueError(f"Expected {num_classes} training classes, found {len(encoder.classes_)}. Use raw data with all labels.")
    if num_classes == 8 and set(encoder.classes_) != set(FAMILY_NAMES):
        raise ValueError("Eight-class training requires the paper's exact family names.")
    unknown = sorted(set(test_labels) - set(encoder.classes_))
    missing = sorted(set(encoder.classes_) - set(test_labels))
    if unknown or missing:
        raise ValueError(f"Test must cover the same {num_classes} labels. Unknown={unknown}; missing={missing}")
    return encoder, encoder.transform(train_labels), encoder.transform(test_labels)


def class_weights(y: np.ndarray, mode: str, num_classes: int = NUM_CLASSES) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError(f"Every training partition must contain all {num_classes} classes.")
    if mode == "none":
        return np.ones(num_classes, dtype=np.float32)
    return (len(y) / (num_classes * counts)).astype(np.float32)


def split_training(y: np.ndarray, validation_size: float, seed: int,
                   num_classes: int = NUM_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    if np.bincount(y, minlength=num_classes).min() < 2:
        raise ValueError("Need at least two valid training rows per class for stratified validation.")
    fit, valid = train_test_split(np.arange(len(y)), test_size=validation_size,
                                 stratify=y, random_state=seed)
    class_weights(y[fit], "none", num_classes)
    class_weights(y[valid], "none", num_classes)
    return fit, valid


@dataclass
class FeatureTransform:
    columns: list[str]
    scaler: StandardScaler
    scores: dict[str, float]
    gain_threshold: float | None = None
    method: str = "median"

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.columns) - set(data.columns))
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        values = data[self.columns].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Prediction features must be numeric and finite.")
        values = self.scaler.transform(values).astype(np.float32, copy=False)
        return pd.DataFrame(values, columns=self.columns, index=data.index)


def above_median_features(scores: dict[str, float]) -> tuple[list[str], float]:
    """Include all original features in the threshold, including zero gains and ties."""
    if len(scores) != NUM_FEATURES:
        raise ValueError(f"Median selection requires all 46 original features; received {len(scores)}.")
    gains = np.asarray(list(scores.values()), dtype=float)
    if not np.isfinite(gains).all() or np.any(gains < 0):
        raise ValueError("Feature gains must be finite and nonnegative.")
    threshold = float(np.median(gains))
    selected = [column for column, gain in scores.items() if gain > threshold]
    if not selected:
        raise ValueError("No feature has gain strictly above the median; check the training data.")
    return sorted(selected, key=lambda c: (-scores[c], c)), threshold


def fit_features(x: pd.DataFrame, y: np.ndarray, args: Any) -> FeatureTransform:
    """Select and standardize features using only the current fitting partition."""
    columns = list(x.columns)
    scores: dict[str, float] = {}
    threshold = None
    num_classes = args.num_classes
    if args.feature_selection == "median":
        if len(columns) != NUM_FEATURES:
            raise ValueError(f"Median selection requires all 46 original features; received {len(columns)}.")
        from lightgbm import LGBMClassifier

        indices = np.arange(len(y))
        if len(indices) > args.selector_sample_size:
            # Reserve one example per class before uniformly sampling the rest.
            rng = np.random.default_rng(args.random_state)
            reserved = np.array([rng.choice(np.flatnonzero(y == k)) for k in range(num_classes)])
            if args.selector_sample_size < num_classes:
                raise ValueError(f"selector-sample-size must be at least {num_classes}.")
            rest = np.setdiff1d(indices, reserved)
            indices = np.concatenate([reserved, rng.choice(rest, args.selector_sample_size - num_classes,
                                                           replace=False)])
        selector_y = y[indices]
        selector = LGBMClassifier(objective="multiclass", num_class=num_classes, n_estimators=50,
                                  num_leaves=15, max_bin=63, importance_type="gain", verbosity=-1,
                                  n_jobs=args.n_jobs, random_state=args.random_state)
        LOGGER.info("Computing gain for all 46 features using %s training rows on CPU", len(indices))
        selector.fit(x.iloc[indices][columns], selector_y,
                     sample_weight=class_weights(selector_y, args.class_weight, num_classes)[selector_y])
        scores = dict(zip(columns, selector.feature_importances_.astype(float)))
        columns, threshold = above_median_features(scores)
        LOGGER.info("Retained %s/46 features with gain strictly above median %.6g", len(columns), threshold)
    scaler = StandardScaler().fit(x[columns].to_numpy(dtype=np.float32))
    return FeatureTransform(columns, scaler, scores, threshold, args.feature_selection)


def device_for(name: str, args: Any) -> str:
    return (args.lightgbm_device or args.device) if name == "lightgbm" else args.device


def scoring_device_for(name: str, args: Any) -> str:
    # LightGBM predicts on CPU. CatBoost's GPU evaluator does not support
    # multidimensional models, even when training uses CUDA.
    return device_for(name, args) if name == "xgboost" else "cpu"


def build_model(name: str, args: Any, rounds: int, early_stopping: bool) -> Any:
    gpu = device_for(name, args) == "gpu"
    num_classes = args.num_classes
    if name == "xgboost":
        from xgboost import XGBClassifier

        options = dict(n_estimators=rounds, learning_rate=args.learning_rate, max_depth=args.max_depth,
                       objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss",
                       tree_method="hist", device=f"cuda:{args.gpu_id}" if gpu else "cpu",
                       max_bin=128, subsample=0.9, colsample_bytree=0.9,
                       random_state=args.random_state, n_jobs=args.n_jobs)
        if early_stopping:
            options["early_stopping_rounds"] = args.early_stopping_rounds
        return XGBClassifier(**options)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        options = dict(n_estimators=rounds, learning_rate=args.learning_rate, max_depth=args.max_depth,
                       num_leaves=min(31, 2 ** args.max_depth), objective="multiclass", num_class=num_classes,
                       metric="multi_logloss", max_bin=63, device_type="gpu" if gpu else "cpu",
                       verbosity=-1, random_state=args.random_state, n_jobs=args.n_jobs)
        if gpu:
            options.update(gpu_device_id=args.gpu_id, gpu_use_dp=True)
        return LGBMClassifier(**options)
    if name == "catboost":
        from catboost import CatBoostClassifier

        options = dict(iterations=rounds, learning_rate=args.learning_rate, depth=args.max_depth,
                       loss_function="MultiClass", eval_metric="MultiClass", classes_count=num_classes,
                       task_type="GPU" if gpu else "CPU", thread_count=args.n_jobs,
                       random_seed=args.random_state, allow_writing_files=False, verbose=False)
        if gpu:
            options.update(devices=str(args.gpu_id), gpu_ram_part=0.7, border_count=128)
        return CatBoostClassifier(**options)
    raise ValueError(f"Unknown multiclass model: {name}")


def fit_model(name: str, model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
              args: Any, validation: tuple[pd.DataFrame, np.ndarray] | None = None) -> None:
    kwargs: dict[str, Any] = {"sample_weight": weights[y]}
    if name == "xgboost":
        kwargs["verbose"] = args.log_period or False
        if validation is not None:
            kwargs.update(eval_set=[validation], sample_weight_eval_set=[weights[validation[1]]])
    elif name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        callbacks = [log_evaluation(args.log_period)]
        if validation is not None:
            kwargs.update(eval_set=[validation], eval_sample_weight=[weights[validation[1]]])
            callbacks.append(early_stopping(args.early_stopping_rounds, verbose=bool(args.log_period)))
        kwargs["callbacks"] = callbacks
    elif name == "catboost":
        from catboost import Pool

        kwargs["verbose"] = args.log_period or False
        if validation is not None:
            kwargs.update(eval_set=Pool(validation[0], validation[1], weight=weights[validation[1]]),
                          early_stopping_rounds=args.early_stopping_rounds, use_best_model=True)
    model.fit(x, y, **kwargs)
    if name == "xgboost" and device_for(name, args) == "gpu":
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError(f"Requested XGBoost GPU but it used {actual}. Check CUDA or use --device cpu.")


def best_rounds(name: str, model: Any, default: int) -> int:
    if name == "xgboost":
        return int(model.best_iteration) + 1
    if name == "lightgbm":
        return int(model.best_iteration_ or default)
    best = int(model.get_best_iteration())
    return best + 1 if best >= 0 else default


def predict_ids(name: str, model: Any, x: pd.DataFrame, batch_size: int = 100_000,
                *, device: str = "cpu", gpu_id: int = 0) -> np.ndarray:
    """Bounded prediction; GPU XGBoost scoring, CPU by default for test timing/bundles."""
    if device not in ("cpu", "gpu") or (device == "gpu" and name != "xgboost"):
        raise ValueError("GPU prediction is supported only for XGBoost in this multiclass pipeline.")
    if batch_size <= 0:
        raise ValueError("Prediction batch_size must be positive.")
    if name == "xgboost":
        model.set_params(device=f"cuda:{gpu_id}" if device == "gpu" else "cpu")
        if device == "gpu":
            from xgboost import DMatrix

            booster = model.get_booster()
            # Native Booster.predict does not automatically honor early stopping.
            # Full refits have no best_iteration and must use every tree.
            best_iteration = getattr(model, "best_iteration", None)
            iteration_range = (0, best_iteration + 1 if best_iteration is not None else 0)
    predictions = []
    for start in range(0, len(x), batch_size):
        batch = x.iloc[start:start + batch_size]
        if name == "xgboost" and device == "gpu":
            # Explicit DMatrix avoids the CPU-input/CUDA inplace-prediction warning;
            # the booster keeps its CUDA predictor. No CuPy dependency is required.
            matrix = DMatrix(batch, nthread=model.n_jobs)
            probabilities = booster.predict(matrix, iteration_range=iteration_range, strict_shape=True)
            values = np.argmax(probabilities, axis=1)
            del probabilities, matrix
        else:
            values = model.predict(batch, task_type="CPU") if name == "catboost" else model.predict(batch)
        predictions.append(np.asarray(values, dtype=np.int64).reshape(-1))
    if not predictions:
        return np.empty(0, dtype=np.int64)
    if name == "xgboost" and device == "gpu":
        actual = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError(f"Requested XGBoost GPU prediction but it used {actual}.")
    return np.concatenate(predictions)


@dataclass
class MulticlassBundle:
    """Saved raw-feature -> label model. Load with src on Python's import path."""
    name: str
    model: Any
    features: FeatureTransform
    encoder: LabelEncoder

    def predict(self, raw_features: pd.DataFrame) -> np.ndarray:
        return self.encoder.inverse_transform(predict_ids(self.name, self.model, self.features.transform(raw_features)))


def load_multiclass_bundle(path: Path) -> MulticlassBundle:
    """Load a trusted local 8/34-class bundle without fitting any component.

Joblib uses pickle: do not load model files from untrusted sources. Keep this
bundle class at its existing module path for compatibility with previous runs.
"""
    bundle = joblib.load(path)
    if not isinstance(bundle, MulticlassBundle):
        raise ValueError("Expected a multiclass model bundle; binary model files use src/predict.py.")
    if bundle.name not in MODEL_NAMES or len(bundle.encoder.classes_) not in (8, 34):
        raise ValueError("Unsupported multiclass model or class count in saved bundle.")
    if not bundle.features.columns or not hasattr(bundle.features.scaler, "mean_"):
        raise ValueError("Saved model is missing its fitted feature transformation.")
    return bundle


def evaluate(y: np.ndarray, predicted: np.ndarray, num_classes: int = NUM_CLASSES) -> dict[str, float]:
    labels = np.arange(num_classes)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(recall_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, predicted, labels=labels, average="weighted", zero_division=0))}


def args_with_parameters(args: Any, parameters: dict[str, Any]) -> Any:
    configured = copy(args)
    for key, value in parameters.items():
        setattr(configured, key, value)
    return configured


def resolve_parameter_grids(args: Any) -> dict[str, dict[str, list]]:
    """Validate finite, explicit Cartesian grids before expensive data loading.

The shared names are CLI names: max_depth maps to CatBoost depth, and
n_estimators maps to CatBoost iterations. Custom model grids replace defaults;
omitted parameters in a custom grid stay at their fixed CLI values.
"""
    custom = {}
    if args.param_grid:
        custom = json.loads(args.param_grid.read_text(encoding="utf-8"))
        if not isinstance(custom, dict) or set(custom) - set(MODEL_NAMES):
            raise ValueError(f"Parameter grid must be an object keyed by {MODEL_NAMES}.")
    grids = {}
    for name in dict.fromkeys(args.models):
        base = {key: [getattr(args, key)] for key in GRID_PARAMETERS}
        if name in custom:
            values = custom[name]
            if not isinstance(values, dict) or not values or set(values) - set(GRID_PARAMETERS):
                raise ValueError(f"{name}: grid keys must be chosen from {GRID_PARAMETERS}.")
            grid = {**base, **values}
        else:
            grid = {**base,
                    "learning_rate": list(dict.fromkeys([args.learning_rate, min(1.0, 2 * args.learning_rate)])),
                    "max_depth": list(dict.fromkeys([max(1, args.max_depth - 2), args.max_depth]))}
        for key, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{name}.{key} must be a nonempty list.")
            for value in values:
                if key == "learning_rate":
                    valid = type(value) in (float, int) and np.isfinite(value) and 0 < value <= 1
                else:
                    valid = type(value) is int and value > 0
                if not valid:
                    raise ValueError(f"Invalid value for {name}.{key}: {value!r}")
                if name == "catboost" and key == "max_depth" and value > 16:
                    raise ValueError("CatBoost max_depth must be at most 16.")
            grid[key] = list(dict.fromkeys(values))
        grids[name] = grid
    return grids


def grid_search(name: str, x: pd.DataFrame, y: np.ndarray, args: Any,
                parameter_grid: dict[str, list], output_path: Path,
                shared_transform: FeatureTransform | None = None) -> dict[str, Any]:
    """Exhaustive ParameterGrid search scored by five-fold stratified macro F1.

    A separate inner split selects early-stopping rounds. In fold mode, learned
    preprocessing sees only inner fitting rows. In paper/train mode it sees the
    full training split (never the test set), as in the paper; CV scores may then
    be optimistic. With refit_full each candidate refits on the outer train fold.
"""
    if args.cv_folds != 5:
        raise ValueError("Grid search requires five-fold CV.")
    num_classes = args.num_classes
    if np.bincount(y, minlength=num_classes).min() < args.cv_folds:
        raise ValueError("Not enough rows in every class for five-fold CV.")
    if args.cv_preprocessing == "train" and shared_transform is None:
        shared_transform = fit_features(x, y, args)
    if args.cv_preprocessing == "fold" and shared_transform is not None:
        raise ValueError("Fold-local CV must not receive a full-training feature transform.")
    started = time.perf_counter()
    candidates = [{"candidate": i, "params": params, "fold_scores": [], "fold_best_rounds": [],
                   "status": "pending", "mean_macro_f1": None, "std_macro_f1": None, "rank": None}
                  for i, params in enumerate(ParameterGrid(parameter_grid), 1)]
    report: dict[str, Any] = {"model": name, "status": "running", "scoring": "macro_f1", "cv_folds": 5,
                              "class_count": num_classes, "cv_preprocessing": args.cv_preprocessing,
                              "training_device": device_for(name, args),
                              "scoring_device": scoring_device_for(name, args),
                              "parameter_grid": parameter_grid, "candidates": candidates, "folds": [],
                              "best_params": None, "best_candidate": None, "best_score": None,
                              "test_used_for_tuning": False, "cv_rows": len(y),
                              "refit_each_fold": args.refit_full,
                              "score_note": "CV scores select parameters; the untouched test set measures generalization."}
    output_path.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        completed = []
        for candidate in candidates:
            if candidate["status"] != "failed" and len(candidate["fold_scores"]) == args.cv_folds:
                candidate.update(status="complete", mean_macro_f1=float(np.mean(candidate["fold_scores"])),
                                 std_macro_f1=float(np.std(candidate["fold_scores"])))
                completed.append(candidate)
        ranked = sorted(completed, key=lambda c: (-c["mean_macro_f1"], c["candidate"]))
        for rank, candidate in enumerate(ranked, 1):
            candidate["rank"] = rank
        if ranked:
            best = ranked[0]
            report.update(best_params=best["params"], best_candidate=best["candidate"],
                          best_score=best["mean_macro_f1"], best_fold_scores=best["fold_scores"])
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output_path / "grid_search_report.json", report)
        rows = []
        for candidate in candidates:
            row = {key: candidate[key] for key in ("candidate", "status", "mean_macro_f1", "std_macro_f1", "rank")}
            row.update(candidate["params"])
            row.update({f"fold_{i}_macro_f1": score for i, score in enumerate(candidate["fold_scores"], 1)})
            row["error"] = candidate.get("error")
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_path / "grid_search_results.csv", index=False)

    checkpoint()
    LOGGER.info("%s: exhaustive grid search, %s candidates x 5 folds = %s CV evaluations%s",
                name, len(candidates), len(candidates) * 5,
                " (each includes early-stopping fit + fold refit)" if args.refit_full else "")
    LOGGER.info("%s CV devices: training=%s, scoring=%s, feature selection=cpu",
                name, report["training_device"], report["scoring_device"])
    folds = StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.random_state)
    try:
        for fold, (train, held_out) in enumerate(folds.split(x, y), 1):
            if all(c["status"] == "failed" for c in candidates):
                break
            fit, valid = split_training(y[train], args.validation_size, args.random_state + fold, num_classes)
            fit, valid = train[fit], train[valid]
            transform = shared_transform if shared_transform is not None else fit_features(x.iloc[fit], y[fit], args)
            report["folds"].append({"fold": fold, "fit_rows": len(fit), "early_stopping_rows": len(valid),
                                    "scoring_rows": len(held_out), "selected_features": transform.columns,
                                    "gain_threshold": transform.gain_threshold, "feature_gains": transform.scores})
            x_fit, x_valid = transform.transform(x.iloc[fit]), transform.transform(x.iloc[valid])
            x_score = transform.transform(x.iloc[held_out])
            x_refit = transform.transform(x.iloc[train]) if args.refit_full else None
            weights = class_weights(y[fit], args.class_weight, num_classes)
            refit_weights = class_weights(y[train], args.class_weight, num_classes)
            for candidate in candidates:
                if candidate["status"] == "failed":
                    continue
                tuned_args = args_with_parameters(args, candidate["params"])
                model = None
                report.update(current_fold=fold, current_candidate=candidate["candidate"])
                checkpoint()
                LOGGER.info("%s: fold %s/5, candidate %s/%s %s", name, fold,
                            candidate["candidate"], len(candidates), candidate["params"])
                try:
                    model = build_model(name, tuned_args, tuned_args.n_estimators, True)
                    fit_model(name, model, x_fit, y[fit], weights, tuned_args, (x_valid, y[valid]))
                    rounds = best_rounds(name, model, tuned_args.n_estimators)
                    if args.refit_full:
                        model = None
                        gc.collect()
                        model = build_model(name, tuned_args, rounds, False)
                        fit_model(name, model, x_refit, y[train], refit_weights, tuned_args)
                    predicted = predict_ids(name, model, x_score,
                                            device=scoring_device_for(name, tuned_args), gpu_id=args.gpu_id)
                    score = evaluate(y[held_out], predicted, num_classes)["macro_f1"]
                    candidate["fold_scores"].append(score)
                    candidate["fold_best_rounds"].append(rounds)
                    LOGGER.info("%s: fold %s candidate %s macro F1=%.4f (%s rounds)",
                                name, fold, candidate["candidate"], score, rounds)
                except Exception as exc:
                    candidate.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_fold=fold)
                    LOGGER.exception("%s: candidate %s failed; excluding it from selection", name, candidate["candidate"])
                finally:
                    model = None
                    gc.collect()
                    checkpoint()
            del x_fit, x_valid, x_score, x_refit, transform
            gc.collect()
        if report["best_params"] is None:
            raise RuntimeError(f"All grid candidates failed for {name}; see grid_search_report.json.")
        report["status"] = "complete"
        report["failed_candidates"] = sum(c["status"] == "failed" for c in candidates)
        LOGGER.info("%s: best CV macro F1=%.4f with %s", name, report["best_score"], report["best_params"])
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        checkpoint()
    return report


def save_evaluation(folder: Path, classes: np.ndarray, y: np.ndarray, predicted: np.ndarray,
                    no_plots: bool) -> None:
    num_classes = len(classes)
    matrix = confusion_matrix(y, predicted, labels=np.arange(num_classes))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(folder / "confusion_matrix.csv", index_label="true_label")
    report = classification_report(y, predicted, labels=np.arange(num_classes), target_names=classes,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(folder / "classification_report_test.csv", index_label="class")
    pd.DataFrame({"true_label": classes[y], "predicted_label": classes[predicted]}).to_csv(
        folder / "test_predictions.csv", index=False)
    if no_plots:
        return
    configure_runtime_environment()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    fig, ax = plt.subplots(figsize=(16, 14))
    plot = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted label", ylabel="True label",
           title=f"{num_classes}-class confusion matrix (row normalized)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix_normalized.png", dpi=150)
    plt.close(fig)


def run_multiclass_training(args: Any) -> dict[str, Any]:
    """Fit the three models sequentially, checkpointing reports after each model."""
    num_classes = args.num_classes
    if args.data_protocol == "paper":
        train_features = validate_source_paths(args.source_input_path, args.label_column)
        LOGGER.info("Eight-class paper experiment: source pool=%s; existing test-input-path is NOT used",
                    args.source_input_path)
    else:
        train_features = read_schema(args.train_input_path, args.label_column)
        test_features = read_schema(args.test_input_path, args.label_column)
        if set(train_features) != set(test_features):
            raise ValueError("Raw train/test feature columns do not match.")
        if args.train_input_path.resolve() == args.test_input_path.resolve():
            raise ValueError("Training and test inputs must be different files.")
    if args.counts_only:
        if args.data_protocol == "paper":
            inventory = scan_source_pool(args.source_input_path, args.label_column, args.chunk_size)
            quotas = proportional_quotas(inventory["valid_family_counts"], args.sampling_target, args.min_samples_per_class)
            args.output_path.mkdir(parents=True, exist_ok=True)
            write_json(args.output_path / "source_inventory.json", {**inventory, "quotas": quotas})
            write_json(args.output_path / "raw_to_family_mapping.json", RAW_TO_FAMILY)
            counts = pd.DataFrame({"source_raw": inventory["raw_family_counts"],
                                   "source_valid": inventory["valid_family_counts"], "sampling_quota": quotas})
            counts.to_csv(args.output_path / "class_counts_raw.csv", index_label="class")
            LOGGER.info("Source families and paper quotas (no models trained):\n%s", counts.to_string())
            return {**inventory, "quotas": quotas}
        train_counts = count_labels(args.train_input_path, args.label_column, args.chunk_size, num_classes)
        test_counts = count_labels(args.test_input_path, args.label_column, args.chunk_size, num_classes)
        if len(train_counts) != num_classes or set(train_counts) != set(test_counts):
            raise ValueError(f"Both inputs must contain the same {num_classes} classes.")
        args.output_path.mkdir(parents=True, exist_ok=True)
        counts = pd.DataFrame({"train_raw": train_counts, "test_raw": test_counts}).sort_index()
        counts.to_csv(args.output_path / "class_counts_raw.csv", index_label="class")
        if num_classes == 8:
            write_json(args.output_path / "raw_to_family_mapping.json", RAW_TO_FAMILY)
        LOGGER.info("Raw class counts:\n%s", counts.to_string())
        return {"train_raw": dict(train_counts), "test_raw": dict(test_counts)}

    names = list(dict.fromkeys(args.models))
    if args.feature_selection == "median" and len(train_features) != NUM_FEATURES:
        raise ValueError(f"Median gain selection requires all 46 raw features; found {len(train_features)}.")
    grids = resolve_parameter_grids(args) if args.cv_folds else {}
    # Missing dependencies fail before scanning gigabytes of CSV data.
    required = set(names) | ({"lightgbm"} if args.feature_selection == "median" else set())
    for name in sorted(required):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(f"Install dependencies first: python -m pip install -r requirements.txt ({name})") from exc
    args.output_path.mkdir(parents=True, exist_ok=True)
    if any((args.output_path / name).exists() for name in names) or (args.output_path / "model_comparison_report.json").exists():
        raise FileExistsError(f"Training outputs already exist in {args.output_path}; choose a new --output-path.")
    write_json(args.output_path / "parameter_grids.json", grids)
    if grids:
        LOGGER.info("Grid tuning enabled: %s total CV evaluations, run sequentially",
                    sum(len(ParameterGrid(grid)) * args.cv_folds for grid in grids.values()))
    for name in names:
        LOGGER.info("%s devices: training/refits=%s, CV/validation prediction=%s, test latency=cpu",
                    name, device_for(name, args), scoring_device_for(name, args))
    LOGGER.info("CSV loading, feature selection, scaling and metric calculation use CPU.")
    if args.cv_preprocessing == "train":
        LOGGER.warning("Paper preprocessing: feature selection/scaling fitted once on training data before CV. "
                       "CV can be optimistic; the held-out test set remains excluded. Use --cv-preprocessing fold "
                       "for stricter fold-local preprocessing.")

    started = time.perf_counter()
    sampling_report = None
    if args.data_protocol == "paper":
        train, test, sampling_report = prepare_paper_split(
            args.source_input_path, args.label_column, args.chunk_size, args.random_state,
            args.sampling_target, args.min_samples_per_class, args.test_size)
        write_json(args.output_path / "sampling_report.json", sampling_report)
        train_report, test_report = (generated_split_report(data, args.label_column) for data in (train, test))
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    else:
        train, train_report = load_raw_split(args.train_input_path, args.label_column, args.max_train_per_class,
                                             args.chunk_size, args.random_state, num_classes)
        test, test_report = load_raw_split(args.test_input_path, args.label_column, args.max_test_per_class,
                                          args.chunk_size, args.random_state + 1, num_classes)
    encoder, y, y_test = encode_labels(train[args.label_column], test[args.label_column], num_classes)
    x, x_test = train[train_features], test[train_features]
    del train, test
    fit_indices, valid_indices = split_training(y, args.validation_size, args.random_state, num_classes)
    counts = pd.DataFrame(index=encoder.classes_)
    for prefix, report in (("train", train_report), ("test", test_report)):
        for stage in ("raw", "valid", "selected"):
            counts[f"{prefix}_{stage}"] = pd.Series(report[f"{stage}_counts"])
    if sampling_report:
        counts["source_valid"] = pd.Series(sampling_report["valid_family_counts"])
        counts["sampling_quota"] = pd.Series(sampling_report["quotas"])
    counts["fit"] = np.bincount(y[fit_indices], minlength=num_classes)
    counts["validation"] = np.bincount(y[valid_indices], minlength=num_classes)
    counts.to_csv(args.output_path / "class_counts.csv", index_label="class")
    write_json(args.output_path / "label_mapping.json", {str(i): c for i, c in enumerate(encoder.classes_)})
    if num_classes == 8:
        write_json(args.output_path / "raw_to_family_mapping.json", RAW_TO_FAMILY)
    write_json(args.output_path / "data_report.json", {"train": train_report, "test": test_report,
               "class_count": num_classes, "args": vars(args), "test_used_for_tuning": False,
               "data_protocol": args.data_protocol, "cv_preprocessing": args.cv_preprocessing,
               "fit_rows": len(fit_indices), "validation_rows": len(valid_indices),
               "versions": {n: importlib.metadata.version(n) for n in sorted(required | {"numpy", "pandas", "scikit-learn"})}})
    LOGGER.info("Class counts before training:\n%s", counts.to_string())
    transform_indices = np.arange(len(y)) if args.cv_preprocessing == "train" else fit_indices
    transform = fit_features(x.iloc[transform_indices], y[transform_indices], args)
    write_json(args.output_path / "selected_features.json", transform.columns)
    write_json(args.output_path / "feature_selection_report.json", {
        "method": transform.method, "original_features": train_features, "original_feature_count": len(train_features),
        "gain_threshold": transform.gain_threshold, "comparison": ">" if transform.method == "median" else None,
        "selected_features": transform.columns, "selected_feature_count": len(transform.columns),
        "feature_gains": transform.scores, "fit_rows": len(transform_indices), "test_used_for_selection": False,
        "cv_preprocessing": args.cv_preprocessing})
    feature_scores = pd.DataFrame(list(transform.scores.items()), columns=["feature", "gain"])
    feature_scores["selected"] = feature_scores["feature"].isin(transform.columns)
    feature_scores.to_csv(args.output_path / "feature_selection_scores.csv", index=False)
    x_transformed, test_transformed = transform.transform(x), transform.transform(x_test)
    del x_test
    x_fit, x_valid = x_transformed.iloc[fit_indices], x_transformed.iloc[valid_indices]
    fit_weights = class_weights(y[fit_indices], args.class_weight, num_classes)
    preprocessing_seconds = time.perf_counter() - started
    LOGGER.info("Ready: %s train rows, %s test rows, %s classes, %s selected features",
                f"{len(y):,}", f"{len(y_test):,}", num_classes, len(transform.columns))
    results, failures, saved_models = [], {}, {}
    selection_metric = "cv_macro_f1_mean" if args.cv_folds else "validation_macro_f1"
    majority = int(np.argmax(np.bincount(y, minlength=num_classes)))
    comparison: dict[str, Any] = {"selection_metric": selection_metric, "results": results,
                                  "saved_models": saved_models,
                                  "class_count": num_classes, "data_protocol": args.data_protocol,
                                  "cv_preprocessing": args.cv_preprocessing,
                                  "failures": failures, "preprocessing_seconds": preprocessing_seconds,
                                  "grid_search_enabled": bool(args.cv_folds), "cv_folds": args.cv_folds,
                                  "majority_class_baseline": {"label": str(encoder.classes_[majority]),
                                      **evaluate(y_test, np.full(len(y_test), majority), num_classes)},
                                  "best_model": None, "status": "running"}

    def checkpoint() -> None:
        comparison["best_model"] = max(results, key=lambda r: r[selection_metric])["model"] if results else None
        write_json(args.output_path / "model_comparison_report.json", comparison)
        if results:
            pd.DataFrame(results).sort_values(selection_metric, ascending=False).to_csv(
                args.output_path / "model_comparison.csv", index=False)

    checkpoint()
    for name in names:
        model = None
        save_report = None
        try:
            folder = args.output_path / name
            folder.mkdir(parents=True, exist_ok=True)
            cv_start = time.perf_counter()
            search = grid_search(name, x, y, args, grids[name], folder,
                                 shared_transform=transform if args.cv_preprocessing == "train" else None) if args.cv_folds else None
            selected_params = search["best_params"] if search else {key: getattr(args, key) for key in GRID_PARAMETERS}
            tuned_args = args_with_parameters(args, selected_params)
            cv_scores = search["best_fold_scores"] if search else []
            cv_seconds = time.perf_counter() - cv_start if cv_scores else 0.0
            LOGGER.info("Training final %s: device=%s, selected parameters=%s",
                        name, device_for(name, args), selected_params)
            model = build_model(name, tuned_args, tuned_args.n_estimators, True)
            timer = time.perf_counter()
            fit_model(name, model, x_fit, y[fit_indices], fit_weights, tuned_args, (x_valid, y[valid_indices]))
            early_fit_seconds = time.perf_counter() - timer
            rounds = best_rounds(name, model, tuned_args.n_estimators)
            validation_metrics = evaluate(y[valid_indices], predict_ids(
                name, model, x_valid, device=scoring_device_for(name, tuned_args), gpu_id=args.gpu_id), num_classes)
            refit_seconds = 0.0
            if args.refit_full:
                LOGGER.info("%s: refitting on ALL %s selected training rows for %s rounds", name, len(y), rounds)
                model = None
                gc.collect()
                model = build_model(name, tuned_args, rounds, False)
                timer = time.perf_counter()
                fit_model(name, model, x_transformed, y, class_weights(y, args.class_weight, num_classes), tuned_args)
                refit_seconds = time.perf_counter() - timer
            # Persist each completed model immediately, before potentially costly
            # test prediction or reporting. Never wait for the other models.
            if name == "xgboost":
                model.set_params(device="cpu")
            model_path = folder / "model.joblib"
            save_joblib_atomic(model_path, MulticlassBundle(name, model, transform, encoder))
            saved_models[name] = {"path": str(model_path.resolve()), "class_count": num_classes,
                                  "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "evaluation_status": "pending"}
            save_report = {
                **saved_models[name], "model": name, "stage": "final_training_complete",
                "training_device": device_for(name, args), "best_rounds": rounds,
                "training_rows": len(y) if args.refit_full else len(fit_indices),
                "selected_hyperparameters": selected_params,
                "contents": ["trained_model", "selected_features", "fitted_scaler", "label_encoder"],
            }
            write_json(folder / "model_save_report.json", save_report)
            checkpoint()
            LOGGER.info("Saved %s model to %s; it can be loaded without retraining", name, model_path)
            # Warm-up is excluded; timing covers only batched CPU prediction.
            predict_ids(name, model, test_transformed.iloc[:min(100, len(y_test))])
            timer = time.perf_counter()
            predicted = predict_ids(name, model, test_transformed)
            inference_seconds = time.perf_counter() - timer
            metrics = evaluate(y_test, predicted, num_classes)
            predicted_counts = np.bincount(predicted, minlength=num_classes)
            if np.count_nonzero(predicted_counts) == 1:
                LOGGER.warning("%s predicts only ONE class. Inspect class reports before using this model.", name)
            row = {"model": name, **metrics, "validation_macro_f1": validation_metrics["macro_f1"],
                   "class_count": num_classes,
                   "predicted_class_count": int(np.count_nonzero(predicted_counts)),
                   "training_device": device_for(name, args), "inference_device": "cpu", "best_rounds": rounds,
                   "cv_scoring_device": scoring_device_for(name, args),
                   "training_rows": len(y) if args.refit_full else len(fit_indices), "test_rows": len(y_test),
                   "early_fit_seconds": early_fit_seconds, "refit_seconds": refit_seconds,
                   "training_seconds": early_fit_seconds + refit_seconds, "inference_seconds": inference_seconds,
                   "total_training_seconds": cv_seconds + early_fit_seconds + refit_seconds,
                   "inference_us_per_sample": inference_seconds / len(y_test) * 1_000_000,
                   "cv_macro_f1_mean": float(np.mean(cv_scores)) if cv_scores else None,
                   "cv_macro_f1_std": float(np.std(cv_scores)) if cv_scores else None, "cv_seconds": cv_seconds,
                   "search_candidates": len(search["candidates"]) if search else 0,
                   "selected_feature_count": len(transform.columns),
                   **{f"selected_{key}": value for key, value in selected_params.items()}}
            write_json(folder / "metrics.json", row)
            write_json(folder / "model_metadata.json", {"classes": encoder.classes_, "features": transform.columns,
                       "class_weights": class_weights(y if args.refit_full else y[fit_indices], args.class_weight, num_classes),
                       "class_count": num_classes, "cv_preprocessing": args.cv_preprocessing,
                       "raw_to_family_mapping": RAW_TO_FAMILY if num_classes == 8 else None,
                       "parameters": model.get_params(), "validation_metrics": validation_metrics,
                       "selected_hyperparameters": selected_params, "grid_search_enabled": bool(search),
                       "feature_selection_method": transform.method, "gain_threshold": transform.gain_threshold,
                       "validation_note": "Diagnostic early-stopping split; not independent of CV parameter selection.",
                       "test_predicted_class_counts": dict(zip(encoder.classes_, predicted_counts)),
                       "cv_macro_f1_scores": cv_scores, "training_device": device_for(name, args),
                       "prediction_device": "cpu", "test_sampled": bool(args.max_test_per_class)})
            save_evaluation(folder, encoder.classes_, y_test, predicted, args.no_plots)
            results.append(row)
            saved_models[name]["evaluation_status"] = "complete"
            save_report["evaluation_status"] = "complete"
            write_json(folder / "model_save_report.json", save_report)
            LOGGER.info("%s: test accuracy=%.4f, macro F1=%.4f; validation macro F1=%.4f",
                        name, metrics["accuracy"], metrics["macro_f1"], row["validation_macro_f1"])
        except Exception as exc:
            failures[name] = {"type": type(exc).__name__, "message": str(exc)}
            if name in saved_models:
                saved_models[name]["evaluation_status"] = "failed"
                save_report["evaluation_status"] = "failed"
                write_json(folder / "model_save_report.json", save_report)
            LOGGER.exception("%s failed; continuing with remaining models", name)
            if name == "lightgbm" and device_for(name, args) == "gpu":
                LOGGER.error("Try --lightgbm-device cpu with a new output path if OpenCL training fails.")
        except KeyboardInterrupt:
            if name in saved_models:
                saved_models[name]["evaluation_status"] = "interrupted"
                save_report["evaluation_status"] = "interrupted"
                write_json(folder / "model_save_report.json", save_report)
            comparison["status"] = "interrupted"
            checkpoint()
            raise
        finally:
            del model
            gc.collect()
            checkpoint()
    comparison["status"] = ("partial_failure" if results else "failed") if failures else "complete"
    checkpoint()
    if failures:
        raise RuntimeError(f"Models failed: {', '.join(failures)}. Successful results are saved in {args.output_path}.")
    LOGGER.info("Saved all results to %s; best model by %s: %s", args.output_path, selection_metric, comparison["best_model"])
    return comparison
