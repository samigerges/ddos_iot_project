"""Per-class errors, family aggregation, seed summaries, and repeated CPU timing."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ddos_iot.attack_families import FAMILY_NAMES, RAW_TO_FAMILY
from ddos_iot.multiclass import MulticlassBundle, evaluate, predict_probabilities
from .config import ExperimentConfig
from .decisions import predict_with_bias
from .storage import write_arrays, write_report, write_table


def predictions(bundle: MulticlassBundle, x: pd.DataFrame, config: ExperimentConfig, decision: str) -> np.ndarray:
    p = predict_probabilities(bundle.name, bundle.model, bundle.features.transform(x), config.prediction_batch_size)
    return p.argmax(axis=1) if decision == "argmax" else predict_with_bias(p, bundle.decision_bias)


def benchmark(bundle: MulticlassBundle, x: pd.DataFrame, config: ExperimentConfig, decision: str) -> dict:
    predictions(bundle, x.iloc[:min(100, len(x))], config, decision)
    times = []
    for _ in range(config.latency_repeats):
        started = time.perf_counter()
        predictions(bundle, x, config, decision)
        times.append(time.perf_counter() - started)
    return {"inference_seconds_median": float(np.median(times)),
            "inference_us_per_sample": float(np.median(times) * 1e6 / len(x)),
            "inference_seconds_runs": times, "inference_device": "cpu",
            "latency_scope": "Batched CPU transformation, predict_proba, and decision; excludes CSV loading.",
            "prediction_batch_size": config.prediction_batch_size,
            "single_flow_latency_measured": False}


def save_scores(folder: Path, classes: np.ndarray, y: np.ndarray, predicted: np.ndarray, *,
                 split: str, role: str) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    k = len(classes)
    matrix = confusion_matrix(y, predicted, labels=np.arange(k))
    report = classification_report(y, predicted, labels=np.arange(k), target_names=list(classes),
                                   output_dict=True, zero_division=0)
    result = {**evaluate(y, predicted, k), "split": split, "score_role": role, "rows": len(y)}
    benign = "Benign" if k == 8 else "BenignTraffic"
    benign_id = list(classes).index(benign)
    mask = y == benign_id
    result["benign_false_positive_rate"] = float(np.mean(predicted[mask] != benign_id)) if mask.any() else None
    attacks = ~mask
    result["attack_to_benign_rate"] = float(np.mean(predicted[attacks] == benign_id)) if attacks.any() else None
    write_table(folder / "confusion_matrix.csv", pd.DataFrame(matrix, index=classes, columns=classes).rename_axis("true_label"), index=True)
    write_table(folder / "classification_report.csv", pd.DataFrame(report).T.rename_axis("class"), index=True)
    errors = [{"true_label": classes[a], "predicted_label": classes[b], "count": int(matrix[a, b]),
               "fraction_of_true_class": float(matrix[a, b] / matrix[a].sum())}
              for a, b in zip(*np.nonzero(matrix)) if a != b]
    write_table(folder / "confusions.csv", pd.DataFrame(
        sorted(errors, key=lambda e: -e["count"]),
        columns=["true_label", "predicted_label", "count", "fraction_of_true_class"]))
    write_arrays(folder / "predictions.npz", y=y, predicted=predicted, classes=np.asarray(classes, dtype=str))
    if k == 34:
        families = np.asarray(FAMILY_NAMES)
        mapping = np.array([list(families).index(RAW_TO_FAMILY[c]) for c in classes])
        true_family, predicted_family = mapping[y], mapping[predicted]
        result["family_macro_f1"] = evaluate(true_family, predicted_family, 8)["macro_f1"]
        result["within_family_error_count"] = int(np.sum((y != predicted) & (true_family == predicted_family)))
        result["between_family_error_count"] = int(np.sum(true_family != predicted_family))
        family_matrix = confusion_matrix(true_family, predicted_family, labels=np.arange(8))
        write_table(folder / "family_confusion_matrix.csv",
                    pd.DataFrame(family_matrix, index=families, columns=families).rename_axis("true_label"), index=True)
    write_report(folder / "metrics.json", result)
    return result


def summarize_seeds(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metrics = ["macro_f1", "weighted_f1", "accuracy", "benign_false_positive_rate", "inference_us_per_sample"]
    metrics.extend(name for name in ("training_seconds", "oof_seconds", "bias_tuning_seconds") if name in frame)
    records = []
    for (candidate, decision), group in frame.groupby(["candidate", "decision"], sort=False):
        record = {"candidate": candidate, "decision": decision, "seed_count": len(group)}
        for metric in metrics:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=0))
        records.append(record)
    return pd.DataFrame(records)


def paired_deltas(rows: list[dict], reference: str) -> pd.DataFrame:
    """Compare each decision with the declared reference on the same seed/split."""
    frame = pd.DataFrame(rows)
    baseline = frame.loc[(frame.candidate == reference) & (frame.decision == "argmax"),
                         ["seed", "macro_f1", "benign_false_positive_rate"]]
    merged = frame.merge(baseline, on="seed", suffixes=("", "_reference"), how="inner")
    merged["macro_f1_delta"] = merged.macro_f1 - merged.macro_f1_reference
    merged["benign_fpr_delta"] = merged.benign_false_positive_rate - merged.benign_false_positive_rate_reference
    merged["reference"] = reference
    return merged[["candidate", "decision", "seed", "reference", "macro_f1_delta", "benign_fpr_delta"]]
