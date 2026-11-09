"""Post-training paired CPU timing, using the saved validation cohort only."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ddos_iot.multiclass import load_multiclass_bundle, read_schema
from .config import config_from_dict
from .evaluation import predictions
from .storage import file_digest, write_report, write_table


def benchmark_saved_models(run_path: Path, *, seed: int = 42, repeats: int = 5) -> pd.DataFrame:
    if repeats < 3:
        raise ValueError("Use at least three measurement repeats.")
    experiment = json.loads((run_path / "experiment_report.json").read_text())
    if experiment["status"] not in ("complete", "partial_failure"):
        raise ValueError("Wait for training to finish before the paired benchmark.")
    config = config_from_dict(json.loads((run_path / "config.json").read_text()))
    if seed not in config.seeds:
        raise ValueError("Benchmark seed was not declared in this experiment.")
    selection = json.loads((run_path / "selection.json").read_text())
    if file_digest(config.train_input_path) != selection["training_source_sha256"]:
        raise ValueError("Training source changed since development.")
    with np.load(run_path / f"seed_{seed}" / "splits.npz") as split:
        row_ids = split["validation_row_ids"]
    count = json.loads((run_path / "data_report.json").read_text())["raw_rows"]
    membership = np.zeros(count, dtype=bool)
    membership[row_ids] = True
    features = read_schema(config.train_input_path, config.label_column)
    parts = []
    for chunk in pd.read_csv(config.train_input_path, usecols=features, dtype="float32", chunksize=config.chunk_size):
        part = chunk.loc[membership[chunk.index.to_numpy()]]
        if len(part):
            parts.append(part)
    x = pd.concat(parts)
    del parts
    np.testing.assert_array_equal(x.index.to_numpy(), row_ids)
    selected = selection["candidate"]["name"]
    names = list(dict.fromkeys([config.reference_name, selected]))
    bundles = {}
    for name in names:
        folder = run_path / f"seed_{seed}" / name
        bundle = load_multiclass_bundle(folder / "model.joblib")
        bundle.decision_bias = np.array(json.loads((folder / "bias.json").read_text())["bias"])
        bundles[name] = bundle
    conditions = list(dict.fromkeys([(config.reference_name, "argmax"), (selected, "argmax"), (selected, selection["decision"])]))
    print(f"Warming {len(conditions)} conditions on all {len(x):,} validation rows", flush=True)
    for name, decision in conditions:
        actual = predictions(bundles[name], x, config, decision)
        with np.load(run_path / f"seed_{seed}" / name / "validation" / decision / "predictions.npz") as saved:
            np.testing.assert_array_equal(actual, saved["predicted"])
    rows = []
    for repeat in range(repeats):
        offset = repeat % len(conditions)
        order = conditions[offset:] + conditions[:offset]
        for name, decision in order:
            started = time.perf_counter()
            predictions(bundles[name], x, config, decision)
            seconds = time.perf_counter() - started
            value = seconds * 1e6 / len(x)
            rows.append({"candidate": name, "decision": decision, "repeat": repeat + 1,
                         "seconds": seconds, "inference_us_per_sample": value})
            print(f"{name} {decision} repeat {repeat + 1}: {value:.3f} us/flow", flush=True)
    output = run_path / "report" / f"paired_benchmark_seed_{seed}"
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["candidate", "decision"], sort=False).agg(
        median_us=("inference_us_per_sample", "median"), min_us=("inference_us_per_sample", "min"),
        max_us=("inference_us_per_sample", "max"), std_us=("inference_us_per_sample", lambda s: s.std(ddof=0))).reset_index()
    write_table(output / "runs.csv", frame)
    write_table(output / "summary.csv", summary)
    write_report(output / "scope.json", {
        "validation_rows": len(x), "full_validation_warmup": True, "repeats": repeats,
        "order": "Rotating conditions each repeat", "n_jobs": config.n_jobs,
        "inference_device": "cpu", "batch_size": config.prediction_batch_size,
        "scope": "Feature transformation, predict_proba, and decision. Loading/warmup excluded.",
        "predictions_match_original_validation": True, "test_file_read": False,
        "note": "Timing spread is measurement variability, not variation across training seeds.",
    })
    return summary
