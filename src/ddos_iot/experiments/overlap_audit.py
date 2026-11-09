"""Bounded, training-source-only overlap diagnostics for saved row splits."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ddos_iot.attack_families import RAW_TO_FAMILY
from ddos_iot.multiclass import read_schema
from .config import config_from_dict
from .storage import close_memmap, file_digest, write_arrays, write_report, write_table


def _class_table(names, development_y, validation_y, mask):
    n = len(names)
    development = np.bincount(development_y, minlength=n)
    validation = np.bincount(validation_y, minlength=n)
    shared = np.bincount(validation_y[mask], minlength=n)
    return pd.DataFrame({"class": names, "development_rows": development,
                         "validation_rows": validation, "overlapping_validation_rows": shared,
                         "unseen_validation_rows": validation - shared,
                         "validation_overlap_fraction": np.divide(shared, validation,
                             out=np.zeros(n, dtype=float), where=validation != 0)})


def audit_overlap(run_path: Path, seed: int, *, output_path: Path | None = None) -> dict:
    """Hash all 46 numeric features; labels are excluded from group identities."""
    config = config_from_dict(json.loads((run_path / "config.json").read_text()))
    data_report = json.loads((run_path / "data_report.json").read_text())
    if seed not in config.seeds:
        raise ValueError("The seed was not declared in this experiment.")
    if file_digest(config.train_input_path) != data_report["source_sha256"]:
        raise ValueError("Training source changed since this run; refusing an incomparable audit.")
    output = output_path or run_path / "overlap_audit" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    start_stat = config.train_input_path.stat()
    report = {"status": "running", "seed": seed, "test_file_read": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": data_report["source_sha256"],
              "method": "64-bit hashes of all numeric float32 features, excluding labels.",
              "scope": "Existing development/validation split; diagnostic only, not grouped retraining."}
    hashes = raw_codes = None
    try:
        features = read_schema(config.train_input_path, config.label_column)
        rows = data_report["raw_rows"]
        hashes = np.lib.format.open_memmap(output / "feature_hashes.npy", mode="w+", dtype=np.uint64, shape=(rows,))
        raw_codes = np.lib.format.open_memmap(output / "raw_codes.npy", mode="w+", dtype=np.int16, shape=(rows,))
        raw_names = np.array(sorted(RAW_TO_FAMILY))
        mapping = {name: i for i, name in enumerate(raw_names)}
        dtypes = {f: "float32" for f in features}
        dtypes[config.label_column] = "string"
        offset = 0
        for chunk in pd.read_csv(config.train_input_path, dtype=dtypes, chunksize=config.chunk_size):
            end = offset + len(chunk)
            if end > rows:
                raise ValueError("Source row count changed during audit.")
            hashes[offset:end] = pd.util.hash_pandas_object(chunk[features], index=False).to_numpy()
            raw_codes[offset:end] = chunk[config.label_column].str.strip().map(mapping).fillna(-1).to_numpy(dtype=np.int16)
            offset = end
        if offset != rows or config.train_input_path.stat().st_mtime_ns != start_stat.st_mtime_ns:
            raise ValueError("Source changed while being scanned.")
        with np.load(run_path / f"seed_{seed}" / "splits.npz") as split:
            development_ids, validation_ids = split["development_row_ids"], split["validation_row_ids"]
            development_y, validation_y, classes = split["development_y"], split["validation_y"], split["classes"]
        shared = np.isin(hashes[validation_ids], np.unique(hashes[development_ids]))
        expected = json.loads((run_path / f"seed_{seed}" / "overlap.json").read_text())
        if int(shared.sum()) != expected["overlapping_right_rows"]:
            raise ValueError("Recomputed overlap does not match the training workflow's audit.")
        if np.any(raw_codes[development_ids] < 0) or np.any(raw_codes[validation_ids] < 0):
            raise ValueError("Saved split contains unknown raw labels.")
        task_table = _class_table(classes, development_y, validation_y, shared)
        raw_table = _class_table(raw_names, raw_codes[development_ids], raw_codes[validation_ids], shared)
        write_table(output / "overlap_by_class.csv", task_table)
        write_table(output / "overlap_by_raw_class.csv", raw_table)
        write_arrays(output / "validation_overlap.npz", row_ids=validation_ids, overlap=shared,
                     y=validation_y, classes=classes)
        report.update(status="complete", validation_rows=len(validation_ids),
                      overlapping_validation_rows=int(shared.sum()), validation_overlap_fraction=float(shared.mean()))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        close_memmap(hashes)
        close_memmap(raw_codes)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(output / "report.json", report)
    return report
