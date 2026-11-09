"""Cross-fitted probabilities with explicit row alignment and partial-run status."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import Candidate, ExperimentConfig
from .decisions import class_f1, validate_probabilities
from .storage import close_memmap, write_arrays, write_report
from .training import fit_partition, partition_probabilities

LOGGER = logging.getLogger(__name__)


def generate_oof(x: pd.DataFrame, y: np.ndarray, strata: np.ndarray, row_ids: np.ndarray,
                  config: ExperimentConfig, candidate: Candidate, seed: int, folder: Path) -> tuple[np.ndarray, dict]:
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "folds": [], "test_used_for_tuning": False,
              "validation_used_for_training": False, "row_count": len(y),
              "class_count": config.num_classes, "candidate": candidate.name, "seed": seed,
              "preprocessing": "fold", "row_alignment": "rows.npz aligns with probabilities.npy",
              "resumable": False}
    probabilities = np.lib.format.open_memmap(
        folder / "probabilities.npy", mode="w+", dtype=np.float32, shape=(len(y), config.num_classes))
    probabilities[:] = np.nan
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    write_report(folder / "report.json", report)
    splitter = StratifiedKFold(config.folds, shuffle=True, random_state=seed)
    try:
        for fold, (train, score) in enumerate(splitter.split(x, strata), 1):
            LOGGER.info("%s seed %s: OOF fold %s/%s", candidate.name, seed, fold, config.folds)
            fitted = fit_partition(x.iloc[train], y[train], strata[train], config, candidate, seed + fold)
            p = validate_probabilities(partition_probabilities(fitted, x.iloc[score], config, candidate, seed))
            if p.shape != (len(score), config.num_classes):
                raise ValueError("Model probability columns do not match the label mapping.")
            probabilities[score] = p
            fold_ids[score] = fold
            probabilities.flush()
            report["folds"].append({"fold": fold, "scoring_rows": len(score),
                                    "macro_f1": float(class_f1(y[score], p.argmax(axis=1), config.num_classes).mean()),
                                    **fitted.report})
            write_arrays(folder / "rows.npz", row_ids=row_ids, y=y, fold_ids=fold_ids)
            write_report(folder / "report.json", report)
            del fitted, p
            gc.collect()
        if np.any(fold_ids < 0):
            raise RuntimeError("OOF coverage is incomplete.")
        report.update(status="complete", seconds=time.perf_counter() - started,
                      oof_macro_f1=float(class_f1(y, probabilities.argmax(axis=1), config.num_classes).mean()))
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
        close_memmap(probabilities)
        raise
    finally:
        if not probabilities._mmap.closed:
            probabilities.flush()
        write_report(folder / "report.json", report)
    return probabilities, report
