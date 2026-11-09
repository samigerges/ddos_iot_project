"""Confirm fixed study finalists across seeds without repeating model search."""

from dataclasses import replace
import gc
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from .recall_constraints import reference_constraints
from .storage import object_digest, write_arrays, write_report, write_table
from .study import _cpu_timing, _score
from .study_config import TrainingTrial, load_study_config
from .study_data import load_study_data, protected_ids, study_split
from .study_training import refit_trial

LOGGER = logging.getLogger(__name__)


def confirmation_summary(rows):
    records = []
    frame = pd.DataFrame(rows)
    for name, group in frame.groupby("candidate", sort=False):
        item = {"candidate": name, "seed_count": len(group),
                "all_reference_constraints_pass": bool(group.reference_constraints_pass.all())}
        for metric in ("macro_f1", "weighted_f1", "accuracy", "benign_false_positive_rate", "inference_us_per_sample", "training_seconds"):
            item[f"{metric}_mean"] = float(group[metric].mean())
            item[f"{metric}_std"] = float(group[metric].std(ddof=0))
        records.append(item)
    return pd.DataFrame(records)


def confirm_training_study(source: Path, output: Path, *, seeds=(42, 123, 2024), reuse_source_seed=True):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    original = read(source / "report.json")
    if original["status"] != "complete":
        raise ValueError("Confirm only a complete study with all planned trial results.")
    if len(set(seeds)) != len(seeds) or len(seeds) < 3:
        raise ValueError("Confirm on at least three distinct seeds.")
    config = load_study_config(source / "config.json")
    if object_digest(read(source / "config.json")) != original["config_sha256"]:
        raise ValueError("Study configuration changed after selection.")
    chosen_seed = config.seeds[0]
    selection = read(source / f"seed_{chosen_seed}/selection.json")
    source_digest = original["selections"][str(chosen_seed)]["sha256"]
    if object_digest(selection) != source_digest:
        raise ValueError("Study selection changed after it was frozen.")
    # Confirm the fixed reference, unrestricted and guarded choice, plus the
    # strongest XGBoost when CatBoost won. No new search is run for later seeds.
    names = list(dict.fromkeys(["paper_fixed", selection["best_macro"], selection["guarded_macro"], selection["best_xgboost"]]))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "loading", "source_study": str(source.resolve()), "source_selection_sha256": source_digest,
              "candidates": names, "seeds": list(seeds), "source_seed": chosen_seed, "results": [],
              "hyperparameters_and_rounds": {name: selection["trials"][name] for name in names},
              "hyperparameters_frozen_across_seeds": True, "test_file_read": False,
              "split_scope": "Resplit the same published training source for each seed; repeated splits are not independent datasets.",
              "upstream_feature_selection": original["upstream_feature_selection"], "seed_std_ddof": 0,
              "source_seed_timing_repeats": config.latency_repeats, "additional_seed_timing_repeats": 1}
    path = output / "report.json"
    write_report(path, report)
    try:
        data = load_study_data(config)
        classes = data.encoder.classes_
        protected = protected_ids(classes, config)
        benign = list(classes).index("Benign" if config.num_classes == 8 else "BenignTraffic")
        for seed in seeds:
            if type(seed) is not int or not 0 <= seed < 2**32 - 1:
                raise ValueError("Invalid confirmation seed.")
            if reuse_source_seed and seed == chosen_seed:
                reused = [r for r in original["validation_results"] if r["seed"] == seed and r["candidate"] in names]
                if {r["candidate"] for r in reused} != set(names):
                    raise ValueError("Source seed's validation results are incomplete.")
                for row in reused:
                    report["results"].append({**row, "reused_source_result": True,
                                               "artifact_folder": str((source / f"seed_{seed}/finalists" / row["candidate"]).resolve())})
                write_report(path, report)
                continue
            fitting, tuning, validation = study_split(data.y, config, seed)
            development = np.sort(np.concatenate([fitting, tuning]))
            root = output / f"seed_{seed}"
            write_arrays(root / "splits.npz", development=development, validation=validation,
                         y=data.y, classes=np.asarray(classes, dtype=str))
            x_train, y_train = data.x.iloc[development], data.y[development]
            x_validation, y_validation = data.x.iloc[validation], data.y[validation]
            constraints = None
            for name in names:
                settings = selection["trials"][name]
                trial = TrainingTrial(**settings["trial"])
                LOGGER.info("Confirmation: %s classes seed %s candidate %s", config.num_classes, seed, name)
                bundle, training = refit_trial(x_train, y_train, data.encoder, trial, settings["rounds"], config, seed)
                destination = root / name
                save_joblib_atomic(destination / "model.joblib", bundle)
                write_report(destination / "training.json", training)
                predicted, timing = _cpu_timing(bundle, x_validation, replace(config, latency_repeats=1))
                if constraints is None:
                    constraints = reference_constraints(y_validation, predicted, config.num_classes,
                                                        benign_id=benign, protected_ids=protected)
                row = _score(destination / "validation", y_validation, predicted, classes,
                             "validation", name, seed, constraints)
                row.update(**timing, training_seconds=training["training_seconds"], reused_source_result=False,
                           artifact_folder=str(destination.resolve()))
                report["results"].append(row)
                write_report(destination / "validation/metrics.json", row)
                write_report(path, report)
                del bundle
                gc.collect()
            del x_train, x_validation
            gc.collect()
        summary = confirmation_summary(report["results"])
        write_table(output / "validation_comparison.csv", pd.DataFrame(report["results"]))
        write_table(output / "validation_summary.csv", summary)
        if not (summary.seed_count == len(seeds)).all():
            raise RuntimeError("Seed confirmation coverage is incomplete.")
        # This validation-based final selection is explicit and precedes any
        # future test evaluation. Biases/hyperparameters are not retuned here.
        feasible = summary.loc[summary.all_reference_constraints_pass].sort_values("macro_f1_mean", ascending=False, kind="stable")
        winner = str(feasible.iloc[0].candidate)
        frozen = {"candidate": winner, "settings": selection["trials"][winner], "seeds": list(seeds),
                  "source_selection_sha256": source_digest,
                  "rule": "Highest mean validation Macro-F1 preserving reference recall/FPR in every confirmation seed.",
                  "test_file_read": False, "full_training_refit_required": True,
                  "upstream_feature_selection": report["upstream_feature_selection"]}
        write_report(output / "selection.json", frozen)
        report.update(status="complete", selected_candidate=winner, selection_sha256=object_digest(frozen))
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(path, report)
    return report
