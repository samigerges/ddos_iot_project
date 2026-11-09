"""Once-only published-test evaluation after multiseed selection is frozen."""

import gc
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import tarfile
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle
from .reference_data import verify_archive
from .storage import object_digest, write_report, write_table
from .study import _cpu_timing, _score
from .study_config import TrainingTrial, load_study_config
from .study_data import load_study_data
from .study_training import refit_trial

LOGGER = logging.getLogger(__name__)


def extract_frozen_test(archive: Path, record_path: Path, destination: Path, num_classes: int):
    """Called only after every final model has been saved; no training occurs here."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    provenance = verify_archive(archive, record)
    name = f"{num_classes}class_test_23features.csv"
    destination.mkdir(parents=True, exist_ok=False)
    found = None
    with tarfile.open(archive, mode="r|xz") as source:
        for member in source:
            if PurePosixPath(member.name).name != name:
                continue
            if not member.isfile() or found is not None:
                raise ValueError("Test member must be a unique regular file.")
            path = destination / name
            sha = hashlib.sha256()
            with source.extractfile(member) as incoming, path.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(4 * 1024 * 1024), b""):
                    outgoing.write(block); sha.update(block)
            found = path
    if found is None:
        raise ValueError("The verified archive has no matching test CSV.")
    write_report(destination / "provenance.json", {"archive": provenance, "member": name, "sha256": sha.hexdigest()})
    return found


def evaluate_study_test(confirmation: Path, output: Path, *, archive: Path, record_path: Path, test_directory: Path,
                        latency_repeats=1, paper_reference=None):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    confirmed = read(confirmation / "report.json")
    frozen = read(confirmation / "selection.json")
    if confirmed["status"] != "complete" or object_digest(frozen) != confirmed["selection_sha256"]:
        raise ValueError("A completed, unchanged multiseed selection is required before test access.")
    source = Path(confirmed["source_study"])
    config = load_study_config(source / "config.json")
    if type(latency_repeats) is not int or latency_repeats < 1:
        raise ValueError("Use at least one timed pass after full-cohort warm-up.")
    timing_config = replace(config, latency_repeats=latency_repeats)
    original = read(source / "report.json")
    chosen_seed = confirmed["source_seed"]
    parent_selection = read(source / f"seed_{chosen_seed}/selection.json")
    if object_digest(parent_selection) != confirmed["source_selection_sha256"] or object_digest(read(source / "config.json")) != original["config_sha256"]:
        raise ValueError("Source settings changed after development.")
    if test_directory.exists():
        raise FileExistsError("The final test directory already exists; repeated evaluation is not automatic.")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    names = list(dict.fromkeys(["paper_fixed", frozen["candidate"]]))
    settings = {"paper_fixed": parent_selection["trials"]["paper_fixed"], frozen["candidate"]: frozen["settings"]}
    protocol = {"confirmation_selection_sha256": confirmed["selection_sha256"], "candidates": settings,
                "seeds": frozen["seeds"], "test_archive": str(archive.resolve()),
                "timing_repeats_after_full_warmup": latency_repeats,
                "rule": "Evaluate all declared reference/selected models; never select seeds or retune from test results.",
                "upstream_feature_selection": original["upstream_feature_selection"]}
    if paper_reference is not None:
        from .paper_reference import paper_reference_descriptor
        protocol["additional_paper_reference"] = paper_reference_descriptor(paper_reference)
    write_report(output / "protocol.json", protocol)
    report = {"status": "fitting_all_final_models", "protocol_sha256": object_digest(protocol), "results": [],
              "test_used_for_selection": False, "test_csv_opened": False, "models_saved": [],
              "training_source_study": str(source.resolve())}
    path = output / "report.json"
    write_report(path, report)
    try:
        data = load_study_data(config)
        if paper_reference is not None and protocol["additional_paper_reference"]["source_sha256"] != data.metadata["source_sha256"]:
            raise ValueError("Paper reference was fitted to a different training subset.")
        classes = data.encoder.classes_
        for seed in frozen["seeds"]:
            for name in names:
                destination = output / f"seed_{seed}" / name
                reuse = source / f"seed_{seed}/paper_reference_full_training"
                if name == "paper_fixed" and (reuse / "model.joblib").exists():
                    training = read(reuse / "training.json")
                    if training["training_rows"] != len(data.y) or training["rounds"] != 500:
                        raise ValueError("Reusable reference is not the full published-training fit.")
                    bundle = load_multiclass_bundle(reuse / "model.joblib")
                    training.update(reused_source_reference=True)
                else:
                    trial = TrainingTrial(**settings[name]["trial"])
                    LOGGER.info("Final fit before test access: %s seed %s", name, seed)
                    bundle, training = refit_trial(data.x, data.y, data.encoder, trial, settings[name]["rounds"], config, seed)
                save_joblib_atomic(destination / "model.joblib", bundle)
                write_report(destination / "training.json", training)
                report["models_saved"].append(str((destination / "model.joblib").resolve()))
                write_report(path, report)
                del bundle
                gc.collect()
        if object_digest(read(output / "protocol.json")) != report["protocol_sha256"]:
            raise ValueError("The frozen final protocol changed during training.")
        # No test member is extracted or parsed before this point.
        LOGGER.info("All final models saved; opening the once-only published test")
        report["test_access_started"] = True
        write_report(path, report)
        test_path = extract_frozen_test(archive, record_path, test_directory, config.num_classes)
        features = list(data.x.columns)
        del data.x
        gc.collect()
        report["test_csv_opened"] = True
        write_report(path, report)
        frame = pd.read_csv(test_path, dtype={**{c: "float32" for c in features}, "label": "string"})
        if list(frame.columns) != [*features, "label"] or set(frame.label) != set(classes):
            raise ValueError("Published test schema/labels differ from frozen training.")
        x_test, y_test = frame[features], data.encoder.transform(frame.label)
        if not np.isfinite(x_test.to_numpy()).all():
            raise ValueError("Published test contains nonfinite features.")
        report.update(status="evaluating_frozen_models", test_csv_opened=True, test_rows=len(y_test))
        write_report(path, report)
        for seed in frozen["seeds"]:
            for name in names:
                folder = output / f"seed_{seed}" / name
                bundle = load_multiclass_bundle(folder / "model.joblib")
                predicted, timing = _cpu_timing(bundle, x_test, timing_config)
                # save_scores' role is replaced explicitly: this is a final test,
                # never a source of guardrail/parameter/seed selection.
                from .evaluation import save_scores
                row = save_scores(folder / "test", classes, y_test, predicted, split="test",
                                  role="Once-only final evaluation of models/settings frozen before test access.")
                row.update(**timing, candidate=name, seed=seed,
                           training_seconds=read(folder / "training.json")["training_seconds"])
                report["results"].append(row)
                write_report(folder / "test/metrics.json", row)
                write_report(path, report)
                del bundle
                gc.collect()
        frame = pd.DataFrame(report["results"])
        summary = []
        for name, group in frame.groupby("candidate", sort=False):
            item = {"candidate": name, "seeds": len(group)}
            for metric in ("macro_f1", "weighted_f1", "accuracy", "benign_false_positive_rate", "inference_us_per_sample"):
                item[f"{metric}_mean"] = float(group[metric].mean())
                item[f"{metric}_std"] = float(group[metric].std(ddof=0))
            summary.append(item)
        write_table(output / "test_comparison.csv", frame)
        write_table(output / "test_summary.csv", pd.DataFrame(summary))
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(path, report)
    return report
