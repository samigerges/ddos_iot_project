"""Pinned-software paper reference, trained before any published-test access."""

import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle
from ddos_iot.multiclass import FeatureTransform, MulticlassBundle, predict_probabilities
from ddos_iot.paths import PROJECT_ROOT
from .evaluation import save_scores
from .storage import file_digest, write_report
from .study import _cpu_timing
from .study_config import TrainingStudyConfig, TrainingTrial
from .study_data import load_study_data
from .study_training import StudyProgress

EXPECTED = {"numpy": "2.1.2", "pandas": "2.3.0", "scikit-learn": "1.6.1", "xgboost": "3.0.2"}


def matched_environment():
    actual = {name: importlib.metadata.version(name) for name in EXPECTED}
    if platform.python_version() != "3.11.2" or actual != EXPECTED:
        raise ValueError("Use the isolated paper runtime: Python 3.11.2, NumPy 2.1.2, pandas 2.3.0, sklearn 1.6.1, XGBoost 3.0.2.")
    return {"python": platform.python_version(), **actual}


def train_paper_reference(classes: int, output: Path):
    environment = matched_environment()
    output.mkdir(parents=True, exist_ok=False)
    config = TrainingStudyConfig(num_classes=classes, catboost=False, weight_powers=(), output_path=output)
    write_report(output / "environment.json", environment)
    started = time.perf_counter()
    report = {"status": "loading", "num_classes": classes, "seed": 42, "environment": environment,
              "parameters": {"max_depth": 10, "n_estimators": 500, "learning_rate": .1,
                             "weights": "inverse frequency", "max_bin": 256, "subsample": 1., "colsample_bytree": 1.},
              "test_file_read": False, "scope": "Published selected-feature reference; hardware differs from authors. No test score until the final protocol freezes this model."}
    write_report(output / "report.json", report)
    try:
        data = load_study_data(config)
        write_report(output / "data_report.json", data.metadata)
        report["status"] = "training"
        write_report(output / "report.json", report)
        # Match the released selected-feature parameter dictionary, including
        # the author's default training thread policy. CPU timing is separately
        # constrained to two threads after fitting.
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        parameters = json.loads((PROJECT_ROOT / "outputs/baseline_audit_v1/author_settings.json").read_text(encoding="utf-8"))[f"{classes}class"]["dictionaries"]["xgb_params_selected"]["values"]
        parameters = {key: (42 if key == "random_state" else value) for key, value in parameters.items()}
        if (parameters["n_estimators"], parameters["max_depth"], parameters["learning_rate"]) != (500, 10, .1):
            raise ValueError("Pinned paper reference parameters changed.")
        matrix = np.ascontiguousarray(data.x.to_numpy(dtype=np.float32))
        counts = np.bincount(data.y, minlength=classes)
        sample_weight = (len(data.y) / (classes * counts))[data.y]
        model = XGBClassifier(**parameters, callbacks=[StudyProgress("paper_fidelity")])
        before = time.perf_counter()
        model.fit(matrix, data.y, sample_weight=sample_weight, verbose=False)
        fit_seconds = time.perf_counter() - before
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError("Paper GPU reference silently fell back to CPU.")
        model.set_params(callbacks=None)
        identity = StandardScaler(with_mean=False, with_std=False).fit(matrix[:1])
        bundle = MulticlassBundle("xgboost", model, FeatureTransform(list(data.x.columns), identity, {}, None, "published_23_identity"), data.encoder)
        training = {"training_rows": len(data.y), "rounds": 500, "seed": 42, "parameters": parameters,
                    "training_seconds": fit_seconds, "training_thread_policy": "Author default; n_jobs not explicitly set.",
                    "input_precision": "float32, matching XGBoost's numerical feature representation", "device": actual,
                    "test_file_read": False}
        save_joblib_atomic(output / "model.joblib", bundle)
        write_report(output / "training.json", training)
        report.update(status="complete", model_sha256=file_digest(output / "model.joblib"),
                      source_sha256=data.metadata["source_sha256"], rows=len(data.y),
                      training_seconds=training["training_seconds"])
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds_total"] = time.perf_counter() - started
        write_report(output / "report.json", report)
    return report


def paper_reference_descriptor(folder: Path):
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "complete" or file_digest(folder / "model.joblib") != report["model_sha256"]:
        raise ValueError("Pinned paper reference is incomplete or changed.")
    return {"directory": str(folder.resolve()), "model_sha256": report["model_sha256"],
            "source_sha256": report["source_sha256"], "environment": report["environment"],
            "num_classes": report["num_classes"], "seed": report["seed"],
            "evaluation_policy": "GPU reproduction metrics plus CPU comparison metrics; CPU throughput uses two threads."}


def score_paper_reference(folder: Path, test_csv: Path, protocol_path: Path, output: Path):
    matched_environment()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol.get("additional_paper_reference")
    descriptor = paper_reference_descriptor(folder)
    if expected != descriptor:
        raise ValueError("This paper reference was not declared in the frozen final protocol.")
    provenance = json.loads((test_csv.parent / "provenance.json").read_text(encoding="utf-8"))
    if provenance["member"] != test_csv.name or file_digest(test_csv) != provenance["sha256"]:
        raise ValueError("The extracted published test CSV changed since archive verification.")
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((folder / "data_report.json").read_text(encoding="utf-8"))
    columns = metadata["features"]
    frame = pd.read_csv(test_csv, dtype={**{c: "float32" for c in columns}, "label": "string"})
    if list(frame.columns) != [*columns, "label"] or set(frame.label) != set(metadata["classes"]):
        raise ValueError("Paper test does not match the pinned training representation.")
    bundle = load_multiclass_bundle(folder / "model.joblib")
    config = TrainingStudyConfig(num_classes=descriptor["num_classes"], catboost=False, latency_repeats=1)
    y = bundle.encoder.transform(frame.label)
    gpu_predictions = predict_probabilities(bundle.name, bundle.model, frame[columns], config.batch_size, device="gpu").argmax(axis=1)
    gpu_scores = save_scores(output / "gpu_reproduction", bundle.encoder.classes_, y, gpu_predictions, split="test",
                            role="Predeclared GPU classification metrics for the pinned paper reference; no GPU latency claim.")
    bundle.model.set_params(n_jobs=2)
    predicted, timing = _cpu_timing(bundle, frame[columns], config)
    result = save_scores(output, bundle.encoder.classes_, y, predicted, split="test",
                         role="Once-only software-matched paper-reference evaluation; parameters frozen before test access.")
    result.update(**timing, status="complete", reference=descriptor, test_used_for_selection=False,
                  gpu_reproduction_metrics=gpu_scores, gpu_cpu_prediction_disagreements=int(np.sum(gpu_predictions != predicted)),
                  inference_threads=2,
                  hardware_scope="CPU timing on local hardware; not a reproduction of the paper's GPU inference latency.")
    write_report(output / "report.json", result)
    return result
