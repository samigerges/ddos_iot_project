"""Final published-test evaluation of a genuinely confirmed advanced model."""

from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle
from .evaluation import save_scores
from .storage import object_digest, write_report, write_table
from .study_config import TrainingTrial, load_study_config
from .study_data import load_study_data
from .study_final import evaluate_study_test, extract_frozen_test
from .study_models import SpecialistBundle, StackBundle
from .study_specialist import fit_specialist
from .study_training import refit_trial


def evaluate_advanced_test(confirmation: Path, output: Path, *, archive: Path, record_path: Path,
                            test_directory: Path, latency_repeats=1, paper_reference=None):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    report0 = read(confirmation / "report.json")
    frozen = read(confirmation / "selection.json")
    if report0["status"] != "complete" or object_digest(frozen) != report0["selection_sha256"]:
        raise ValueError("Require a complete frozen advanced confirmation.")
    if not frozen["advanced_confirmed"]:
        return evaluate_study_test(Path(frozen["base_confirmation"]), output, archive=archive,
                                  record_path=record_path, test_directory=test_directory,
                                  latency_repeats=latency_repeats, paper_reference=paper_reference)
    if type(latency_repeats) is not int or latency_repeats < 1:
        raise ValueError("Invalid timing repeat count.")
    source = Path(report0["source_study"])
    config = load_study_config(source / "config.json")
    if object_digest(read(source / "config.json")) != read(source / "report.json")["config_sha256"]:
        raise ValueError("Source study configuration changed.")
    base_confirmation = read(Path(frozen["base_confirmation"]) / "report.json")
    seed0 = base_confirmation["source_seed"]
    parent_selection = read(source / f"seed_{seed0}/selection.json")
    if object_digest(parent_selection) != frozen["source_selection_sha256"]:
        raise ValueError("Underlying source parameters changed.")
    advanced_source = Path(frozen["advanced_source"])
    advanced_selection = read(advanced_source / "selection.json")
    if object_digest(advanced_selection) != report0["advanced_source_selection_sha256"]:
        raise ValueError("Advanced settings changed after confirmation.")
    if test_directory.exists():
        raise FileExistsError("The test input already exists; repeated evaluation is not automatic.")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"advanced_confirmation_sha256": report0["selection_sha256"], "candidate": frozen["candidate"],
                "model_kind": frozen["model_kind"], "settings": frozen["advanced_source_selection"],
                "seeds": frozen["seeds"], "test_used_for_selection": False,
                "timing_repeats_after_full_warmup": latency_repeats,
                "meta_scope": "Confirmed development-fitted meta and frozen biases are retained; base models refit on all published training rows."}
    if paper_reference is not None:
        from .paper_reference import paper_reference_descriptor
        protocol["additional_paper_reference"] = paper_reference_descriptor(paper_reference)
    write_report(output / "protocol.json", protocol)
    report = {"status": "fitting_all_final_models", "protocol_sha256": object_digest(protocol),
              "models_saved": [], "results": [], "test_csv_opened": False, "test_used_for_selection": False}
    path = output / "report.json"
    write_report(path, report)
    started = time.perf_counter()
    try:
        data = load_study_data(config)
        if paper_reference is not None and protocol["additional_paper_reference"]["source_sha256"] != data.metadata["source_sha256"]:
            raise ValueError("Paper reference was fitted to a different training subset.")
        features, classes = list(data.x.columns), data.encoder.classes_
        for seed in frozen["seeds"]:
            cached = {}
            required = ["paper_fixed"]
            if frozen["model_kind"] == "stack": required += advanced_selection["base_names"]
            else: required.append(advanced_selection["base_candidate"])
            training_by_name = {}
            for name in dict.fromkeys(required):
                reuse = source / f"seed_{seed}/paper_reference_full_training"
                if name == "paper_fixed" and (reuse / "model.joblib").exists():
                    bundle = load_multiclass_bundle(reuse / "model.joblib")
                    training = read(reuse / "training.json")
                    if training["training_rows"] != len(data.y) or training["rounds"] != 500:
                        raise ValueError("Invalid full-training reference cache.")
                else:
                    settings = parent_selection["trials"][name]
                    bundle, training = refit_trial(data.x, data.y, data.encoder, TrainingTrial(**settings["trial"]),
                                                   settings["rounds"], config, seed)
                cached[name], training_by_name[name] = bundle, training
            reference_folder = output / f"seed_{seed}/paper_fixed"
            save_joblib_atomic(reference_folder / "model.joblib", cached["paper_fixed"])
            write_report(reference_folder / "training.json", training_by_name["paper_fixed"])
            report["models_saved"].append(str((reference_folder / "model.joblib").resolve()))
            if frozen["model_kind"] == "bias":
                bundle = cached[advanced_selection["base_candidate"]]
                bundle.decision_bias = np.asarray(advanced_selection["single_model_conditions"][frozen["candidate"]])
                training = training_by_name[advanced_selection["base_candidate"]]
            elif frozen["model_kind"] == "specialist":
                settings = advanced_selection["candidates"][frozen["candidate"]]
                group = np.asarray(advanced_selection["group_ids"])
                specialist, training = fit_specialist(data.x, data.y, group, config, seed=seed,
                                                       power=settings["weight_power"], rounds=settings["rounds"], depth=settings.get("depth", 6))
                specialist.set_params(device="cpu")
                decision_bias = np.asarray(settings["decision_bias"]) if settings.get("decision_bias") is not None else None
                bundle = SpecialistBundle(cached[advanced_selection["base_candidate"]], specialist, group, decision_bias)
                training["training_seconds"] += training_by_name[advanced_selection["base_candidate"]]["training_seconds"]
            else:
                row = next(r for r in report0["results"] if r["seed"] == seed)
                confirmed_bundle = joblib.load(Path(row["artifact_folder"]) / "model.joblib")
                if not isinstance(confirmed_bundle, StackBundle) or list(confirmed_bundle.encoder.classes_) != list(classes):
                    raise ValueError("Confirmed meta/model class mapping changed.")
                bases = tuple(cached[name] for name in advanced_selection["base_names"])
                bundle = StackBundle(bases, confirmed_bundle.meta, np.asarray(advanced_selection["conditions"][frozen["candidate"]]), config.n_jobs)
                training = {"training_seconds": sum(training_by_name[n]["training_seconds"] for n in advanced_selection["base_names"]),
                            "meta_reused_from_confirmation": str(row["artifact_folder"]),
                            "meta_training_seconds_not_repeated_here": row["training_seconds"],
                            "scope": protocol["meta_scope"]}
            destination = output / f"seed_{seed}" / frozen["candidate"]
            save_joblib_atomic(destination / "model.joblib", bundle)
            write_report(destination / "training.json", training)
            report["models_saved"].append(str((destination / "model.joblib").resolve()))
            write_report(path, report)
            del bundle, cached
            gc.collect()
        if object_digest(read(output / "protocol.json")) != report["protocol_sha256"]:
            raise ValueError("Frozen test protocol changed during fitting.")
        report["test_access_started"] = True
        write_report(path, report)
        test_path = extract_frozen_test(archive, record_path, test_directory, config.num_classes)
        report["test_csv_opened"] = True
        write_report(path, report)
        del data.x
        gc.collect()
        frame = pd.read_csv(test_path, dtype={**{c: "float32" for c in features}, "label": "string"})
        if list(frame.columns) != [*features, "label"] or set(frame.label) != set(classes):
            raise ValueError("Published test schema/labels changed.")
        x, y = frame[features], data.encoder.transform(frame.label)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("Test features must be finite.")
        for seed in frozen["seeds"]:
            for name in ("paper_fixed", frozen["candidate"]):
                folder = output / f"seed_{seed}" / name
                bundle = joblib.load(folder / "model.joblib")
                predicted = bundle.encoder.transform(bundle.predict(x))
                runs = []
                for _ in range(latency_repeats):
                    before = time.perf_counter(); current = bundle.encoder.transform(bundle.predict(x)); runs.append(time.perf_counter() - before)
                    np.testing.assert_array_equal(current, predicted)
                row = save_scores(folder / "test", classes, y, predicted, split="test",
                                  role="Once-only evaluation of models, seeds and decisions frozen before test access.")
                row.update(candidate=name, seed=seed, inference_us_per_sample=float(np.median(runs) * 1e6 / len(x)),
                           inference_seconds_runs=runs, full_cohort_warmup=True, inference_device="cpu",
                           training_seconds=read(folder / "training.json")["training_seconds"],
                           latency_scope="Complete published-format bundle prediction and decision; no raw feature extraction.")
                report["results"].append(row)
                write_report(path, report)
                del bundle
                gc.collect()
        write_table(output / "test_comparison.csv", pd.DataFrame(report["results"]))
        summary_rows = []
        for name, group in pd.DataFrame(report["results"]).groupby("candidate", sort=False):
            item = {"candidate": name, "seeds": len(group)}
            for metric in ("macro_f1", "weighted_f1", "accuracy", "benign_false_positive_rate", "inference_us_per_sample", "training_seconds"):
                item[f"{metric}_mean"] = float(group[metric].mean())
                item[f"{metric}_std"] = float(group[metric].std(ddof=0))
            summary_rows.append(item)
        summary = pd.DataFrame(summary_rows)
        write_table(output / "test_summary.csv", summary)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(path, report)
    return report
