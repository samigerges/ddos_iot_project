"""Freeze both task choices before either published test is opened."""

from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

from ddos_iot.paths import PROJECT_ROOT
from .paper_reference import paper_reference_descriptor
from .storage import file_digest, object_digest, write_report
from .study_advanced_final import evaluate_advanced_test
from .study_final import evaluate_study_test
from .study_uncertainty import report_test_uncertainty


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _environment():
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost", "scipy", "joblib", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "versions": versions}


def assess_recall_policy(rows, reference_rows, labels, *, maximum_recall_loss=0., scope="each_seed"):
    """Compare paired seeds under an explicitly declared development recall policy."""
    if not labels or scope not in ("mean", "each_seed") or not math.isfinite(maximum_recall_loss) or not 0 <= maximum_recall_loss <= 1:
        raise ValueError("Declare a valid recall-loss fraction and scope.")
    seeds = [42, 123, 2024]
    if sorted(r["seed"] for r in rows) != seeds or sorted(r["seed"] for r in reference_rows) != seeds:
        raise ValueError("Policy assessment requires exactly the three paired seeds.")
    references = {r["seed"]: r for r in reference_rows}
    deltas = {label: [r["recall_by_class"][label] - references[r["seed"]]["recall_by_class"][label] for r in rows] for label in labels}
    means = {label: sum(values) / 3 for label, values in deltas.items()}
    floors_pass = all((means[label] if scope == "mean" else min(values)) >= -maximum_recall_loss - 1e-12 for label, values in deltas.items())
    fpr_pass = all(r["benign_false_positive_rate"] <= references[r["seed"]]["benign_false_positive_rate"] + 1e-12 for r in rows)
    paired_gains = [r["macro_f1"] - references[r["seed"]]["macro_f1"] for r in rows]
    return {"accepted": floors_pass and fpr_pass and sum(paired_gains) > 0,
        "scope": scope, "maximum_recall_loss": maximum_recall_loss,
        "mean_recall_deltas": means, "worst_seed_recall_deltas": {k: min(v) for k, v in deltas.items()},
        "paired_macro_f1_gains": paired_gains, "mean_macro_f1_gain": sum(paired_gains) / 3,
        "benign_fpr_nonincreasing_every_seed": fpr_pass, "recall_floors_pass": floors_pass}


def select_under_recall_policy(evidence, candidate, output, *, maximum_recall_loss=0., scope="each_seed", policy_note):
    """Promote existing frozen evidence under the user's settled policy, without tuning."""
    report = _read(evidence / "report.json")
    previous = _read(evidence / "selection.json")
    if report["status"] != "complete" or object_digest(previous) != report["selection_sha256"] or report.get("test_file_read", False):
        raise ValueError("Require completed unchanged development evidence.")
    source = Path(report["source_study"])
    if (source / "test_evaluation_v1").exists():
        raise ValueError("Set the recall policy before any final test evaluation.")
    base_folder = Path(report["base_confirmation"])
    base = _read(base_folder / "report.json")
    if base["status"] != "complete" or object_digest(_read(base_folder / "selection.json")) != base["selection_sha256"]:
        raise ValueError("Base confirmation changed.")
    advanced = Path(report["advanced_source"])
    frozen = _read(advanced / "selection.json")
    if object_digest(frozen) != report["advanced_source_selection_sha256"]:
        raise ValueError("Advanced settings changed after their measurement.")
    pool = frozen.get("candidates", frozen.get("single_model_conditions", frozen.get("conditions", {})))
    if candidate not in pool:
        raise ValueError("The candidate was not frozen before validation.")
    rows = [r for r in report.get("validation_results", report["results"]) if r["candidate"] == candidate]
    reference = [r for r in base["results"] if r["candidate"] == "paper_fixed"]
    parent = _read(source / f"seed_{base['source_seed']}/selection.json")
    if object_digest(parent) != base["source_selection_sha256"] or report["source_selection_sha256"] != base["source_selection_sha256"]:
        raise ValueError("Underlying base choices changed.")
    assessment = assess_recall_policy(rows, reference, parent["protected_labels"], maximum_recall_loss=maximum_recall_loss, scope=scope)
    base_choice = _read(base_folder / "selection.json")["candidate"]
    base_mean = sum(r["macro_f1"] for r in base["results"] if r["candidate"] == base_choice) / 3
    if sum(r["macro_f1"] for r in rows) / 3 <= base_mean:
        assessment["accepted"] = False
    if not assessment["accepted"]:
        raise ValueError("This frozen candidate does not satisfy the declared policy.")
    if not policy_note.strip():
        raise ValueError("Record why this policy was chosen before test access.")
    kind = "bias" if "single_model_conditions" in frozen else "stack" if "conditions" in frozen else "specialist"
    selection = {"candidate": candidate, "advanced_confirmed": True, "model_kind": kind,
        "advanced_source_selection": frozen, "seeds": [42, 123, 2024],
        "base_confirmation": str(base_folder.resolve()), "advanced_source": str(advanced.resolve()),
        "source_selection_sha256": base["source_selection_sha256"], "comparison_only": False,
        "test_file_read": False, "recall_policy": {**assessment, "note": policy_note},
        "rule": "Policy chosen from completed development comparisons before test; original strict guard flags retained unchanged."}
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "selection.json", selection)
    write_report(output / "report.json", {**report, "status": "complete", "results": rows,
        "validation_results": rows, "comparison_only": False, "advanced_confirmed": True,
        "selected_candidate": candidate, "selection_sha256": object_digest(selection),
        "policy_evidence": str(evidence.resolve()), "policy_evidence_report_sha256": file_digest(evidence / "report.json"),
        "recall_policy": selection["recall_policy"]})
    return selection


def freeze_research_protocol(choices, output, *, archive, record, recall_policy,
                             paper_python, development_report):
    """Choices are completed confirmation directories, never partial comparisons."""
    if set(choices) != {8, 34}:
        raise ValueError("Freeze both 8-class and 34-class choices together.")
    if not recall_policy or _read(development_report)["status"] != "complete":
        raise ValueError("Finish development comparisons and declare the recall policy first.")
    tasks, pins = {}, {}
    environment = _environment()

    def pin(path):
        path = Path(path).resolve()
        pins[str(path)] = file_digest(path)

    for classes, confirmation in sorted(choices.items()):
        confirmation = Path(confirmation).resolve()
        report = _read(confirmation / "report.json")
        selection = _read(confirmation / "selection.json")
        if report["status"] != "complete" or report["selection_sha256"] != object_digest(selection):
            raise ValueError("Require complete, unchanged confirmation choices.")
        if selection.get("comparison_only") or report.get("comparison_only"):
            raise ValueError("Comparison-only evidence needs an explicit policy selection before final evaluation.")
        if "advanced_confirmed" in selection and not selection["advanced_confirmed"]:
            raise ValueError("Use the base confirmation directly when no advanced model was accepted.")
        if selection["seeds"] != [42, 123, 2024] or selection.get("test_file_read", False):
            raise ValueError("Require all three predeclared seeds and no prior test selection.")
        candidate = selection["candidate"]
        rows = [row for row in report["results"] if row["candidate"] == candidate]
        if len(rows) != 3 or sorted(row["seed"] for row in rows) != [42, 123, 2024]:
            raise ValueError("The selected candidate must have exactly three completed seed results.")
        source = Path(report["source_study"]).resolve()
        original = _read(source / "report.json")
        if (original.get("python", environment["python"]) != environment["python"] or
            any(environment["versions"].get(name) != value for name, value in original.get("versions", {}).items())):
            raise ValueError("Final environment differs from the development reference.")
        config = _read(source / "config.json")
        if config["num_classes"] != classes:
            raise ValueError("Confirmation task/class count mismatch.")
        test_directory = PROJECT_ROOT / f"data/raw/paper_v3/final_tests/{classes}class"
        evaluation = source / "test_evaluation_v1"
        if test_directory.exists() or evaluation.exists():
            raise FileExistsError("Final test access/evaluation has already started.")
        reference = PROJECT_ROOT / f"outputs/studies/paper_fidelity_v3/{classes}class"
        descriptor = paper_reference_descriptor(reference)
        for path in (confirmation / "report.json", confirmation / "selection.json",
                     source / "config.json", source / "report.json",
                     source / f"seed_{report['source_seed']}/selection.json", reference / "model.joblib"):
            pin(path)
        for seed in selection["seeds"]:
            cache = source / f"seed_{seed}/paper_reference_full_training"
            if (cache / "model.joblib").exists():
                pin(cache / "model.joblib")
                pin(cache / "training.json")
        if "advanced_source" in selection:
            pin(Path(selection["advanced_source"]) / "selection.json")
        tasks[str(classes)] = {"confirmation": str(confirmation), "candidate": candidate,
            "evaluation": str(evaluation), "test_directory": str(test_directory),
            "paper_reference": descriptor, "development_macro_f1_mean": sum(r["macro_f1"] for r in rows) / 3}
    for path in (archive, record, development_report):
        pin(path)
    # Bind the implementation used to execute final fitting, extraction and scoring.
    for relative in ("study_protocol.py", "study_final.py", "study_advanced_final.py",
                     "study_uncertainty.py", "study_training.py", "study_models.py", "paper_reference.py",
                     "study.py", "study_specialist.py", "study_data.py", "evaluation.py", "decisions.py", "reference_data.py"):
        pin(Path(__file__).with_name(relative))
    pin(PROJECT_ROOT / "src/ddos_iot/multiclass.py")
    for relative in ("tmp/evaluate_frozen_research_worker.py", "tmp/benchmark_selected_confirmation.py"):
        helper = PROJECT_ROOT / relative
        if helper.exists():
            pin(helper)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "tasks": tasks,
        "archive": str(Path(archive).resolve()), "record": str(Path(record).resolve()),
        "paper_python": str(Path(paper_python).resolve()), "recall_policy": recall_policy,
        "environment": environment,
        "files_sha256": pins, "test_used_for_selection": False,
        "final_latency_repeats_after_full_warmup": 1,
        "uncertainty": {"model_seed": 42, "repetitions": 2000, "bootstrap_seed": 2024,
            "method": "Paired true-label-stratified percentile 95% bootstrap; descriptive, no post-test selection."},
        "rule": "Both task choices frozen before either test. Evaluate every declared seed/reference; no test-driven model, seed, threshold or policy changes."}
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "protocol.json", protocol)
    write_report(output / "freeze.json", {"protocol_sha256": object_digest(protocol), "test_access_started": False})
    return protocol


def verify_research_protocol(folder):
    protocol = _read(folder / "protocol.json")
    if object_digest(protocol) != _read(folder / "freeze.json")["protocol_sha256"]:
        raise ValueError("The global final protocol changed after freezing.")
    if protocol["environment"] != _environment():
        raise ValueError("The final execution environment changed after freezing.")
    for name, expected in protocol["files_sha256"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError(f"A frozen input changed: {name}")
    return protocol


def evaluate_research_protocol(folder):
    """Serial evaluation; completed stages may be reused, partial ones need inspection."""
    protocol = verify_research_protocol(folder)
    state = {"status": "running", "protocol_sha256": object_digest(protocol), "tasks": {}}
    write_report(folder / "report.json", state)
    try:
        for classes in sorted(protocol["tasks"], key=int):
            task = protocol["tasks"][classes]
            verify_research_protocol(folder)
            output = Path(task["evaluation"])
            if output.exists():
                if not (output / "report.json").exists() or _read(output / "report.json")["status"] != "complete":
                    raise RuntimeError("Inspect the preserved incomplete final evaluation; do not overwrite it.")
            else:
                selection = _read(Path(task["confirmation"]) / "selection.json")
                evaluate = evaluate_advanced_test if selection.get("advanced_confirmed") else evaluate_study_test
                evaluate(Path(task["confirmation"]), output, archive=Path(protocol["archive"]),
                    record_path=Path(protocol["record"]), test_directory=Path(task["test_directory"]),
                    paper_reference=Path(task["paper_reference"]["directory"]), latency_repeats=1)
            paper_output = output / "paper_software_reference"
            if not paper_output.exists():
                command = [protocol["paper_python"], "-B", str(PROJECT_ROOT / "src/score_published_reference.py"),
                    "--reference", task["paper_reference"]["directory"],
                    "--test-csv", str(Path(task["test_directory"]) / f"{classes}class_test_23features.csv"),
                    "--protocol", str(output / "protocol.json"), "--output-dir", str(paper_output)]
                with (folder / f"{classes}class_paper_reference.log").open("wb") as log:
                    subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if _read(paper_output / "report.json")["status"] != "complete":
                raise RuntimeError("Inspect the preserved incomplete software-matched reference evaluation.")
            uncertainty = output / "paired_uncertainty_seed42"
            if not uncertainty.exists():
                reference = output / "seed_42/paper_fixed/test/predictions.npz"
                candidate = output / f"seed_42/{task['candidate']}/test/predictions.npz"
                report_test_uncertainty(reference, candidate, uncertainty)
            if _read(uncertainty / "report.json")["status"] != "complete":
                raise RuntimeError("Inspect the preserved incomplete uncertainty analysis.")
            state["tasks"][classes] = {"status": "complete", "evaluation": str(output)}
            write_report(folder / "report.json", state)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(folder / "report.json", state)
    return state
