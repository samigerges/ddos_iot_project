"""One-time, restart-aware orchestration; GPU fits and timing stay sequential."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ddos_iot.paths import PROJECT_ROOT
from .storage import object_digest, write_report
from .study_config import load_study_config

LOGGER = logging.getLogger(__name__)


def run_research_pipeline(config_paths, state_directory, *, paper_python=None, wait_existing=False, final_test=False):
    state_directory.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "stages": {}, "final_test_requested": final_test,
             "gpu_policy": "Sequential fitting and inference timing; no concurrent research model jobs."}
    state_path = state_directory / "state.json"
    write_report(state_path, state)

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def invoke(key, script, arguments, report_path, *, interpreter=None, may_wait=False):
        if report_path.exists():
            while True:
                snapshot = read(report_path)
                status = snapshot.get("status")
                if status in ("complete", "skipped"):
                    state["stages"][key] = {"status": status, "reused_completed_stage": True, "report": str(report_path)}
                    write_report(state_path, state)
                    return snapshot
                if status in ("failed", "interrupted", "partial_failure") or not (may_wait and wait_existing):
                    raise RuntimeError(f"Stage {key} already exists with status {status}; inspect preserved artifacts instead of overwriting.")
                state["stages"][key] = {"status": "waiting_for_existing_run", "report": str(report_path)}
                write_report(state_path, state)
                time.sleep(5)
        logfile = state_directory / f"{key}.log"
        command = [str(interpreter or sys.executable), "-u", "-B", str(PROJECT_ROOT / "src" / script), *map(str, arguments)]
        state["stages"][key] = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                                  "log": str(logfile.resolve()), "command": command}
        write_report(state_path, state)
        LOGGER.info("Starting %s; log %s", key, logfile)
        with logfile.open("wb") as stream:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while process.poll() is None:
                    time.sleep(1)
            except BaseException:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
        if process.returncode != 0:
            raise RuntimeError(f"Stage {key} exited {process.returncode}; inspect {logfile}.")
        snapshot = read(report_path)
        if snapshot.get("status") not in ("complete", "skipped"):
            raise RuntimeError(f"Stage {key} ended without a complete/skipped report: {snapshot.get('status')}.")
        state["stages"][key].update(status=snapshot["status"], report=str(report_path.resolve()),
                                    completed_utc=datetime.now(timezone.utc).isoformat())
        write_report(state_path, state)
        LOGGER.info("Completed %s: %s", key, snapshot["status"])
        return snapshot

    try:
        studies = []
        # Finish all externally running base studies before launching another GPU fit.
        for config_path in config_paths:
            config = load_study_config(config_path)
            root = config.output_path
            key = f"{config.num_classes}class"
            invoke(key + "_base", "study_multiclass.py", ["run", "--config", config_path], root / "report.json", may_wait=True)
            studies.append((key, config, root))
        if paper_python is not None:
            for key, config, root in studies:
                reference = PROJECT_ROOT / "outputs/studies/paper_fidelity_v3" / key
                invoke(key + "_paper_software", "train_published_reference.py", ["--classes", config.num_classes, "--output-dir", reference],
                       reference / "report.json", interpreter=paper_python)
        for key, config, root in studies:
            invoke(key + "_collisions", "audit_study_collisions.py", ["--run-dir", root], root / "collision_audit/report.json")
            invoke(key + "_bias", "bias_multiclass_study.py", ["--run-dir", root], root / "bias_v1/report.json")
            invoke(key + "_specialist", "specialize_multiclass_study.py", ["--run-dir", root], root / "specialist_v1/report.json")
            invoke(key + "_stack", "stack_multiclass_study.py", ["--run-dir", root], root / "stack_v1/report.json")
            advanced_runs = {name: root / f"{name}_v1" for name in ("bias", "specialist", "stack")}
            expanded = root / "specialist_expanded_v2"
            invoke(key + "_specialist_expanded", "specialize_multiclass_study.py",
                   ["--run-dir", root, "--output-dir", expanded, "--expanded-fixed"], expanded / "report.json")
            advanced_runs["specialist"] = expanded
            meta_training = root / "stack_v1/meta_training.json"
            if not meta_training.exists(): meta_training = root / "stack_v1/optimization_failure.json"
            if meta_training.exists():
                optimizer = read(meta_training)
                if optimizer["hit_iteration_limit"] and optimizer.get("solver", "lbfgs") == "lbfgs":
                    converged_stack = root / "stack_optimizer_v2"
                    invoke(key + "_stack_optimizer", "stack_multiclass_study.py",
                           ["--run-dir", root, "--output-dir", converged_stack,
                            "--meta-solver", "newton-cg", "--meta-max-iter", 100], converged_stack / "report.json")
                    advanced_runs["stack"] = converged_stack
            confirmation = root / "confirmation_v1"
            base = invoke(key + "_confirmation", "confirm_multiclass_study.py", ["--run-dir", root], confirmation / "report.json")
            for name in ("bias", "specialist", "stack"):
                destination = root / ("stack_optimizer_confirmation_v2" if advanced_runs[name].name == "stack_optimizer_v2" else
                                      "specialist_expanded_confirmation_v2" if advanced_runs[name].name == "specialist_expanded_v2" else f"{name}_confirmation_v1")
                invoke(key + "_confirm_" + name, "confirm_multiclass_advanced.py",
                       ["--confirmation", confirmation, "--advanced-run", advanced_runs[name], "--output-dir", destination], destination / "report.json")
            # Finish the cheap unconstrained comparison before final recall limits
            # are settled. Comparison-only results cannot enter final selection.
            bias_comparison = root / "bias_comparison_v1"
            invoke(key + "_bias_comparison", "confirm_multiclass_advanced.py",
                   ["--confirmation", confirmation, "--advanced-run", advanced_runs["bias"], "--output-dir", bias_comparison,
                    "--candidate", "xgb_oof_bias", "--compare-only"], bias_comparison / "report.json")
            expanded_report = read(expanded / "report.json")
            if expanded_report["status"] == "complete":
                expanded_choice = read(expanded / "selection.json")["best_macro"]
                observation = next(r for r in expanded_report["validation_results"] if r["candidate"] == expanded_choice)
                reference = next(r for r in base["results"] if r["seed"] == expanded_report["seed"] and r["candidate"] == "paper_fixed")
                if observation["macro_f1"] > reference["macro_f1"] and not observation["reference_constraints_pass"]:
                    destination = root / "specialist_expanded_comparison_v2"
                    invoke(key + "_specialist_expanded_comparison", "confirm_multiclass_advanced.py",
                           ["--confirmation", confirmation, "--advanced-run", expanded, "--output-dir", destination,
                            "--candidate", expanded_choice, "--compare-only"], destination / "report.json")
            stack_report = read(advanced_runs["stack"] / "report.json")
            if stack_report["status"] == "complete" and stack_report["validation_results"]:
                observation = max(stack_report["validation_results"], key=lambda row: row["macro_f1"])
                reference = next(r for r in base["results"] if r["seed"] == stack_report["seed"] and r["candidate"] == "paper_fixed")
                default_folder = root / ("stack_optimizer_confirmation_v2" if advanced_runs["stack"].name == "stack_optimizer_v2" else "stack_confirmation_v1")
                measured = read(default_folder / "report.json")
                already_measured = measured["status"] == "complete" and measured.get("candidate") == observation["candidate"]
                if observation["macro_f1"] > reference["macro_f1"] and not already_measured:
                    destination = root / ("stack_optimizer_comparison_v2" if advanced_runs["stack"].name == "stack_optimizer_v2" else "stack_comparison_v1")
                    invoke(key + "_stack_comparison", "confirm_multiclass_advanced.py",
                           ["--confirmation", confirmation, "--advanced-run", advanced_runs["stack"], "--output-dir", destination,
                            "--candidate", observation["candidate"], "--compare-only"], destination / "report.json")
            chosen = read(confirmation / "selection.json")["candidate"]
            candidates = [{"confirmation": str(confirmation.resolve()), "candidate": chosen,
                           "macro_f1_mean": float(np.mean([r["macro_f1"] for r in base["results"] if r["candidate"] == chosen]))}]
            # Include additional explicitly confirmed candidates before any test is opened.
            for path in sorted(root.glob("*_confirmation*/report.json")):
                snapshot = read(path)
                if snapshot.get("status") == "complete" and snapshot.get("advanced_confirmed"):
                    if Path(snapshot["source_study"]).resolve() != root.resolve():
                        raise ValueError("An advanced confirmation belongs to a different source study.")
                    candidates.append({"confirmation": str(path.parent.resolve()), "candidate": snapshot["selected_candidate"],
                                       "macro_f1_mean": float(np.mean([r["macro_f1"] for r in snapshot["results"]]))})
            winner = max(candidates, key=lambda r: r["macro_f1_mean"])
            decision = {"status": "development_choice", "selected": winner, "eligible_confirmations": candidates,
                        "test_used_for_selection": False,
                        "note": "Provisional until the separately declared final protocol is saved. No thresholds/parameters are retuned here."}
            decision_path = root / "final_choice_v1.json"
            if (root / "test_evaluation_v1").exists() and decision_path.exists() and object_digest(read(decision_path)) != object_digest(decision):
                raise ValueError("Development choice changed after test evaluation began; automatic reevaluation is forbidden.")
            write_report(decision_path, decision)
            if final_test:
                reference = PROJECT_ROOT / "outputs/studies/paper_fidelity_v3" / key
                arguments = ["--confirmation", winner["confirmation"], "--output-dir", root / "test_evaluation_v1",
                             "--test-directory", PROJECT_ROOT / "data/raw/paper_v3/final_tests" / key]
                if paper_python is not None: arguments += ["--paper-reference", reference]
                invoke(key + "_final_test", "evaluate_multiclass_study.py", arguments, root / "test_evaluation_v1/report.json")
                if paper_python is not None:
                    test_csv = PROJECT_ROOT / "data/raw/paper_v3/final_tests" / key / f"{config.num_classes}class_test_23features.csv"
                    invoke(key + "_score_paper_reference", "score_published_reference.py",
                           ["--reference", reference, "--test-csv", test_csv, "--protocol", root / "test_evaluation_v1/protocol.json",
                            "--output-dir", root / "test_evaluation_v1/paper_software_reference"],
                           root / "test_evaluation_v1/paper_software_reference/report.json", interpreter=paper_python)
        from .research_reporting import report_research
        report_research([root for _, _, root in studies], PROJECT_ROOT / "outputs/studies/comparison_v1")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(state_path, state)
    return state
