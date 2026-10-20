"""Read-only multiclass run discovery and explicit background training launches.

Discovery reads JSON only; it never unpickles a model or starts training.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ddos_iot.artifacts import atomic_output
from ddos_iot.paths import MODELS_DIR, PROJECT_ROOT


MODEL_NAMES = ("xgboost", "lightgbm", "catboost")


def read_report(path: Path) -> dict[str, Any]:
    """A writer may be refreshing a legacy, non-atomic report while we read it."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_job(path: Path, job: dict[str, Any]) -> None:
    with atomic_output(path) as stream:
        stream.write(json.dumps(job, indent=2).encode("utf-8"))


def run_class_count(folder: Path) -> int | None:
    for filename in ("data_report.json", "model_comparison_report.json", "dashboard_job.json"):
        report = read_report(folder / filename)
        args = report.get("args")
        count = report.get("class_count", args.get("num_classes") if isinstance(args, dict) else None)
        if count in (8, 34):
            return count
    # Older bundles predate class_count metadata, but have a label mapping.
    mapping = read_report(folder / "label_mapping.json")
    if len(mapping) in (8, 34):
        return len(mapping)
    if read_report(folder / "raw_to_family_mapping.json"):
        return 8
    return None


def discover_runs(num_classes: int, root: Path = MODELS_DIR) -> list[Path]:
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                   and run_class_count(p) == num_classes),
                  key=lambda p: p.stat().st_mtime_ns, reverse=True)


def saved_models(folder: Path) -> list[Path]:
    return [folder / name / "model.joblib" for name in MODEL_NAMES
            if (folder / name / "model.joblib").is_file()]


def tail_log(path: Path, limit: int = 24_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def build_training_command(num_classes: int, options: list[str]) -> list[str]:
    """Use the CLI's actual defaults/validation rather than a second training API."""
    from multiclass import parse_args

    if num_classes not in (8, 34):
        raise ValueError("Choose 8 or 34 classes.")
    if "--num-classes" in options or any(s.startswith("--num-classes=") for s in options):
        raise ValueError("The task selector controls the class count.")
    arguments = ["--num-classes", str(num_classes), *options]
    args = parse_args(arguments)
    if not args.models:
        raise ValueError("Select at least one model.")
    return [sys.executable, "-B", "-u", str(PROJECT_ROOT / "src" / "multiclass.py"), *arguments]


def start_training_job(command: list[str], *, models_root: Path = MODELS_DIR) -> Path:
    """Reserve a NEW run directory, then launch a hidden, shell-free worker.

    Existing CLI jobs are never attached to, stopped, or overwritten. A fresh
    directory also prevents duplicate launches when a button is clicked twice.
    """
    from multiclass import parse_args

    expected = [sys.executable, "-B", "-u", str(PROJECT_ROOT / "src" / "multiclass.py")]
    if command[:4] != expected:
        raise ValueError("Only the project's multiclass training entry point is allowed.")
    args = parse_args(command[4:])
    root = models_root.resolve()
    folder = args.output_path
    folder = (PROJECT_ROOT / folder).resolve() if not folder.is_absolute() else folder.resolve()
    if folder.parent != root:
        raise ValueError(f"Dashboard runs must be new direct subdirectories of {root}.")
    folder.mkdir(parents=True, exist_ok=False)
    job = {"status": "queued", "class_count": args.num_classes,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "command": command, "arguments": command[4:]}
    write_job(folder / "dashboard_job.json", job)
    try:
        with (folder / "dashboard_training.log").open("ab", buffering=0) as log:
            subprocess.Popen(
                [sys.executable, "-B", "-u", str(PROJECT_ROOT / "src" / "dashboard_worker.py"),
                 "--run-dir", str(folder)],
                cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
    except Exception as exc:
        job.update(status="failed", error=str(exc))
        write_job(folder / "dashboard_job.json", job)
        raise
    return folder
