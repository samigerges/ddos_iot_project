"""Run one dashboard-requested training job independently of Streamlit reruns."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from ddos_iot.dashboard_runs import read_report, write_job
from ddos_iot.utils import configure_logging


def run_job(folder: Path) -> int:
    path = folder / "dashboard_job.json"
    job = read_report(path)
    if job.get("status") != "queued" or not isinstance(job.get("arguments"), list):
        raise ValueError("This job is not queued; refusing to restart or overwrite it.")
    job.update(status="running", pid=os.getpid(), started_at_utc=datetime.now(timezone.utc).isoformat())
    write_job(path, job)
    code = 0
    try:
        from multiclass import parse_args
        from ddos_iot.multiclass import run_multiclass_training

        args = parse_args(job["arguments"])
        if args.output_path.resolve() != folder.resolve():
            raise ValueError("Job directory and training output do not match.")
        configure_logging(args.verbose)
        run_multiclass_training(args)
        job["status"] = "complete"
    except KeyboardInterrupt:
        job.update(status="interrupted", error="Training was interrupted.")
        code = 130
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        job.update(returncode=code, finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_job(path, job)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    raise SystemExit(run_job(parser.parse_args().run_dir))
