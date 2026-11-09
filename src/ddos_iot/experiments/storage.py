"""Small atomic reports and array exports for auditable experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ddos_iot.artifacts import atomic_output
from ddos_iot.utils import make_json_safe


def write_report(path: Path, value: Any) -> None:
    with atomic_output(path) as stream:
        stream.write((json.dumps(make_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def write_arrays(path: Path, **arrays: np.ndarray) -> None:
    with atomic_output(path) as stream:
        np.savez_compressed(stream, **arrays)


def write_table(path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    with atomic_output(path) as stream:
        stream.write(frame.to_csv(index=index).encode("utf-8"))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(make_json_safe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def close_memmap(value: np.ndarray | None) -> None:
    """Release Windows file handles explicitly, including exception traceback paths."""
    if isinstance(value, np.memmap) and not value._mmap.closed:
        value.flush()
        value._mmap.close()
