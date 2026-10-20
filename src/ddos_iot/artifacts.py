"""Atomic artifact writes: incomplete files never replace a completed artifact."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

import joblib


@contextmanager
def atomic_output(path: Path) -> Iterator[BinaryIO]:
    """Flush a same-directory temporary file, then atomically replace its target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # Only our own unique temporary file is removed after a failed write.
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_joblib_atomic(path: Path, value: object) -> None:
    """Serialize into one temporary file; publish only after serialization succeeds."""
    with atomic_output(path) as stream:
        joblib.dump(value, stream)
