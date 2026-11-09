"""Verify the published archive and extract only its two multiclass train CSVs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import pandas as pd

from ddos_iot.attack_families import FAMILY_NAMES, RAW_TO_FAMILY
from .storage import write_report

LOGGER = logging.getLogger(__name__)
TRAIN_FILES = {"8class_train_23features.csv": 8, "34class_train_23features.csv": 34}


def verify_archive(archive: Path, record: dict) -> dict:
    item = next(f for f in record["files"] if f["key"] == "datasets.tar.xz")
    expected = item["checksum"]
    if not expected.startswith("md5:") or archive.stat().st_size != item["size"]:
        raise ValueError("Archive size/checksum specification does not match the published record.")
    md5, sha = hashlib.md5(), hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != expected.split(":", 1)[1]:
        raise ValueError("Archive failed the published MD5 checksum.")
    return {"path": str(archive.resolve()), "bytes": item["size"], "md5": md5.hexdigest(),
            "sha256": sha.hexdigest(), "version_record_id": record["id"], "url": item["links"]["self"]}


def inspect_training_subset(path: Path, classes: int, *, chunk_size: int = 100_000) -> dict:
    columns = list(pd.read_csv(path, nrows=0).columns)
    features = [c for c in columns if c != "label"]
    if "label" not in columns or len(features) != 23:
        raise ValueError("The published subset must contain label and exactly 23 numeric features.")
    counts: dict[str, int] = {}
    rows, invalid = 0, 0
    dtypes = {c: "float32" for c in features}
    dtypes["label"] = "string"
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunk_size):
        rows += len(chunk)
        valid = np.isfinite(chunk[features].to_numpy()).all(axis=1) & chunk.label.notna().to_numpy()
        invalid += int((~valid).sum())
        for label, n in chunk.loc[valid, "label"].value_counts().items():
            counts[str(label)] = counts.get(str(label), 0) + int(n)
    expected = set(FAMILY_NAMES) if classes == 8 else set(RAW_TO_FAMILY)
    if invalid or set(counts) != expected:
        raise ValueError(f"Published training subset contains invalid rows or unexpected labels: {path}")
    return {"path": str(path.resolve()), "class_count": classes, "rows": rows, "invalid_rows": invalid,
            "feature_count": len(features), "features": features, "class_counts": counts,
            "representation": "Published selected/scaled features; do not treat as the original 46-feature raw source.",
            "cv_warning": "The author selected these features using the entire published training split. Resplitting this CSV cannot reproduce fold-local feature selection.",
            "test_file_read": False}


def prepare_reference_training(archive: Path, record_path: Path, output: Path) -> dict:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    provenance = verify_archive(archive, record)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "extracting", "archive": provenance, "members": [], "training_subsets": {},
              "test_members_extracted": False, "test_csv_parsed": False, "model_trained": False,
              "scope": "Archive decompression traverses compressed bytes; only explicitly named training members are opened/extracted. Test CSV contents are not parsed or evaluated."}
    write_report(output / "preparation_report.json", report)
    found = set()
    try:
        with tarfile.open(archive, mode="r|xz") as source:
            for member in source:
                name = PurePosixPath(member.name).name
                report["members"].append({"name": member.name, "bytes": member.size})
                if name not in TRAIN_FILES:
                    continue
                # No archive paths or links are used as filesystem destinations.
                if not member.isfile() or name in found:
                    raise ValueError(f"Duplicate or non-regular training member: {member.name}")
                found.add(name)
                destination = output / name
                LOGGER.info("Extracting training member %s (%s bytes)", name, member.size)
                sha, written = hashlib.sha256(), 0
                with source.extractfile(member) as incoming, destination.open("xb") as outgoing:
                    while block := incoming.read(4 * 1024 * 1024):
                        written += len(block)
                        outgoing.write(block)
                        sha.update(block)
                if written != member.size:
                    raise ValueError("Extracted training member size differs from archive metadata.")
                report["training_subsets"][name] = {"sha256": sha.hexdigest(), "bytes": written}
                write_report(output / "preparation_report.json", report)
        if found != set(TRAIN_FILES):
            raise ValueError(f"Published archive is missing training subsets: {set(TRAIN_FILES) - found}")
        for name, classes in TRAIN_FILES.items():
            LOGGER.info("Inventorying %s without training or test access", name)
            inventory = inspect_training_subset(output / name, classes)
            report["training_subsets"][name].update(inventory)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(output / "preparation_report.json", report)
    return report
