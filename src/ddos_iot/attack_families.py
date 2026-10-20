"""CICIoT2023's eight families and the paper's sampled 80/20 data protocol.

Names follow Fig. 7 of the supplied paper. Membership follows Table 6 and
https://www.unb.ca/cic/datasets/iotdataset-2023.html (including VulnerabilityScan).
"""

from __future__ import annotations

import logging
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)
FAMILY_MEMBERS = {
    "Benign": ("BenignTraffic",),
    "BruteForce": ("DictionaryBruteForce",),
    "DDoS": ("DDoS-ACK_Fragmentation", "DDoS-HTTP_Flood", "DDoS-ICMP_Flood",
             "DDoS-ICMP_Fragmentation", "DDoS-PSHACK_Flood", "DDoS-RSTFINFlood",
             "DDoS-SYN_Flood", "DDoS-SlowLoris", "DDoS-SynonymousIP_Flood",
             "DDoS-TCP_Flood", "DDoS-UDP_Flood", "DDoS-UDP_Fragmentation"),
    "DoS": ("DoS-HTTP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-UDP_Flood"),
    "Mirai": ("Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain"),
    "Recon": ("Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep", "Recon-PortScan", "VulnerabilityScan"),
    "Spoofing": ("DNS_Spoofing", "MITM-ArpSpoofing"),
    "WebBased": ("Backdoor_Malware", "BrowserHijacking", "CommandInjection", "SqlInjection", "Uploading_Attack", "XSS"),
}
FAMILY_NAMES = tuple(FAMILY_MEMBERS)
RAW_TO_FAMILY = {raw: family for family, members in FAMILY_MEMBERS.items() for raw in members}


def map_family_labels(labels: pd.Series) -> pd.Series:
    """Map explicit raw labels; never silently treat an unknown attack as benign."""
    labels = labels.astype("string").str.strip()
    present = labels.notna() & labels.ne("")
    unknown = sorted(set(labels[present]) - set(RAW_TO_FAMILY))
    if unknown:
        raise ValueError(f"Unknown CICIoT2023 raw labels for eight-class mapping: {unknown}")
    return labels.map(RAW_TO_FAMILY).astype("string")


def proportional_quotas(counts: dict[str, int], target: int, minimum: int) -> dict[str, int]:
    """Paper Eq. 1: max(minimum, floor(count / total * target)), without oversampling.

The paper does not specify rounding/minimum reconciliation to an exact total.
Keep its literal rule and record the actual total instead of inventing one.
"""
    if set(counts) != set(FAMILY_NAMES) or any(n <= 0 for n in counts.values()):
        raise ValueError("The source pool must contain all eight paper families.")
    if target <= 0 or minimum <= 0 or target < len(FAMILY_NAMES) * minimum:
        raise ValueError("Sampling target must be at least 8 * min-samples-per-class, with positive values.")
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"Sampling target {target:,} exceeds {total:,} valid source rows. "
                         "Provide more raw data or explicitly lower --sampling-target.")
    quotas = {family: max(minimum, counts[family] * target // total) for family in FAMILY_NAMES}
    short = {family: {"available": counts[family], "required": quota}
             for family, quota in quotas.items() if quota > counts[family]}
    if short:
        raise ValueError(f"Insufficient real samples for the requested family minimum: {short}. "
                         "No duplicated or synthetic rows will be created.")
    return quotas


def validate_source_paths(paths: list[Path], label_column: str) -> list[str]:
    resolved = [p.resolve() for p in paths]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("Provide distinct raw source files; duplicate source paths are not allowed.")
    features = None
    for path in paths:
        columns = list(pd.read_csv(path, nrows=0).columns)
        if label_column not in columns:
            raise ValueError(f"{path} has no {label_column!r} column.")
        current = [column for column in columns if column != label_column]
        if len(current) != 46 or (features is not None and set(current) != set(features)):
            raise ValueError("Paper sampling requires the same 46 numeric features in every raw source file.")
        if features is None:
            features = current
    return features


def clean_source_chunks(paths: list[Path], label_column: str, chunk_size: int,
                        features: list[str]) -> Iterator[tuple[pd.DataFrame, dict[str, Any]]]:
    dtypes = {column: "float32" for column in features}
    dtypes[label_column] = "string"
    for path in paths:
        with pd.read_csv(path, dtype=dtypes, chunksize=chunk_size) as reader:
            for chunk in reader:
                raw_labels = chunk[label_column].str.strip()
                families = map_family_labels(raw_labels)
                chunk[label_column] = families
                valid = families.notna().to_numpy() & np.isfinite(chunk[features]).all(axis=1).to_numpy()
                yield chunk.loc[valid].copy(), {
                    "raw_rows": len(chunk), "removed_invalid_rows": int((~valid).sum()),
                    "raw_label_counts": raw_labels[raw_labels.notna() & raw_labels.ne("")].value_counts().to_dict(),
                    "raw_family_counts": families.dropna().value_counts().to_dict(),
                }


def scan_source_pool(paths: list[Path], label_column: str, chunk_size: int) -> dict[str, Any]:
    features = validate_source_paths(paths, label_column)
    raw_labels, raw_families, valid_families = Counter(), Counter(), Counter()
    total = invalid = 0
    for index, (chunk, stats) in enumerate(clean_source_chunks(paths, label_column, chunk_size, features), 1):
        total += stats["raw_rows"]
        invalid += stats["removed_invalid_rows"]
        raw_labels.update(stats["raw_label_counts"])
        raw_families.update(stats["raw_family_counts"])
        valid_families.update(chunk[label_column])
        if index % 4 == 0:
            LOGGER.info("Source inventory: %s rows, %s families", f"{total:,}", len(valid_families))
    return {"source_paths": [str(path.resolve()) for path in paths], "raw_rows": total,
            "removed_invalid_rows": invalid, "valid_rows": total - invalid,
            "raw_label_counts": dict(raw_labels), "raw_family_counts": dict(raw_families),
            "valid_family_counts": dict(valid_families), "original_features": features}


def prepare_paper_split(paths: list[Path], label_column: str, chunk_size: int, seed: int,
                        target: int, minimum: int, test_size: float
                        ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Count clean families, fill stream quotas, then form a new stratified split."""
    inventory = scan_source_pool(paths, label_column, chunk_size)
    quotas = proportional_quotas(inventory["valid_family_counts"], target, minimum)
    LOGGER.info("Paper family quotas: %s; actual target=%s", quotas, f"{sum(quotas.values()):,}")
    remaining = dict(quotas)
    rng = np.random.default_rng(seed)
    parts = []
    with closing(clean_source_chunks(paths, label_column, chunk_size, inventory["original_features"])) as chunks:
        for chunk, _ in chunks:
            for family, group in chunk.groupby(label_column, sort=False):
                count = min(remaining[family], len(group))
                if count:
                    chosen = group.iloc[rng.choice(len(group), count, replace=False)] if count < len(group) else group
                    parts.append(chosen.copy())
                    remaining[family] -= count
            if not any(remaining.values()):
                break
    if any(remaining.values()):
        raise ValueError(f"Source data changed between inventory and sampling; unfilled quotas: {remaining}")
    sampled = pd.concat(parts, ignore_index=True)
    del parts
    train_ids, test_ids = train_test_split(np.arange(len(sampled)), test_size=test_size,
                                          stratify=sampled[label_column], random_state=seed)
    train, test = sampled.iloc[train_ids].copy(), sampled.iloc[test_ids].copy()
    # Indices identify distinct sampled rows. Preserve them for split-isolation tests.
    report = {**inventory, "protocol": "paper", "target_rows": target, "minimum_per_family": minimum,
              "quota_formula": "max(minimum, floor(valid_family_count / valid_total * target))",
              "quotas": quotas, "sampled_rows": len(sampled), "sampling": "chunk-wise random quota filling",
              "sampling_with_replacement": False, "duplicates": "retained as present in source",
              "test_size": test_size, "random_state": seed,
              "train_rows": len(train), "test_rows": len(test),
              "train_family_counts": train[label_column].value_counts().to_dict(),
              "test_family_counts": test[label_column].value_counts().to_dict(),
              "note": "New experiment split from the listed source pool, not the pre-existing test file. "
                      "The literal quota formula can produce a total slightly different from the target."}
    return train, test, report


def generated_split_report(data: pd.DataFrame, label_column: str) -> dict[str, Any]:
    counts = data[label_column].value_counts().to_dict()
    return {"origin": "new stratified split after paper family sampling", "raw_rows": len(data),
            "valid_rows": len(data), "selected_rows": len(data), "removed_invalid_rows": 0,
            "raw_counts": counts, "valid_counts": counts, "selected_counts": counts,
            "max_rows_per_class": 0, "duplicates": "retained as present in source"}
