"""Empirical label conflicts in identical represented training flow vectors."""

import numpy as np
import pandas as pd

from .storage import write_report, write_table
from .study_config import load_study_config
from .study_data import load_study_data


def audit_training_collisions(source, output):
    config = load_study_config(source / "config.json")
    data = load_study_data(config)
    output.mkdir(parents=True, exist_ok=False)
    hashes = pd.util.hash_pandas_object(data.x, index=False).to_numpy()
    pairs = pd.DataFrame({"hash": hashes, "label": data.y}).groupby(["hash", "label"], sort=False).size().rename("count").reset_index()
    grouped = pairs.groupby("hash").agg(labels=("label", "size"), rows=("count", "sum"), majority=("count", "max"))
    ambiguous = grouped.loc[grouped.labels > 1]
    # Verify feature equality: an accidental 64-bit hash collision cannot be
    # used as evidence of contradictory identical vectors.
    order = np.argsort(hashes)
    sorted_hash = hashes[order]
    x = data.x.to_numpy()
    verified, rejected = [], []
    for value in ambiguous.index:
        begin, end = np.searchsorted(sorted_hash, value, side="left"), np.searchsorted(sorted_hash, value, side="right")
        ids = order[begin:end]
        if np.all(x[ids] == x[ids[0]]): verified.append(value)
        else: rejected.append(int(value))
    conflicting = pairs.loc[pairs.hash.isin(verified)].copy()
    conflicting["class"] = data.encoder.inverse_transform(conflicting.label)
    benign = "Benign" if config.num_classes == 8 else "BenignTraffic"
    benign_hashes = set(conflicting.loc[conflicting["class"] == benign, "hash"])
    records = []
    support = np.bincount(data.y, minlength=config.num_classes)
    for cid, label in enumerate(data.encoder.classes_):
        rows = conflicting.loc[conflicting["class"] == label]
        records.append({"class": label, "training_support": int(support[cid]),
                        "rows_in_exact_cross_label_groups": int(rows["count"].sum()),
                        "rows_in_groups_also_containing_benign": int(rows.loc[rows.hash.isin(benign_hashes), "count"].sum())})
    selected = ambiguous.loc[verified]
    unavoidable = int((selected.rows - selected.majority).sum())
    report = {"status": "complete", "source_sha256": data.metadata["source_sha256"],
              "training_rows": len(data.y), "features": data.metadata["features"],
              "exact_conflicting_groups": len(verified), "rejected_hash_collision_groups": rejected,
              "rows_in_exact_conflicting_groups": int(selected.rows.sum()),
              "minimum_errors_on_these_training_groups": unavoidable,
              "empirical_training_accuracy_upper_bound": 1. - unavoidable / len(data.y),
              "scope": "Exact equality of the float32 selected/scaled representation used by these models; training-only diagnostic.",
              "interpretation": "A deterministic classifier cannot assign different labels to identical represented vectors. This is not a Macro-F1 bound or a population/test-set ceiling; label noise, quantization or missing information can all contribute.",
              "test_file_read": False}
    write_report(output / "report.json", report)
    write_table(output / "per_class.csv", pd.DataFrame(records))
    write_table(output / "conflicting_groups.csv", conflicting[["hash", "class", "count"]])
    return report
