"""Paired, label-stratified bootstrap analysis of frozen test predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from .storage import file_digest, write_report, write_table


def _f1(matrix):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)


def paired_bootstrap(y, reference, candidate, classes, *, repetitions=2000, seed=2024):
    y, reference, candidate = (np.asarray(v) for v in (y, reference, candidate))
    k = len(classes)
    if type(repetitions) is not int or repetitions < 100 or k < 2 or not len(y):
        raise ValueError("Require covered classes, predictions and at least 100 bootstrap draws.")
    for values in (y, reference, candidate):
        if values.shape != (len(y),) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= k):
            raise ValueError("Predictions must be aligned integer class IDs.")
    y, reference, candidate = (v.astype(np.int64, copy=False) for v in (y, reference, candidate))
    support = np.bincount(y, minlength=k)
    if not support.min(): raise ValueError("Every declared class must be represented.")
    joint = np.bincount((y * k + reference) * k + candidate, minlength=k ** 3).reshape(k, k, k)
    a = np.zeros((repetitions, k, k), dtype=np.int64)
    b = np.zeros_like(a)
    rng = np.random.default_rng(seed)
    for true_id in range(k):
        counts = joint[true_id].ravel()
        ids = np.flatnonzero(counts)
        draws = rng.multinomial(int(support[true_id]), counts[ids] / support[true_id], size=repetitions)
        for column, pair in enumerate(ids):
            # Add separately: different joint pairs may share a marginal index.
            a[:, true_id, pair // k] += draws[:, column]
            b[:, true_id, pair % k] += draws[:, column]
    original_a, original_b = joint.sum(axis=2), joint.sum(axis=1)
    f1_a, f1_b = _f1(original_a), _f1(original_b)
    sampled_delta = _f1(b) - _f1(a)
    low, high = np.quantile(sampled_delta, [.025, .975], axis=0)
    macro_low, macro_high = np.quantile(sampled_delta.mean(axis=1), [.025, .975])
    return {"rows": len(y), "repetitions": repetitions, "bootstrap_seed": seed,
        "method": "Percentile 95% paired bootstrap, stratified by true label; joint predictions resampled together within each fixed-size class.",
        "scope": "Conditional flow-level uncertainty on this benchmark. Correlated flows/captures can make these intervals too narrow; this is not new-device/capture uncertainty.",
        "macro_f1": {"reference": float(f1_a.mean()), "candidate": float(f1_b.mean()),
            "delta": float((f1_b - f1_a).mean()), "delta_ci95": [float(macro_low), float(macro_high)]},
        "per_class": [{"class": str(label), "support": int(support[i]), "reference_f1": float(f1_a[i]),
            "candidate_f1": float(f1_b[i]), "delta": float(f1_b[i] - f1_a[i]),
            "delta_ci95_low": float(low[i]), "delta_ci95_high": float(high[i])} for i, label in enumerate(classes)],
        "test_used_for_selection": False,
        "interpretation": "Descriptive class intervals have no multiple-comparison correction. Do not tune or select a new model from this analysis."}


def report_test_uncertainty(reference: Path, candidate: Path, output: Path):
    import json
    for path in (reference, candidate):
        metadata = json.loads(path.with_name("metrics.json").read_text(encoding="utf-8"))
        if metadata.get("split") != "test":
            raise ValueError("This analysis accepts frozen final-test predictions only.")
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        np.testing.assert_array_equal(a["y"], b["y"])
        np.testing.assert_array_equal(a["classes"], b["classes"])
        result = paired_bootstrap(a["y"], a["predicted"], b["predicted"], a["classes"])
    output.mkdir(parents=True, exist_ok=False)
    result.update(status="complete", reference_predictions_sha256=file_digest(reference),
                  candidate_predictions_sha256=file_digest(candidate))
    write_report(output / "report.json", result)
    write_table(output / "per_class_intervals.csv", pd.DataFrame(result["per_class"]))
    metric = result["macro_f1"]
    (output / "REPORT.md").write_text(
        "# Frozen-test paired uncertainty\n\n" + result["method"] + "\n\n"
        f"Macro-F1 delta: **{metric['delta']:+.5f}**, 95% interval **[{metric['delta_ci95'][0]:+.5f}, {metric['delta_ci95'][1]:+.5f}]**.\n\n"
        + result["scope"] + "\n\n" + result["interpretation"] + "\n", encoding="utf-8")
    return result
