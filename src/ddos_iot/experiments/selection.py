"""Compare candidates at a declared per-seed Benign false-positive budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .evaluation import summarize_seeds


@dataclass
class SelectionAnalysis:
    comparisons: pd.DataFrame
    summary: pd.DataFrame
    selected: dict[str, Any] | None
    report: dict[str, Any]


def analyze_selection(rows: list[dict], config: ExperimentConfig) -> SelectionAnalysis:
    """Use validation metrics only; never retune probabilities or read test data.

    A complete result must contain each declared seed exactly once. Guarded
    choices must meet the reference's argmax FPR plus the declared absolute
    allowance in *every* seed, not just on average.
    """
    frame = pd.DataFrame(rows)
    required = {"candidate", "decision", "seed", "macro_f1", "benign_false_positive_rate"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Selection requires nonempty validation results with class metrics.")
    if frame.duplicated(["candidate", "decision", "seed"]).any():
        raise ValueError("Duplicate candidate/decision/seed results cannot enter selection.")
    if not set(frame.seed).issubset(config.seeds) or not set(frame.candidate).issubset(c.name for c in config.candidates):
        raise ValueError("Selection results do not match the declared candidates/seeds.")
    if not set(frame.decision).issubset({"argmax", "bias"}):
        raise ValueError("Unknown decision rule in validation results.")
    metrics = frame[["macro_f1", "benign_false_positive_rate"]].to_numpy(dtype=float)
    if not np.isfinite(metrics).all() or np.any(metrics < 0) or np.any(metrics > 1):
        raise ValueError("Validation Macro-F1 and Benign FPR must be finite fractions in [0, 1].")

    reference = frame.loc[(frame.candidate == config.reference_name) & (frame.decision == "argmax"),
                          ["seed", "benign_false_positive_rate"]].rename(
                              columns={"benign_false_positive_rate": "reference_benign_fpr"})
    comparisons = frame.merge(reference, on="seed", how="left", validate="many_to_one")
    comparisons["benign_fpr_limit"] = (comparisons.reference_benign_fpr + config.max_benign_fpr_increase).clip(upper=1.)
    comparisons["benign_fpr_delta"] = comparisons.benign_false_positive_rate - comparisons.reference_benign_fpr
    comparisons["fpr_guardrail_pass"] = comparisons.reference_benign_fpr.notna() & (
        comparisons.benign_false_positive_rate <= comparisons.benign_fpr_limit + 1e-12)
    summary = summarize_seeds(rows)
    complete, guard_pass, max_delta = [], [], []
    for row in summary.itertuples():
        group = comparisons.loc[(comparisons.candidate == row.candidate) & (comparisons.decision == row.decision)]
        complete.append(set(group.seed) == set(config.seeds))
        guard_pass.append(bool(group.fpr_guardrail_pass.all()) and complete[-1])
        max_delta.append(float(group.benign_fpr_delta.max()) if group.reference_benign_fpr.notna().all() else None)
    summary["all_seeds_complete"] = complete
    summary["fpr_guardrail_pass"] = guard_pass
    summary["max_benign_fpr_delta"] = max_delta
    summary["eligible_for_selection"] = summary.all_seeds_complete & (
        summary.fpr_guardrail_pass if config.selection_policy == "reference_fpr" else True)
    order = {c.name: i for i, c in enumerate(config.candidates)}
    ranked = summary.assign(candidate_order=summary.candidate.map(order),
                            decision_order=(summary.decision != "argmax").astype(int)).sort_values(
                                ["macro_f1_mean", "decision_order", "candidate_order"], ascending=[False, True, True])

    def winner(mask: pd.Series) -> dict[str, Any] | None:
        eligible = ranked.loc[mask.reindex(ranked.index)]
        if eligible.empty:
            return None
        row = eligible.iloc[0]
        return {"candidate": str(row.candidate), "decision": str(row.decision),
                "macro_f1_mean": float(row.macro_f1_mean), "macro_f1_std": float(row.macro_f1_std),
                "benign_fpr_mean": float(row.benign_false_positive_rate_mean),
                "fpr_guardrail_pass": bool(row.fpr_guardrail_pass)}

    unconstrained = winner(summary.all_seeds_complete)
    guarded = winner(summary.all_seeds_complete & summary.fpr_guardrail_pass)
    selected = guarded if config.selection_policy == "reference_fpr" else unconstrained
    report = {
        "selection_policy": config.selection_policy, "selection_metric": config.selection_metric,
        "reference_candidate": config.reference_name, "reference_decision": "argmax",
        "max_benign_fpr_increase": config.max_benign_fpr_increase,
        "reference_benign_fpr_by_seed": {str(int(r.seed)): float(r.reference_benign_fpr) for r in reference.itertuples()},
        "reference_complete": set(reference.seed) == set(config.seeds),
        "guardrail_scope": "Independent validation; each declared seed must pass. No guarantee on unseen traffic.",
        "unconstrained_winner": unconstrained, "guarded_winner": guarded, "selected": selected,
        "test_used_for_selection": False,
    }
    return SelectionAnalysis(comparisons, summary, selected, report)
