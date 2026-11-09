"""The Benign FPR budget must hold per seed, with an auditable fallback."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.config import Candidate, ExperimentConfig
from ddos_iot.experiments.selection import analyze_selection


def row(candidate, seed, score, fpr, decision="argmax"):
    return {"candidate": candidate, "decision": decision, "seed": seed,
            "macro_f1": score, "benign_false_positive_rate": fpr,
            "weighted_f1": .95, "accuracy": .95, "inference_us_per_sample": 1.}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(
            candidates=(Candidate("reference"), Candidate("safer"), Candidate("aggressive")),
            seeds=(42, 123))
        self.rows = [row("reference", 42, .80, .10), row("reference", 123, .80, .20),
                     row("safer", 42, .82, .09), row("safer", 123, .83, .19),
                     row("aggressive", 42, .95, .08), row("aggressive", 123, .95, .22)]

    def test_one_bad_seed_cannot_be_hidden_by_an_acceptable_mean(self):
        result = analyze_selection(self.rows, self.config)
        self.assertEqual(result.selected["candidate"], "safer")
        self.assertEqual(result.report["unconstrained_winner"]["candidate"], "aggressive")
        self.assertFalse(result.report["unconstrained_winner"]["fpr_guardrail_pass"])
        bad = result.comparisons.loc[(result.comparisons.candidate == "aggressive") & (result.comparisons.seed == 123)].iloc[0]
        self.assertFalse(bad.fpr_guardrail_pass)
        self.assertAlmostEqual(bad.benign_fpr_limit, .20)
        self.assertEqual(result.report["reference_benign_fpr_by_seed"], {"42": .10, "123": .20})

    def test_reference_is_fallback_and_exact_boundary_is_allowed(self):
        rows = [r for r in self.rows if r["candidate"] != "safer"]
        result = analyze_selection(rows, self.config)
        self.assertEqual(result.selected["candidate"], "reference")
        self.assertTrue(result.selected["fpr_guardrail_pass"])
        result = analyze_selection(self.rows, replace(self.config, max_benign_fpr_increase=.02))
        self.assertEqual(result.selected["candidate"], "aggressive")

    def test_missing_reference_seed_does_not_silently_disable_the_guard(self):
        rows = [r for r in self.rows if not (r["candidate"] == "reference" and r["seed"] == 123)]
        result = analyze_selection(rows, self.config)
        self.assertIsNone(result.selected)
        self.assertIsNone(result.report["guarded_winner"])
        self.assertIsNotNone(result.report["unconstrained_winner"])
        self.assertFalse(result.report["reference_complete"])

    def test_incomplete_candidate_cannot_win_and_duplicates_are_rejected(self):
        rows = [r for r in self.rows if not (r["candidate"] == "aggressive" and r["seed"] == 123)]
        result = analyze_selection(rows, self.config)
        self.assertEqual(result.report["unconstrained_winner"]["candidate"], "safer")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze_selection(self.rows + [self.rows[0]], self.config)

    def test_legacy_macro_policy_keeps_the_guarded_alternative_visible(self):
        result = analyze_selection(self.rows, replace(self.config, selection_policy="macro_f1"))
        self.assertEqual(result.selected["candidate"], "aggressive")
        self.assertEqual(result.report["guarded_winner"]["candidate"], "safer")

    def test_ties_prefer_argmax_then_declared_candidate_order(self):
        rows = self.rows[:4] + [row("safer", 42, .82, .09, "bias"), row("safer", 123, .83, .19, "bias")]
        result = analyze_selection(rows, self.config)
        self.assertEqual(result.selected["decision"], "argmax")

    def test_invalid_rate_and_configuration_fail_before_selection(self):
        for invalid in (float("nan"), float("inf"), -0.1, 1.1):
            rows = [dict(r) for r in self.rows]
            rows[0]["benign_false_positive_rate"] = invalid
            with self.subTest(rate=invalid), self.assertRaises(ValueError):
                analyze_selection(rows, self.config)
        for change in ({"reference_candidate": "missing"}, {"max_benign_fpr_increase": -.01},
                       {"max_benign_fpr_increase": float("nan")}, {"selection_policy": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change)


if __name__ == "__main__":
    unittest.main()
