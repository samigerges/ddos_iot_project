"""Stage comparisons keep tuning, partial runs and final-test phases separate."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.research_reporting import report_research, target_error_summary
from ddos_iot.experiments.storage import file_digest, write_report


class ResearchReportingTests(unittest.TestCase):
    def test_sleep_interruption_excludes_only_timing_and_preserves_original_report(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary) / "study"
            write_report(root / "report.json", {"status": "complete", "num_classes": 8, "validation_results": []})
            rows = [{"candidate": "paper_fixed", "seed": seed, "macro_f1": score,
                "weighted_f1": .99, "benign_false_positive_rate": .03, "inference_us_per_sample": timing}
                for seed, score, timing in ((42, .8, 10.), (123, .81, 12.), (2024, .82, 3000.))]
            source = root / "final/report.json"
            write_report(source, {"status": "complete", "test_csv_opened": True, "results": rows})
            before = file_digest(source)
            write_report(source.with_name("timing_audit.json"), {"status": "complete", "invalid_timed_passes": [
                {"seed": 2024, "candidate": "paper_fixed", "original_inference_us_per_sample": 3000., "reason": "Recorded sleep"}]})
            output = Path(temporary) / "report"
            report_research([root], output, plots=False)
            self.assertEqual(file_digest(source), before)
            results = pd.read_csv(output / "comparison.csv")
            self.assertEqual(len(results), 3)
            self.assertTrue(pd.isna(results.loc[results.seed == 2024, "inference_us_per_sample"]).all())
            self.assertEqual(results.loc[results.seed == 2024, "recorded_inference_us_per_sample"].iloc[0], 3000.)
            summary = pd.read_csv(output / "summary.csv").iloc[0]
            self.assertEqual(summary.seeds, 3)
            self.assertEqual(summary.valid_timing_passes, 2)
            self.assertAlmostEqual(summary.macro_f1_mean, .81)
            self.assertEqual(summary.inference_us_per_sample_mean, 11.)

    def test_target_errors_distinguish_benign_within_family_and_other_families(self):
        frame = pd.DataFrame({"true_label": ["SqlInjection", "SqlInjection", "SqlInjection", "BenignTraffic"],
            "predicted_label": ["BenignTraffic", "XSS", "DNS_Spoofing", "SqlInjection"], "count": [2, 3, 5, 100]})
        summary, selected = target_error_summary(frame, 34)
        self.assertEqual(summary["target_errors"], 10)
        self.assertEqual(summary["to_benign_fraction"], .2)
        self.assertEqual(summary["within_family_fraction"], .3)
        self.assertEqual(summary["other_family_fraction"], .5)
        self.assertEqual(len(selected), 3)

    def test_only_completed_evaluation_rows_and_correct_component_timing(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary) / "study"
            def row(seed, score):
                return {"candidate": "paper_fixed", "seed": seed, "macro_f1": score, "weighted_f1": .99,
                    "benign_false_positive_rate": .03, "reference_constraints_pass": True,
                    "shared_probability_inference_us": 10., "decision_us_per_sample": .2}
            write_report(root / "report.json", {"status": "complete", "num_classes": 8,
                "results": [row(42, .01)], "validation_results": [row(42, .8)]})
            write_report(root / "confirmation/report.json", {"status": "complete", "results": [row(42, .8), row(123, .9)]})
            tuned = {**row(42, .82), "candidate": "tuned"}
            write_report(root / "bias/report.json", {"status": "complete", "validation_results": [tuned]})
            write_report(root / "interrupted/report.json", {"status": "loading", "results": [row(42, 1.)]})
            write_report(root / "final/report.json", {"status": "complete", "test_csv_opened": True, "results": [row(42, .85)]})
            parent = root / "seed_42/finalists/paper_fixed/validation"
            parent.mkdir(parents=True)
            pd.DataFrame({"class": ["WebBased", "DDoS", "macro avg"], "precision": [.8, 1., .9],
                "recall": [.6, 1., .8], "f1-score": [.6857, 1., .8429], "support": [100, 1000, 1100]}).set_index("class").to_csv(parent / "classification_report.csv")
            before = file_digest(root / "report.json")
            output = Path(temporary) / "report"
            report_research([root], output, plots=False)
            self.assertEqual(file_digest(root / "report.json"), before)
            rows = pd.read_csv(output / "comparison.csv")
            self.assertEqual(len(rows), 5)
            self.assertNotIn(.01, rows.macro_f1.tolist())
            self.assertNotIn("interrupted", rows.stage.tolist())
            self.assertEqual(len(rows.loc[rows.phase == "test"]), 1)
            self.assertTrue((rows.inference_us_per_sample == 10.2).all())
            self.assertAlmostEqual(rows.loc[rows.candidate == "tuned", "macro_f1_delta_vs_reference"].iloc[0], .02)
            summary = pd.read_csv(output / "summary.csv")
            confirmed = summary.loc[summary.stage == "confirmation"].iloc[0]
            self.assertAlmostEqual(confirmed.macro_f1_mean, .85)
            self.assertAlmostEqual(confirmed.macro_f1_std, .05)
            minority = pd.read_csv(output / "minority_class_summary.csv")
            self.assertEqual(minority["class"].tolist(), ["WebBased"])
            self.assertEqual(minority.recall_mean.iloc[0], .6)
            self.assertEqual(minority.f1_delta_mean.iloc[0], 0.)
            self.assertEqual(minority.recall_delta_mean.iloc[0], 0.)
            self.assertTrue((pd.read_csv(output / "stages.csv").status == "loading").any())


if __name__ == "__main__":
    unittest.main()
