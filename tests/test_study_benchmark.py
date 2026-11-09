"""Paired timing uses common held-out rows and balanced model order."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.experiments.storage import file_digest, object_digest, write_report
from ddos_iot.experiments.study import run_training_study
from ddos_iot.experiments.study_benchmark import benchmark_study
from ddos_iot.experiments.study_config import TrainingTrial
from ddos_iot.multiclass import load_multiclass_bundle
import test_training_study as study_fixture


class StudyBenchmarkTests(unittest.TestCase):
    def test_common_validation_cohort_and_rotating_order_without_model_changes(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        original = run_training_study(config)
        parent = config.output_path / "seed_42/finalists/paper_fixed/model.joblib"
        digest = file_digest(parent)
        biased = load_multiclass_bundle(parent)
        biased.decision_bias = np.zeros(8)
        folder = config.output_path / "bias_v1"
        save_joblib_atomic(folder / "xgb_oof_bias/model.joblib", biased)
        frozen = {"single_model_conditions": {"xgb_oof_bias": biased.decision_bias.tolist()}}
        write_report(folder / "selection.json", frozen)
        write_report(folder / "report.json", {"status": "complete", "selection_sha256": object_digest(frozen),
            "validation_results": [{**original["validation_results"][0], "candidate": "xgb_oof_bias"}]})
        result = benchmark_study(config.output_path, fixture.root / "timing", rows=32, repeats=4)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["test_file_read"])
        self.assertEqual(result["cohort_rows"], 32)
        self.assertEqual(result["orders"], [[0, 1], [1, 0], [0, 1], [1, 0]])
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(len(row["seconds_runs"]) == 4 for row in result["results"]))
        self.assertEqual(file_digest(parent), digest)
        with np.load(config.output_path / "seed_42/splits.npz") as splits, np.load(fixture.root / "timing/cohort.npz") as saved:
            self.assertTrue(set(saved["row_ids"]) <= set(splits["validation"]))
        with self.assertRaises(FileExistsError):
            benchmark_study(config.output_path, fixture.root / "timing", rows=32, repeats=4)


if __name__ == "__main__":
    unittest.main()
