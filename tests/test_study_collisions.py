"""Exact represented-vector ambiguity is distinct from a generalization bound."""

import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.storage import file_digest, write_report
from ddos_iot.experiments.study_collisions import audit_training_collisions
import test_training_study as study_fixture


class StudyCollisionTests(unittest.TestCase):
    def test_conflicting_identical_vectors_are_verified_and_counted(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        frame, _ = fixture.fixture()
        config = fixture.config()
        original = json.loads((fixture.root / "preparation_report.json").read_text())
        features = original["training_subsets"][config.source_path.name]["features"]
        # Row 0 is Benign and row 30 BruteForce in the fixture.
        frame.loc[30, features] = frame.loc[0, features].to_numpy()
        frame.to_csv(config.source_path, index=False)
        original["training_subsets"][config.source_path.name]["sha256"] = file_digest(config.source_path)
        write_report(fixture.root / "preparation_report.json", original)
        config.output_path.mkdir()
        write_report(config.output_path / "config.json", config.as_dict())
        report = audit_training_collisions(config.output_path, fixture.root / "collisions")
        self.assertEqual(report["exact_conflicting_groups"], 1)
        self.assertEqual(report["rows_in_exact_conflicting_groups"], 2)
        self.assertEqual(report["minimum_errors_on_these_training_groups"], 1)
        self.assertAlmostEqual(report["empirical_training_accuracy_upper_bound"], 1 - 1 / len(frame))
        self.assertFalse(report["test_file_read"])
        classes = pd.read_csv(fixture.root / "collisions/per_class.csv", index_col=0)
        self.assertEqual(classes.loc["BruteForce", "rows_in_groups_also_containing_benign"], 1)


if __name__ == "__main__":
    unittest.main()
