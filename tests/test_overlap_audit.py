"""Class-level audits preserve raw row IDs and group features without labels."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.attack_families import FAMILY_NAMES, RAW_TO_FAMILY
from ddos_iot.experiments.config import ExperimentConfig
from ddos_iot.experiments.overlap_audit import audit_overlap
from ddos_iot.experiments.storage import file_digest, write_arrays, write_report
from test_multiclass_8 import family_fixture


class OverlapAuditTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="overlap_audit_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        data = family_fixture(10)
        features = data.columns.drop("label")
        data.loc[321, features] = data.loc[100, features].to_numpy()
        self.assertNotEqual(RAW_TO_FAMILY[data.loc[321, "label"]], RAW_TO_FAMILY[data.loc[100, "label"]])
        data.loc[0, "f0"] = np.inf
        train = self.root / "train.csv"
        data.to_csv(train, index=False)
        config = ExperimentConfig(train_input_path=train, test_input_path=self.root / "missing_test.csv",
                                  output_path=self.run, chunk_size=37)
        write_report(self.run / "config.json", config.as_dict())
        write_report(self.run / "data_report.json", {"raw_rows": len(data), "source_sha256": file_digest(train)})
        development = np.array([i for i in range(len(data)) if i not in (0, 321)])
        validation = np.array([321])
        classes = np.array(FAMILY_NAMES)
        y = np.array([FAMILY_NAMES.index(RAW_TO_FAMILY[label]) for label in data.label])
        write_arrays(self.run / "seed_42/splits.npz", development_row_ids=development,
                     validation_row_ids=validation, development_y=y[development], validation_y=y[validation], classes=classes)
        write_report(self.run / "seed_42/overlap.json", {"overlapping_right_rows": 1})

    def test_cross_label_duplicates_are_counted_without_touching_test_data(self):
        report = audit_overlap(self.run, 42)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["overlapping_validation_rows"], 1)
        self.assertFalse(report["test_file_read"])
        folder = self.run / "overlap_audit/seed_42"
        classes = pd.read_csv(folder / "overlap_by_class.csv")
        self.assertEqual(classes.overlapping_validation_rows.sum(), 1)
        self.assertEqual(classes.development_rows.sum(), 338)
        with np.load(folder / "validation_overlap.npz") as saved:
            np.testing.assert_array_equal(saved["row_ids"], [321])
            np.testing.assert_array_equal(saved["overlap"], [True])

    def test_mismatched_overlap_is_reported_and_array_handles_close_on_failure(self):
        write_report(self.run / "seed_42/overlap.json", {"overlapping_right_rows": 0})
        with self.assertRaisesRegex(ValueError, "does not match"):
            audit_overlap(self.run, 42)
        report = json.loads((self.run / "overlap_audit/seed_42/report.json").read_text())
        self.assertEqual(report["status"], "failed")

    def test_source_change_is_rejected_before_scanning(self):
        with (self.root / "train.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            audit_overlap(self.run, 42)


if __name__ == "__main__":
    unittest.main()
