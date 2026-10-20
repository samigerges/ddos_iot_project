"""Evaluation axes, normalization, and published report integrity."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.dashboard_metrics import (
    confusion_figure, load_classification_table, load_confusion_table, normalize_confusion_table,
)


class EvaluationReportTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="evaluation_reports_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_binary_counts_preserve_actual_rows_and_predicted_columns(self):
        frame = load_confusion_table(self.folder, {"confusion_matrix": {"tn": 90, "fp": 10, "fn": 5, "tp": 45}})
        self.assertEqual(frame.index.tolist(), ["Benign", "DDoS"])
        self.assertEqual(frame.columns.tolist(), ["Benign", "DDoS"])
        self.assertEqual(frame.to_numpy().tolist(), [[90, 10], [5, 45]])
        np.testing.assert_allclose(normalize_confusion_table(frame), [[0.9, 0.1], [0.1, 0.9]])
        figure = confusion_figure(frame, normalized=False)
        self.assertEqual(figure.axes[0].get_xlabel(), "Predicted class")
        self.assertEqual(figure.axes[0].get_ylabel(), "Actual class")
        np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), frame)

    def test_csv_class_order_is_aligned_and_zero_support_stays_zero(self):
        pd.DataFrame([[0, 0], [2, 8]], index=["B", "A"], columns=["A", "B"]).to_csv(self.folder / "confusion_matrix.csv")
        frame = load_confusion_table(self.folder)
        self.assertEqual(frame.index.tolist(), ["A", "B"])
        self.assertEqual(frame.to_numpy().tolist(), [[2, 8], [0, 0]])
        normalized = normalize_confusion_table(frame)
        np.testing.assert_allclose(normalized, [[0.2, 0.8], [0, 0]])
        self.assertTrue(np.isfinite(normalized.to_numpy()).all())
        figure = confusion_figure(frame, normalized=True)
        np.testing.assert_allclose(figure.axes[0].images[0].get_array(), normalized)

    def test_wrong_labels_negative_and_fractional_counts_are_rejected(self):
        for frame in (pd.DataFrame([[1]], index=["actual"], columns=["different"]),
                      pd.DataFrame([[-1]], index=["A"], columns=["A"]),
                      pd.DataFrame([[0.5]], index=["A"], columns=["A"])):
            frame.to_csv(self.folder / "confusion_matrix.csv")
            with self.assertRaises(ValueError):
                load_confusion_table(self.folder)

    def test_test_matrix_is_never_used_for_missing_train_counts(self):
        pd.DataFrame([[5]], index=["A"], columns=["A"]).to_csv(self.folder / "confusion_matrix.csv")
        self.assertTrue(load_confusion_table(self.folder, split="train").empty)

    def test_classification_accuracy_is_not_misreported_as_class_support(self):
        metrics = {"classification_report": {"Benign": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 100},
                                              "accuracy": 0.8}}
        frame = load_classification_table(self.folder, metrics=metrics)
        self.assertEqual(frame["class"].tolist(), ["Benign"])
        self.assertEqual(frame["support"].tolist(), [100])

    def test_all_saved_models_have_consistent_evaluation_reports(self):
        models = list((ROOT / "models").rglob("model.joblib"))
        self.assertGreater(len(models), 0)
        for model in models:
            with self.subTest(model=str(model.relative_to(ROOT))):
                metrics = json.loads(model.with_name("metrics.json").read_text(encoding="utf-8"))
                binary = "test" in metrics
                saved = metrics["test"] if binary else metrics
                matrix = load_confusion_table(model.parent, saved if binary else None)
                self.assertFalse(matrix.empty)
                expected = saved["inference_rows"] if binary else saved["test_rows"]
                self.assertEqual(int(matrix.to_numpy().sum()), expected)
                if not binary:
                    mapping = json.loads((model.parent.parent / "label_mapping.json").read_text(encoding="utf-8"))
                    self.assertEqual(matrix.index.tolist(), [mapping[str(i)] for i in range(len(mapping))])
                table = load_classification_table(model.parent, metrics=saved if binary else None)
                for label in matrix.index:
                    support = table.loc[table["class"] == label, "support"].iloc[0]
                    self.assertEqual(support, matrix.loc[label].sum())


if __name__ == "__main__":
    unittest.main()
