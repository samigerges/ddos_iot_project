"""Binary model reports stay current when artifacts arrive after deployment."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot import dashboard


class BinaryReportTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        previous_temp = tempfile.tempdir
        tempfile.tempdir = str(ROOT / "tmp")
        self.addCleanup(setattr, tempfile, "tempdir", previous_temp)
        self.temp = tempfile.TemporaryDirectory(prefix="binary_report_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = self.root / "model_comparison_report.json"
        dashboard.clear_dashboard_caches()
        self.addCleanup(dashboard.clear_dashboard_caches)

    def write_report(self, model):
        self.report.write_text(json.dumps({"best_model": {"model": model}}), encoding="utf-8")

    def test_report_uploaded_after_first_visit_is_detected(self):
        self.assertIsNone(dashboard.read_json(self.report))
        self.write_report("lightgbm")
        self.assertEqual(dashboard.best_model_name(dashboard.read_json(self.report)), "lightgbm")

    def test_updated_report_replaces_previously_displayed_winner(self):
        self.write_report("xgboost")
        self.assertEqual(dashboard.best_model_name(dashboard.read_json(self.report)), "xgboost")
        self.write_report("lightgbm")
        self.assertEqual(dashboard.best_model_name(dashboard.read_json(self.report)), "lightgbm")

    def test_removed_report_is_not_displayed_as_available(self):
        self.write_report("lightgbm")
        self.assertIsNotNone(dashboard.read_json(self.report))
        self.report.unlink()
        self.assertIsNone(dashboard.read_json(self.report))

    def test_hosted_binary_overview_and_predict_need_no_local_datasets(self):
        from streamlit.testing.v1 import AppTest

        self.write_report("lightgbm")
        for name in ("lightgbm", "xgboost", "random_forest", "hist_gradient_boosting", "naive_bayes"):
            model = self.root / name / "model.joblib"
            model.parent.mkdir()
            model.write_bytes(b"browsing must not deserialize model files")
        (self.root / "model_comparison.csv").write_text("model,f1\nlightgbm,0.99\n", encoding="utf-8")
        missing = self.root / "absent_data"
        with patch.multiple(dashboard, MODEL_COMPARISON_DIR=self.root, RAW_DATA_DIR=missing / "raw",
                            RAW_TRAIN_FILE=missing / "raw/train/train.csv",
                            RAW_TEST_FILE=missing / "raw/test/test.csv", PROCESSED_DATA_DIR=missing / "processed"), \
             patch("joblib.load", side_effect=AssertionError("Browsing must not load models")), \
             patch.object(dashboard, "run_command", side_effect=AssertionError("Browsing must not train")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            metrics = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(metrics["Raw CSV files"], "0")
            self.assertEqual(metrics["Best model"], "lightgbm")
            self.assertTrue(any("Saved binary models are available" in message.value for message in app.info))
            app.radio[0].set_value("Predict").run()
            self.assertEqual(len(app.exception), 0)
            models = next(select for select in app.selectbox if select.label == "Model")
            self.assertEqual(len(models.options), 5)


if __name__ == "__main__":
    unittest.main()
