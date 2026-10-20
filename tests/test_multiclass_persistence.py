"""Models survive evaluation failures and can be reused without any fitting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiclass import parse_args
from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import load_multiclass_bundle, predict_ids, run_multiclass_training
from ddos_iot.multiclass_inference import run_multiclass_prediction
from test_multiclass import fixture
from test_multiclass_8 import family_fixture


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="persistence_test_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, classes=34):
        return parse_args(["--num-classes", str(classes), "--data-protocol", "existing",
                           "--train-input-path", str(self.root / "train.csv"),
                           "--test-input-path", str(self.root / "test.csv"),
                           "--output-path", str(self.root / "results"),
                           "--models", "xgboost", "--n-estimators", "3", "--n-jobs", "2",
                           "--feature-selection", "none", "--cv-folds", "0", "--log-period", "0", "--no-plots"])

    def test_failed_atomic_write_keeps_previous_file_and_removes_temporary(self):
        path = self.root / "model.joblib"
        save_joblib_atomic(path, {"valid": True})
        original = path.read_bytes()

        def broken_dump(value, stream):
            stream.write(b"incomplete model")
            raise OSError("simulated disk failure")

        with patch("ddos_iot.artifacts.joblib.dump", side_effect=broken_dump):
            with self.assertRaisesRegex(OSError, "disk failure"):
                save_joblib_atomic(path, {"valid": False})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(joblib.load(path), {"valid": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_model_saved_before_test_prediction_and_survives_failure(self):
        fixture().to_csv(self.root / "train.csv", index=False)
        fixture(3).to_csv(self.root / "test.csv", index=False)
        path = self.root / "results" / "xgboost" / "model.joblib"

        def fail_at_test_warmup(name, model, batch, **kwargs):
            if len(batch) == 100:
                self.assertTrue(path.exists(), "Must save before test warm-up/prediction")
                self.assertEqual(len(load_multiclass_bundle(path).encoder.classes_), 34)
                raise RuntimeError("simulated test prediction failure")
            return predict_ids(name, model, batch, **kwargs)

        with patch("ddos_iot.multiclass.predict_ids", side_effect=fail_at_test_warmup), \
             self.assertLogs("ddos_iot.multiclass", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "Models failed"):
                run_multiclass_training(self.args())
        bundle = load_multiclass_bundle(path)
        self.assertEqual(len(bundle.predict(fixture(3).drop(columns="label"))), 102)
        report = json.loads((path.parent / "model_save_report.json").read_text())
        self.assertEqual(report["evaluation_status"], "failed")
        comparison = json.loads((path.parent.parent / "model_comparison_report.json").read_text())
        self.assertIn("xgboost", comparison["saved_models"])

    def test_load_and_stream_predict_both_class_counts_without_retraining(self):
        for classes, make_data in ((34, fixture), (8, family_fixture)):
            with self.subTest(classes=classes):
                train, test = make_data(12), make_data(3)
                train.to_csv(self.root / "train.csv", index=False)
                test.to_csv(self.root / "test.csv", index=False)
                args = self.args(classes)
                args.output_path = self.root / f"results_{classes}"
                run_multiclass_training(args)
                path = args.output_path / "xgboost" / "model.joblib"
                original = path.read_bytes()
                expected = load_multiclass_bundle(path).predict(test.drop(columns="label"))
                output = self.root / f"predictions_{classes}.csv"
                with patch("ddos_iot.multiclass.fit_model", side_effect=AssertionError("Must not train")), \
                     patch("ddos_iot.multiclass.grid_search", side_effect=AssertionError("Must not tune")), \
                     patch("ddos_iot.multiclass.fit_features", side_effect=AssertionError("Must not refit scaler")):
                    result = run_multiclass_prediction(path, self.root / "test.csv", output, chunk_size=17)
                self.assertEqual(result["class_count"], classes)
                self.assertEqual(result["rows"], len(test))
                np.testing.assert_array_equal(pd.read_csv(output).predicted_label, expected)
                self.assertEqual(path.read_bytes(), original)
                with self.assertRaises(FileExistsError):
                    run_multiclass_training(args)
                with self.assertRaises(FileExistsError):
                    run_multiclass_prediction(path, self.root / "test.csv", output)
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    run_multiclass_prediction(path, self.root / "test.csv", self.root / "test.csv")
                wrong_task_output = self.root / f"wrong_task_{classes}.csv"
                with self.assertRaisesRegex(ValueError, "selected"):
                    run_multiclass_prediction(path, self.root / "test.csv", wrong_task_output,
                                             expected_num_classes=8 if classes == 34 else 34)
                self.assertFalse(wrong_task_output.exists())
                save_report = json.loads((path.parent / "model_save_report.json").read_text())
                self.assertEqual(save_report["evaluation_status"], "complete")

    def test_binary_bundle_rejected_by_multiclass_loader(self):
        path = self.root / "binary.joblib"
        save_joblib_atomic(path, {"model": None, "feature_columns": []})
        with self.assertRaisesRegex(ValueError, "binary model files"):
            load_multiclass_bundle(path)


if __name__ == "__main__":
    unittest.main()
