"""All-class SHAP axes, raw/probability units, persistence and no-retraining tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiclass import parse_args
from explain_multiclass import parse_args as parse_shap_args
from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.multiclass import FeatureTransform, MulticlassBundle, build_model, load_multiclass_bundle
from ddos_iot.multiclass_shap import (
    check_additivity, explain_bundle, run_multiclass_shap, sample_raw_features,
)


def tiny_bundle(name: str, count: int = 8) -> tuple[MulticlassBundle, pd.DataFrame]:
    rng = np.random.default_rng(19)
    y = np.repeat(np.arange(count), 12)
    raw = pd.DataFrame({"signal": y.astype(float), "noise": rng.normal(size=len(y)),
                        "squared": y.astype(float) ** 2, "constant": 1.0})
    scaler = StandardScaler().fit(raw.to_numpy(dtype="float32"))
    features = FeatureTransform(list(raw.columns), scaler, {}, method="none")
    encoder = LabelEncoder().fit([f"label_{i:02d}" for i in range(count)])
    args = parse_args(["--num-classes", str(count), "--data-protocol", "existing", "--n-jobs", "1",
                       "--n-estimators", "3", "--max-depth", "3", "--feature-selection", "none"])
    model = build_model(name, args, 3, False)
    model.fit(features.transform(raw), y)
    return MulticlassBundle(name, model, features, encoder), raw


class MulticlassShapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundles = {(name, count): tiny_bundle(name, count)
                       for count in (8, 34) for name in ("xgboost", "lightgbm", "catboost")}

    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="multiclass_shap_test_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save_fixture(self, name="xgboost", count=8):
        bundle, raw = self.bundles[name, count]
        model = self.root / f"{name}_{count}" / "model.joblib"
        save_joblib_atomic(model, bundle)
        csv = self.root / f"raw_{count}.csv"
        raw.assign(label="ignored_label").to_csv(csv, index=False)
        return model, csv

    def test_all_six_native_bundles_align_classes_and_preserve_models(self):
        for name, count in self.bundles:
            with self.subTest(name=name, count=count):
                model, csv = self.save_fixture(name, count)
                original = model.read_bytes()
                output = self.root / f"shap_{name}_{count}"
                with patch("ddos_iot.multiclass.fit_model", side_effect=AssertionError("Must not fit")), \
                     patch("ddos_iot.multiclass.fit_features", side_effect=AssertionError("Must not fit scaler")), \
                     patch("ddos_iot.multiclass.grid_search", side_effect=AssertionError("Must not tune")):
                    report = run_multiclass_shap(model, csv, output, sample_size=4, max_scan_rows=0,
                                                 local_count=1, no_plots=True, n_jobs=1,
                                                 expected_num_classes=count)
                self.assertEqual(model.read_bytes(), original)
                self.assertEqual(report["class_count"], count)
                self.assertEqual(report["output_space"], "raw_margin")
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["background_rows"], 0)
                with np.load(output / "shap_values.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["values"].shape, (4, 4, count))
                    np.testing.assert_allclose(arrays["base_values"] + arrays["values"].sum(axis=1),
                                               arrays["outputs"], atol=1e-5, rtol=1e-4)
                    np.testing.assert_allclose(arrays["probabilities"].sum(axis=1), 1, atol=1e-6)
                    np.testing.assert_array_equal(arrays["classes"], self.bundles[name, count][0].encoder.classes_)
                table = pd.read_csv(output / "shap_class_importance.csv")
                self.assertEqual(set(table["class"]), set(report["classes"]))
                self.assertEqual(len(table), count * 4)
                with self.assertRaises(FileExistsError):
                    run_multiclass_shap(model, csv, output, no_plots=True)

    def test_sampler_is_bounded_reproducible_and_preserves_csv_row_positions(self):
        csv = self.root / "data.csv"
        pd.DataFrame({"f": [0, 1, np.nan, 3, np.inf, 5, 6, 7, 8, 9]}).to_csv(csv, index=False)
        left, report = sample_raw_features(csv, ["f"], 4, max_scan_rows=7, chunk_size=2)
        right, _ = sample_raw_features(csv, ["f"], 4, max_scan_rows=7, chunk_size=3)
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(report["invalid_rows_skipped"], 2)
        self.assertEqual(report["scanned_rows"], 7)
        self.assertEqual(report["sampling_scope"], "prefix")
        self.assertTrue(all(i < 7 and i not in (2, 4) for i in left.index))
        np.testing.assert_array_equal(left.f, left.index)
        _, full = sample_raw_features(csv, ["f"], 4, max_scan_rows=0)
        self.assertEqual(full["sampling_scope"], "full_file")
        with self.assertRaisesRegex(ValueError, "Missing"):
            sample_raw_features(csv, ["missing"], 4)
        with self.assertRaises(ValueError):
            sample_raw_features(csv, ["f"], 0)

    def test_early_stopped_xgboost_uses_the_same_tree_range_as_prediction(self):
        model, _ = self.save_fixture()
        bundle = load_multiclass_bundle(model)
        bundle.model.get_booster().set_attr(best_iteration="0")
        result = explain_bundle(bundle, self.bundles["xgboost", 8][1].iloc[:5], method="native", n_jobs=1)
        np.testing.assert_allclose(result["probabilities"], bundle.model.predict_proba(result["transformed"]), atol=1e-6)
        self.assertLess(result["max_additivity_error"], 1e-5)

    def test_permutation_fallback_is_explicit_capped_and_probability_valued(self):
        model, _ = self.save_fixture()
        bundle = load_multiclass_bundle(model)
        raw = self.bundles["xgboost", 8][1].iloc[::12]
        with patch("ddos_iot.multiclass_shap.native_contributions", side_effect=RuntimeError("simulated native incompatibility")):
            with self.assertRaisesRegex(RuntimeError, "separate training-background"):
                explain_bundle(bundle, raw, method="auto")
            result = explain_bundle(bundle, raw, background=raw.iloc[[0, 3, 6]], fallback_size=2, n_jobs=1)
        self.assertEqual(result["values"].shape, (2, 4, 8))
        self.assertEqual(result["output_space"], "probability")
        self.assertEqual(result["method"], "permutation")
        self.assertIn("simulated", result["fallback_reason"])
        np.testing.assert_allclose(result["base_values"] + result["values"].sum(axis=1), result["probabilities"], atol=1e-6)

    def test_bad_configuration_and_wrong_task_are_rejected_before_outputs(self):
        model, csv = self.save_fixture()
        for options in ({"sample_size": 0}, {"expected_num_classes": 34}, {"class_label": "not_a_class"},
                        {"method": "permutation"}, {"background_file": csv}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_multiclass_shap(model, csv, self.root / "bad", **options)
            self.assertFalse((self.root / "bad").exists())
        args = parse_shap_args(["--model-path", str(model), "--input-file", str(csv)])
        self.assertEqual(args.sample_size, 200)
        self.assertEqual(args.method, "auto")
        self.assertEqual(args.max_scan_rows, 50_000)

    def test_error_does_not_touch_model_and_is_not_published_as_success(self):
        model, csv = self.save_fixture()
        original = model.read_bytes()
        folder = self.root / "failed"
        with patch("ddos_iot.multiclass_shap.native_contributions", side_effect=RuntimeError("calculation failed")):
            with self.assertRaises(RuntimeError):
                run_multiclass_shap(model, csv, folder, method="native", no_plots=True)
        self.assertEqual(model.read_bytes(), original)
        self.assertEqual(json.loads((folder / "shap_report.json").read_text())["status"], "failed")
        self.assertFalse((folder / "shap_values.npz").exists())

    def test_class_axis_and_additivity_errors_are_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "axes"):
            check_additivity(np.zeros((2, 8, 4)), np.zeros((2, 8)), np.zeros((2, 8)), 2, 4, 8)
        with self.assertRaisesRegex(ValueError, "additivity"):
            check_additivity(np.zeros((2, 4, 8)), np.zeros((2, 8)), np.ones((2, 8)), 2, 4, 8)

    def test_plot_exports_and_saved_array_view_need_no_model(self):
        from ddos_iot.multiclass_shap_plots import load_arrays, waterfall_figure, plt

        model, csv = self.save_fixture()
        report = run_multiclass_shap(model, csv, self.root / "plots", sample_size=8, local_count=1, n_jobs=1)
        folder = Path(report["output_path"])
        self.assertEqual(len(list(folder.glob("*.png"))), 4)
        self.assertTrue(all(p.stat().st_size > 1000 for p in folder.glob("*.png")))
        with patch("joblib.load", side_effect=AssertionError("Viewing must not load model")):
            arrays = load_arrays(folder)
            figure = waterfall_figure(arrays, 0, 7)
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
