"""Regression tests for raw 34-class handling, validation isolation, and artifacts."""

from __future__ import annotations

import sys
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiclass import parse_args
from ddos_iot.multiclass import (
    MODEL_NAMES, above_median_features, build_model, class_weights, device_for, encode_labels, evaluate,
    fit_features, fit_model, grid_search, load_raw_split, predict_ids, resolve_parameter_grids,
    run_multiclass_training, scoring_device_for, split_training,
)


def fixture(rows_per_class: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    y = np.repeat(np.arange(34), rows_per_class)
    data = pd.DataFrame({"signal": y.astype(float), "noise": rng.normal(size=len(y)), "constant": 1.0})
    for k in range(34):
        data[f"class_feature_{k}"] = (y == k).astype(float)
    for k in range(9):
        data[f"noise_{k}"] = rng.normal(size=len(y))
    data["label"] = [f"attack_{33 - k:02d}" for k in y]
    return data


class MulticlassTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="multiclass_test_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, *extra):
        return parse_args(["--train-input-path", str(self.root / "train.csv"),
                           "--test-input-path", str(self.root / "test.csv"),
                           "--output-path", str(self.root / "results"),
                           "--n-estimators", "3", "--n-jobs", "2", "--log-period", "0",
                           "--feature-selection", "none", "--cv-folds", "0", "--no-plots", *extra])

    def test_default_configuration_enables_five_fold_median_grid_search(self):
        args = parse_args([])
        self.assertEqual(args.cv_folds, 5)
        self.assertEqual(args.feature_selection, "median")
        grids = resolve_parameter_grids(args)
        self.assertEqual(set(grids), set(MODEL_NAMES))
        for grid in grids.values():
            self.assertEqual(grid, {"n_estimators": [500], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]})

    def test_gpu_training_and_scoring_devices(self):
        args = self.args("--device", "gpu", "--lightgbm-device", "cpu", "--gpu-id", "1")
        for name in MODEL_NAMES:
            self.assertEqual(device_for(name, args), "cpu" if name == "lightgbm" else "gpu")
            self.assertEqual(scoring_device_for(name, args), "gpu" if name == "xgboost" else "cpu")
        self.assertEqual(build_model("xgboost", args, 3, False).get_params()["device"], "cuda:1")
        self.assertEqual(build_model("catboost", args, 3, False).get_params()["task_type"], "GPU")
        self.assertEqual(build_model("lightgbm", args, 3, False).get_params()["device_type"], "cpu")
        for name in MODEL_NAMES:
            self.assertEqual(scoring_device_for(name, self.args()), "cpu")

    def test_gpu_prediction_batches_and_preserves_early_stopping(self):
        x = fixture().drop(columns="label").iloc[:5]
        for best_iteration, expected_range in ((1, (0, 2)), (None, (0, 0))):
            with self.subTest(best_iteration=best_iteration):
                booster = MagicMock()
                booster.save_config.return_value = json.dumps({"learner": {"generic_param": {"device": "cuda:1"}}})
                booster.predict.side_effect = lambda batch, **kwargs: np.eye(34)[np.arange(len(batch)) % 2]
                model = SimpleNamespace(n_jobs=2, set_params=MagicMock(), get_booster=lambda: booster)
                if best_iteration is not None:
                    model.best_iteration = best_iteration
                with patch("xgboost.DMatrix", side_effect=lambda batch, **kwargs: batch):
                    predicted = predict_ids("xgboost", model, x, batch_size=2, device="gpu", gpu_id=1)
                np.testing.assert_array_equal(predicted, [0, 1, 0, 1, 0])
                model.set_params.assert_called_once_with(device="cuda:1")
                self.assertEqual(booster.predict.call_count, 3)
                for call in booster.predict.call_args_list:
                    self.assertEqual(call.kwargs["iteration_range"], expected_range)
                    self.assertTrue(call.kwargs["strict_shape"])

    def test_gpu_prediction_rejects_silent_cpu_fallback(self):
        model = MagicMock()
        model.best_iteration = 0
        model.n_jobs = 2
        model.get_booster.return_value.predict.return_value = np.eye(34)[:1]
        model.get_booster.return_value.save_config.return_value = json.dumps(
            {"learner": {"generic_param": {"device": "cpu"}}})
        x = fixture().drop(columns="label").iloc[:1]
        with patch("xgboost.DMatrix"), self.assertRaisesRegex(RuntimeError, "GPU prediction but it used cpu"):
            predict_ids("xgboost", model, x, device="gpu")

    def test_default_prediction_is_cpu_and_unsupported_gpu_is_explicit(self):
        x = fixture().drop(columns="label").iloc[:2]
        model = MagicMock()
        model.predict.return_value = np.array([0, 1])
        predict_ids("xgboost", model, x)
        model.set_params.assert_called_once_with(device="cpu")
        model.reset_mock()
        predict_ids("catboost", model, x)
        self.assertEqual(model.predict.call_args.kwargs, {"task_type": "CPU"})
        for name in ("catboost", "lightgbm"):
            with self.assertRaisesRegex(ValueError, "GPU prediction"):
                predict_ids(name, model, x, device="gpu")

    @unittest.skipUnless(os.environ.get("DDOS_TEST_GPU") == "1", "opt-in CUDA integration test")
    def test_real_gpu_cv_and_prediction_match_cpu(self):
        data = fixture(12)
        x = data.drop(columns="label")
        _, y, _ = encode_labels(data["label"], data["label"])
        args = self.args("--device", "gpu", "--models", "xgboost", "catboost", "--cv-folds", "5")
        for name in ("xgboost", "catboost"):
            result = grid_search(name, x, y, args, {"n_estimators": [3]}, self.root / name)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["training_device"], "gpu")
            self.assertEqual(result["scoring_device"], "gpu" if name == "xgboost" else "cpu")
            self.assertEqual(len(result["best_fold_scores"]), 5)
        fit, valid = split_training(y, .2, 42)
        for early_stopping in (True, False):
            model = build_model("xgboost", args, 3, early_stopping)
            validation = (x.iloc[valid], y[valid]) if early_stopping else None
            fit_model("xgboost", model, x.iloc[fit], y[fit], class_weights(y[fit], "balanced"), args, validation)
            gpu = predict_ids("xgboost", model, x.iloc[valid], batch_size=31, device="gpu")
            cpu = predict_ids("xgboost", model, x.iloc[valid], batch_size=31)
            np.testing.assert_array_equal(gpu, cpu)

    def test_strict_median_keeps_zeros_in_population_and_excludes_ties(self):
        scores = dict(zip(fixture().drop(columns="label").columns, [0.] * 24 + [2.] * 10 + [4.] * 12))
        selected, threshold = above_median_features(scores)
        self.assertEqual(threshold, 0)
        self.assertEqual(len(selected), 22)
        tied = dict(zip(scores, [0.] * 22 + [2.] * 12 + [4.] * 12))
        selected, threshold = above_median_features(tied)
        self.assertEqual(threshold, 2)
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(tied[c] > threshold for c in selected))
        with self.assertRaisesRegex(ValueError, "No feature"):
            above_median_features(dict.fromkeys(scores, 0))
        with self.assertRaisesRegex(ValueError, "all 46"):
            above_median_features({k: v for k, v in scores.items() if k != "constant"})

    def test_selector_receives_all_46_including_constants_and_identifiers(self):
        data = fixture().rename(columns={"noise_0": "source_id"})
        x = data.drop(columns="label")
        _, y, _ = encode_labels(data["label"], data["label"])
        with patch("lightgbm.LGBMClassifier") as estimator:
            estimator.return_value.feature_importances_ = np.array([0.] * 24 + [2.] * 22)
            transform = fit_features(x, y, self.args("--feature-selection", "median"))
        seen = estimator.return_value.fit.call_args.args[0]
        self.assertEqual(list(seen.columns), list(x.columns))
        self.assertIn("constant", seen)
        self.assertIn("source_id", seen)
        self.assertEqual(len(transform.scores), 46)
        self.assertEqual(transform.gain_threshold, 0)
        self.assertEqual(len(transform.columns), 22)

    def test_weights_equalize_class_contributions(self):
        y = np.repeat(np.arange(34), np.arange(2, 36))
        weights = class_weights(y, "balanced")
        np.testing.assert_allclose(np.bincount(y, weights=weights[y]), len(y) / 34, rtol=1e-6)
        self.assertAlmostEqual(float(weights[y].mean()), 1.0, places=6)
        np.testing.assert_array_equal(class_weights(y, "none"), np.ones(34))

    def test_label_mapping_reused_and_wrong_classes_rejected(self):
        labels = fixture()["label"]
        encoder, y, yt = encode_labels(labels, labels.iloc[::-1])
        np.testing.assert_array_equal(encoder.inverse_transform(y), labels)
        np.testing.assert_array_equal(yt, y[::-1])
        with self.assertRaisesRegex(ValueError, "Expected 34"):
            encode_labels(labels[labels != "attack_00"], labels)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            encode_labels(labels, labels.replace("attack_00", "unknown"))

    def test_stream_sampling_scans_all_classes_and_records_invalid_rows(self):
        data = fixture(8)
        data.loc[0, "signal"] = np.inf
        data.loc[1, "label"] = None
        path = self.root / "train.csv"
        data.to_csv(path, index=False)
        sampled, report = load_raw_split(path, "label", 3, 11, 42)
        again, _ = load_raw_split(path, "label", 3, 11, 42)
        pd.testing.assert_frame_equal(sampled, again)
        self.assertEqual(sampled["label"].nunique(), 34)
        self.assertEqual(len(sampled), 102)
        self.assertEqual(report["removed_invalid_rows"], 2)
        full, _ = load_raw_split(path, "label", 0, 17, 42)
        self.assertEqual(len(full), len(data) - 2)
        self.assertTrue(np.isfinite(sampled.drop(columns="label")).all().all())

    def test_split_and_features_use_training_only(self):
        data = fixture()
        _, y, _ = encode_labels(data["label"], data["label"])
        fit, valid = split_training(y, 0.2, 42)
        self.assertFalse(set(fit) & set(valid))
        self.assertEqual(len(set(y[fit])), 34)
        self.assertEqual(len(set(y[valid])), 34)
        x = data.drop(columns="label")
        transform = fit_features(x.iloc[fit], y[fit], self.args())
        self.assertEqual(len(transform.columns), 46)
        np.testing.assert_allclose(transform.scaler.mean_, x.iloc[fit][transform.columns].mean(), rtol=1e-5)
        expected = transform.transform(x.iloc[valid])
        reordered = transform.transform(x.iloc[valid][list(reversed(x.columns))])
        pd.testing.assert_frame_equal(expected, reordered)
        before = transform.scaler.mean_.copy()
        transform.transform(x.iloc[valid] + 1_000_000)
        np.testing.assert_array_equal(transform.scaler.mean_, before)

    def test_all_models_train_and_saved_labels_match_report(self):
        train, test = fixture(12), fixture(3)
        train.to_csv(self.root / "train.csv", index=False)
        test.to_csv(self.root / "test.csv", index=False)
        grid_path = self.root / "grid.json"
        grid_path.write_text(json.dumps({name: {"max_depth": [4, 6]} for name in MODEL_NAMES}), encoding="utf-8")
        args = self.args("--cv-folds", "5", "--param-grid", str(grid_path), "--feature-selection", "median")
        result = run_multiclass_training(args)
        self.assertEqual(result["status"], "complete")
        self.assertEqual({r["model"] for r in result["results"]}, set(MODEL_NAMES))
        self.assertEqual(result["selection_metric"], "cv_macro_f1_mean")
        for row in result["results"]:
            folder = self.root / "results" / row["model"]
            bundle = joblib.load(folder / "model.joblib")
            actual = bundle.predict(test.drop(columns="label"))
            saved = pd.read_csv(folder / "test_predictions.csv")
            np.testing.assert_array_equal(actual, saved["predicted_label"])
            self.assertGreater(len(set(actual)), 5, row["model"])
            self.assertEqual(row["training_rows"], len(train))
            metrics = evaluate(bundle.encoder.transform(test["label"]), bundle.encoder.transform(actual))
            self.assertEqual(metrics["macro_f1"], row["macro_f1"])
            matrix = pd.read_csv(folder / "confusion_matrix.csv", index_col=0)
            self.assertEqual(matrix.shape, (34, 34))
            self.assertEqual(matrix.to_numpy().sum(), len(test))
            search = json.loads((folder / "grid_search_report.json").read_text())
            metadata = json.loads((folder / "model_metadata.json").read_text())
            self.assertEqual(search["status"], "complete")
            self.assertEqual(len(search["candidates"]), 2)
            self.assertTrue(all(len(c["fold_scores"]) == 5 for c in search["candidates"]))
            self.assertAlmostEqual(row["cv_macro_f1_mean"], search["best_score"])
            self.assertEqual(metadata["selected_hyperparameters"], search["best_params"])
            depth_key = "depth" if row["model"] == "catboost" else "max_depth"
            self.assertEqual(metadata["parameters"][depth_key], search["best_params"]["max_depth"])
            self.assertEqual(len(bundle.features.scores), 46)
            self.assertTrue(all(bundle.features.scores[c] > bundle.features.gain_threshold for c in bundle.features.columns))
        with self.assertRaises(FileExistsError):
            run_multiclass_training(args)

    def test_feature_selector_and_five_fold_cv(self):
        data = fixture(12)
        x = data.drop(columns="label")
        _, y, _ = encode_labels(data["label"], data["label"])
        args = self.args("--feature-selection", "median", "--selector-sample-size", "100")
        transform = fit_features(x, y, args)
        self.assertGreater(len(transform.columns), 0)
        self.assertLessEqual(len(transform.columns), 23)
        self.assertEqual(len(transform.scores), 46)
        self.assertIn("signal", transform.scores)
        cv_args = self.args("--cv-folds", "5", "--no-refit-full")
        with patch("ddos_iot.multiclass.fit_features", wraps=fit_features) as fit_spy:
            result = grid_search("xgboost", x, y, cv_args,
                                 {"n_estimators": [3], "max_depth": [4, 6]}, self.root / "search")
        self.assertEqual(len(result["best_fold_scores"]), 5)
        self.assertEqual(fit_spy.call_count, 5)
        from sklearn.model_selection import StratifiedKFold
        folds = StratifiedKFold(5, shuffle=True, random_state=cv_args.random_state)
        for call, (train, held_out) in zip(fit_spy.call_args_list, folds.split(x, y)):
            self.assertLess(len(call.args[0]), len(x) * 0.8)
            self.assertFalse(set(call.args[0].index) & set(held_out))
            self.assertTrue(set(call.args[0].index) <= set(train))

    def test_grid_winner_is_largest_mean_score_not_last_candidate(self):
        data = fixture()
        x = data.drop(columns="label")
        _, y, _ = encode_labels(data["label"], data["label"])
        args = self.args("--cv-folds", "5", "--no-refit-full")

        def predict(name, model, batch, **kwargs):
            return y[batch.index] if model.depth == 4 else np.zeros(len(batch), dtype=int)

        with patch("ddos_iot.multiclass.build_model", side_effect=lambda name, a, *v: SimpleNamespace(depth=a.max_depth)), \
             patch("ddos_iot.multiclass.fit_model"), \
             patch("ddos_iot.multiclass.best_rounds", return_value=3), \
             patch("ddos_iot.multiclass.predict_ids", side_effect=predict):
            result = grid_search("xgboost", x, y, args, {"max_depth": [4, 6]}, self.root / "search")
        self.assertEqual(result["best_params"], {"max_depth": 4})
        self.assertEqual(result["best_score"], 1.0)
        saved = pd.read_csv(self.root / "search" / "grid_search_results.csv")
        self.assertEqual(saved.loc[saved["rank"] == 1, "max_depth"].iloc[0], 4)

    def test_grid_rejects_invalid_values_before_loading_data(self):
        path = self.root / "grid.json"
        args = self.args("--cv-folds", "5", "--param-grid", str(path))
        for invalid in ({"xgboost": {"max_depth": []}}, {"xgboost": {"device": ["cpu"]}},
                        {"catboost": {"max_depth": [17]}}, {"xgboost": {"learning_rate": [float("nan")]}},
                        {"xgboost": {"n_estimators": [True]}}, {"typo": {"max_depth": [4]}}):
            with self.subTest(grid=invalid):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    resolve_parameter_grids(args)

    def test_failed_grid_candidate_is_excluded(self):
        from ddos_iot.multiclass import build_model
        data = fixture()
        x = data.drop(columns="label")
        _, y, _ = encode_labels(data["label"], data["label"])

        def fail_candidate(name, args, *other):
            if args.max_depth == 4:
                raise RuntimeError("Simulated invalid candidate")
            return build_model(name, args, *other)

        args = self.args("--cv-folds", "5", "--no-refit-full")
        with patch("ddos_iot.multiclass.build_model", side_effect=fail_candidate), \
             self.assertLogs("ddos_iot.multiclass", level="ERROR"):
            result = grid_search("xgboost", x, y, args, {"max_depth": [4, 6]}, self.root / "search")
        self.assertEqual(result["best_params"], {"max_depth": 6})
        self.assertEqual(result["failed_candidates"], 1)
        self.assertEqual(result["candidates"][0]["status"], "failed")

    def test_failure_is_reported_and_next_model_runs(self):
        from ddos_iot.multiclass import build_model

        fixture().to_csv(self.root / "train.csv", index=False)
        fixture(3).to_csv(self.root / "test.csv", index=False)

        def fail_one(name, *args, **kwargs):
            if name == "xgboost":
                raise RuntimeError("Simulated GPU failure")
            return build_model(name, *args, **kwargs)

        with patch("ddos_iot.multiclass.build_model", side_effect=fail_one), \
             self.assertLogs("ddos_iot.multiclass", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "xgboost"):
                run_multiclass_training(self.args("--models", "xgboost", "lightgbm"))
        report = json.loads((self.root / "results" / "model_comparison_report.json").read_text())
        self.assertEqual(report["status"], "partial_failure")
        self.assertEqual(report["best_model"], "lightgbm")
        self.assertIn("xgboost", report["failures"])


if __name__ == "__main__":
    unittest.main()
