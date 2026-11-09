"""Scientific isolation, row alignment, tempered weights, biases, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.config import Candidate, ExperimentConfig, config_from_dict, experiment_plan
from ddos_iot.experiments.data import development_split, load_training_data
from ddos_iot.experiments.decisions import predict_with_bias, tune_biases
from ddos_iot.experiments.runner import evaluate_frozen_test, run_development
from ddos_iot.experiments.storage import object_digest
from ddos_iot.multiclass import fit_features, load_multiclass_bundle, predict_probabilities
from ddos_iot.weighting import balanced_class_weights
from test_multiclass_8 import family_fixture


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="experiments_test_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, classes=8, **overrides):
        values = dict(num_classes=classes, train_input_path=self.root / "train.csv",
                      test_input_path=self.root / "test.csv", output_path=self.root / f"run_{classes}",
                      candidates=(Candidate("all_unweighted", "none", 0.),), folds=2,
                      n_estimators=3, max_depth=2, selector_sample_size=500,
                      bias_grid=(-1., 0., 1.), bias_passes=1, latency_repeats=1, n_jobs=2)
        values.update(overrides)
        return ExperimentConfig(**values)

    def test_tempered_weights_have_mean_sample_weight_one(self):
        y = np.repeat(np.arange(3), [100, 10, 2])
        for power in (0., 0.5, 0.75, 1.):
            weights = balanced_class_weights(y, 3, power)
            self.assertAlmostEqual(float(weights[y].mean()), 1., places=6)
            self.assertAlmostEqual(float(weights[2] / weights[0]), 50**power, places=4)
        np.testing.assert_array_equal(balanced_class_weights(y, 3, 0.), np.ones(3))
        for power in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                balanced_class_weights(y, 3, power)
        with self.assertRaisesRegex(ValueError, "Every training"):
            balanced_class_weights(y, 4)

    def test_bias_search_can_reduce_false_positives_and_matches_exported_decision(self):
        y = np.array([0] * 8 + [1] * 2)
        p = np.array([[.45, .55]] * 4 + [[.9, .1]] * 4 + [[.1, .9]] * 2, dtype=np.float32)
        bias, report = tune_biases(y, p, grid=(-2., -1., 0., 1., 2.))
        self.assertLess(bias[1], 0)
        predicted = predict_with_bias(p, bias)
        self.assertEqual(report["tuned_oof_macro_f1"], f1_score(y, predicted, average="macro"))
        self.assertGreater(report["tuned_oof_macro_f1"], report["baseline_oof_macro_f1"])
        self.assertFalse(report["validation_used_for_tuning"])
        # The opposite error direction must also be recoverable.
        p = np.array([[.9, .1]] * 8 + [[.55, .45]] * 2, dtype=np.float32)
        bias, _ = tune_biases(y, p, grid=(-1., 0., 1.))
        self.assertGreater(bias[1], 0)
        np.testing.assert_array_equal(predict_with_bias(p, bias), y)
        np.testing.assert_array_equal(predict_with_bias(np.array([[.5, .5]]), np.zeros(2)), [0])
        with self.assertRaises(ValueError):
            predict_with_bias(np.array([[.2, .3]]), np.zeros(2))

    def test_configuration_rejects_unsafe_or_ambiguous_experiments(self):
        for fields in ({"folds": 1}, {"seeds": (42, 42)}, {"latency_repeats": 0},
                       {"bias_grid": (float("nan"),)}, {"n_jobs": 0}, {"validation_size": 1.}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.config(**fields)
        with self.assertRaises(ValueError):
            Candidate("../escape")
        with self.assertRaises(TypeError):
            config_from_dict({"unknown_setting": True})
        config = self.config()
        self.assertEqual(config_from_dict(config.as_dict()), config)
        self.assertEqual(config_from_dict({"num_classes": 34}).output_path.name, "34class_v1")
        self.assertFalse(experiment_plan(config)["test_file_read"])

    def test_common_raw_rows_survive_caps_invalid_rows_and_both_label_mappings(self):
        frame = family_fixture(20)
        frame.loc[0, "f0"] = np.inf
        frame.to_csv(self.root / "train.csv", index=False)
        first = load_training_data(self.config(8, max_train_per_raw_class=10))
        second = load_training_data(self.config(34, max_train_per_raw_class=10))
        self.assertNotIn(0, first.row_ids)
        self.assertEqual(first.report["removed_invalid_rows"], 1)
        np.testing.assert_array_equal(first.row_ids, second.row_ids)
        for left, right in zip(development_split(first, self.config(8), 42),
                               development_split(second, self.config(34), 42)):
            np.testing.assert_array_equal(first.row_ids[left], second.row_ids[right])

    def test_real_oof_validation_and_frozen_test_both_tasks(self):
        frame = family_fixture(20)
        frame.to_csv(self.root / "train.csv", index=False)
        previous_splits = None
        for classes in (8, 34):
            with self.subTest(classes=classes):
                config = self.config(classes, seeds=(42, 123))
                # A missing test CSV is intentional: development cannot touch it.
                with patch("ddos_iot.experiments.runner.load_test_data", side_effect=AssertionError("Test touched")), \
                     patch("ddos_iot.experiments.training.fit_features", wraps=fit_features) as selector:
                    result = run_development(config)
                self.assertEqual(result["status"], "complete")
                selection_report = json.loads((config.output_path / "selection_report.json").read_text())
                self.assertEqual(selection_report["selection_policy"], "reference_fpr")
                self.assertTrue(selection_report["selected"]["fpr_guardrail_pass"])
                summary = pd.read_csv(config.output_path / "validation_summary.csv")
                self.assertTrue((summary.seed_count == 2).all())
                self.assertFalse(config.test_input_path.exists())
                folder = config.output_path / "seed_42" / "all_unweighted"
                with np.load(config.output_path / "seed_42" / "splits.npz") as splits:
                    development = splits["development_row_ids"].copy()
                    validation = splits["validation_row_ids"].copy()
                if previous_splits is not None:
                    np.testing.assert_array_equal(development, previous_splits[0])
                    np.testing.assert_array_equal(validation, previous_splits[1])
                previous_splits = development, validation
                with np.load(folder / "oof" / "rows.npz") as rows:
                    fold_ids, y = rows["fold_ids"].copy(), rows["y"].copy()
                    np.testing.assert_array_equal(rows["row_ids"], development)
                self.assertTrue(np.all(fold_ids > 0))
                p = np.load(folder / "oof" / "probabilities.npy")
                self.assertEqual(p.shape, (len(development), classes))
                np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
                for fold, call in enumerate(selector.call_args_list[:2], 1):
                    seen = set(call.args[0].index)
                    self.assertFalse(seen & set(development[fold_ids == fold]))
                    self.assertFalse(seen & set(validation))
                self.assertFalse(set(selector.call_args_list[config.folds].args[0].index) & set(validation))
                bundle = load_multiclass_bundle(folder / "model.joblib")
                x = frame.drop(columns="label").iloc[:20]
                expected = bundle.predict(x)
                # Previous pickle state remains loadable even without the new attribute.
                del bundle.decision_bias
                np.testing.assert_array_equal(bundle.predict(x), expected)
                np.testing.assert_array_equal(bundle.predict_proba(x).argmax(axis=1), bundle.encoder.transform(expected))
                with self.assertRaises(FileExistsError):
                    run_development(config)
                # Create a valid, complete test source only after development.
                family_fixture(3).to_csv(config.test_input_path, index=False)
                from ddos_iot.experiments.data import load_test_data

                def load_after_save(*args):
                    for seed in config.seeds:
                        self.assertTrue((config.output_path / "test_evaluation" / f"seed_{seed}" / "model.joblib").exists())
                    return load_test_data(*args)

                with patch("ddos_iot.experiments.runner.load_test_data", side_effect=load_after_save):
                    final = evaluate_frozen_test(config.output_path)
                self.assertEqual(final["status"], "complete")
                self.assertEqual({r["seed"] for r in final["results"]}, {42, 123})
                self.assertGreater(final["results"][0]["training_seconds"], 0)
                with self.assertRaises(FileExistsError):
                    evaluate_frozen_test(config.output_path)
                final_folder = config.output_path / "test_evaluation" / "seed_42"
                saved = load_multiclass_bundle(final_folder / "model.joblib")
                test_frame = pd.read_csv(config.test_input_path).drop(columns="label")
                with np.load(final_folder / "predictions.npz") as exported:
                    np.testing.assert_array_equal(saved.encoder.transform(saved.predict(test_frame)), exported["predicted"])
                if classes == 34:
                    self.assertTrue((final_folder / "family_confusion_matrix.csv").exists())
                config.test_input_path.unlink()

    def test_modified_frozen_selection_is_rejected_before_training(self):
        family_fixture(20).to_csv(self.root / "train.csv", index=False)
        config = self.config()
        run_development(config)
        path = config.output_path / "selection.json"
        payload = json.loads(path.read_text())
        payload["decision"] = "modified"
        path.write_text(json.dumps(payload))
        with patch("ddos_iot.experiments.runner.fit_partition") as fit:
            with self.assertRaisesRegex(ValueError, "modified"):
                evaluate_frozen_test(config.output_path)
            fit.assert_not_called()

    def test_older_frozen_config_without_guardrail_fields_can_still_be_evaluated(self):
        family_fixture(20).to_csv(self.root / "train.csv", index=False)
        config = self.config(selection_policy="macro_f1")
        run_development(config)
        config_path = config.output_path / "config.json"
        payload = json.loads(config_path.read_text())
        for field in ("selection_policy", "reference_candidate", "max_benign_fpr_increase"):
            payload.pop(field)
        config_path.write_text(json.dumps(payload))
        selection_path = config.output_path / "selection.json"
        selection = json.loads(selection_path.read_text())
        selection["config_sha256"] = object_digest(payload)
        selection_path.write_text(json.dumps(selection))
        report_path = config.output_path / "experiment_report.json"
        report = json.loads(report_path.read_text())
        report["config_sha256"] = object_digest(payload)
        report["selection_sha256"] = object_digest(selection)
        report_path.write_text(json.dumps(report))
        family_fixture(3).to_csv(config.test_input_path, index=False)
        self.assertEqual(evaluate_frozen_test(config.output_path)["status"], "complete")

    def test_failure_status_retains_partial_oof_and_does_not_select_a_winner(self):
        family_fixture(20).to_csv(self.root / "train.csv", index=False)
        config = self.config()
        with patch("ddos_iot.experiments.oof.fit_partition", side_effect=RuntimeError("simulated fit error")), \
             self.assertLogs("ddos_iot.experiments.runner", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "No experiments"):
                run_development(config)
        report = json.loads((config.output_path / "experiment_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["failures"]), 1)
        self.assertFalse((config.output_path / "selection.json").exists())
        oof = config.output_path / "seed_42" / "all_unweighted" / "oof"
        self.assertEqual(json.loads((oof / "report.json").read_text())["status"], "failed")

    def test_gpu_probability_export_preserves_class_order_and_tree_range(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        booster = MagicMock()
        booster.predict.side_effect = lambda batch, **kwargs: np.tile([.1, .7, .2], (len(batch), 1))
        booster.save_config.return_value = json.dumps({"learner": {"generic_param": {"device": "cuda:0"}}})
        model = SimpleNamespace(set_params=MagicMock(), get_booster=lambda: booster, n_jobs=2, best_iteration=3)
        with patch("xgboost.DMatrix", side_effect=lambda batch, **kwargs: batch):
            p = predict_probabilities("xgboost", model, pd.DataFrame({"f": range(5)}), 2, device="gpu")
        np.testing.assert_allclose(p, np.tile([.1, .7, .2], (5, 1)))
        self.assertEqual(booster.predict.call_count, 3)
        for call in booster.predict.call_args_list:
            self.assertEqual(call.kwargs["iteration_range"], (0, 4))


if __name__ == "__main__":
    unittest.main()
