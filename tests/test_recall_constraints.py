"""Recall/FPR feasibility, frozen OOF selection, and cached-run isolation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.config import Candidate, ExperimentConfig
from ddos_iot.experiments.constrained import export_constrained_choice, run_constrained_tuning
from ddos_iot.experiments.decisions import predict_with_bias, tune_biases
from ddos_iot.experiments.recall_constraints import (
    DecisionConstraints, constraint_statistics, decision_statistics, reference_constraints,
    tune_constrained_biases,
)
from ddos_iot.experiments.runner import run_development
from ddos_iot.experiments.storage import file_digest
from test_multiclass_8 import family_fixture


class RecallConstraintTests(unittest.TestCase):
    def test_reduces_false_alarms_without_losing_attack_recall(self):
        y = np.array([0] * 8 + [1] * 2)
        p = np.array([[.45, .55]] * 4 + [[.9, .1]] * 4 + [[.1, .9]] * 2, dtype=np.float32)
        policy = reference_constraints(y, p.argmax(axis=1), 2, benign_id=0, protected_ids=(1,))
        bias, report = tune_constrained_biases(y, p, constraints=policy, grid=(-2., -1., 0., 1., 2.))
        np.testing.assert_array_equal(predict_with_bias(p, bias), y)
        self.assertEqual(report["tuned"]["macro_f1"], 1.)
        self.assertTrue(report["tuned"]["feasible"])
        self.assertFalse(report["validation_used_for_tuning"])
        self.assertFalse(report["test_used_for_tuning"])

    def test_reference_fallback_multistart_and_predictions_are_exact(self):
        rng = np.random.default_rng(9)
        y = rng.integers(0, 4, 1200)
        p = rng.dirichlet(np.ones(4), len(y)).astype(np.float32)
        policy = reference_constraints(y, p.argmax(axis=1), 4, benign_id=0, protected_ids=(1, 3))
        unconstrained, _ = tune_biases(y, p, grid=(-1., 0., 1.))
        args = dict(constraints=policy, grid=(-2., -1., 0., 1., 2.), starts=(unconstrained,))
        bias, report = tune_constrained_biases(y, p, **args)
        again, _ = tune_constrained_biases(y, p, **args)
        np.testing.assert_array_equal(bias, again)
        predicted = predict_with_bias(p, bias)
        stats = decision_statistics(y, predicted, 4)
        self.assertEqual(stats, {key: report["tuned"][key] for key in stats})
        self.assertAlmostEqual(stats["macro_f1"], f1_score(y, predicted, average="macro"))
        self.assertTrue(constraint_statistics(stats, policy)["feasible"])
        self.assertGreaterEqual(stats["macro_f1"] + 1e-12, report["baseline"]["macro_f1"])
        for start in report["starts"]:
            feasible_seen = False
            for step in start["history"]:
                if feasible_seen:
                    self.assertTrue(step["feasible"])
                feasible_seen |= step["feasible"]

    def test_no_feasible_search_point_is_not_exported_as_a_model(self):
        y = np.array([0] * 8 + [1] * 2)
        p = np.tile([.9, .1], (len(y), 1))
        bias, report = tune_constrained_biases(y, p, constraints=DecisionConstraints(0, {1: 1.}, 0.),
                                              grid=(-4., 0., 4.))
        self.assertIsNone(bias)
        self.assertIsNone(report["bias"])
        self.assertEqual(report["status"], "no_feasible_point_found")

    def test_rate_bounds_use_unrounded_counts_and_argmax_ties(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        predicted = np.array([0, 0, 1, 1, 1, 0])
        policy = reference_constraints(y, predicted, 2, benign_id=0, protected_ids=(1,))
        self.assertTrue(constraint_statistics(decision_statistics(y, predicted, 2), policy)["feasible"])
        worse = predicted.copy(); worse[-2] = 0
        self.assertFalse(constraint_statistics(decision_statistics(y, worse, 2), policy)["feasible"])
        p = np.full((len(y), 2), .5)
        bias, report = tune_constrained_biases(y, p, constraints=DecisionConstraints(0, {1: 0.}, 0.), grid=(0.,))
        np.testing.assert_array_equal(predict_with_bias(p, bias), np.zeros(len(y)))
        self.assertTrue(report["tuned"]["feasible"])

    def test_invalid_constraints_and_inputs(self):
        for values in [(0, {}, .2), (0, {0: .2}, .2), (0, {1: float("nan")}, .2),
                       (0, {1: .2}, -.1), (True, {1: .2}, .1)]:
            with self.assertRaises(ValueError):
                DecisionConstraints(*values)
        policy = DecisionConstraints(0, {1: .5}, .5)
        for y, p in [(np.array([0, 0]), np.full((2, 2), .5)),
                     (np.array([0., 1.]), np.full((2, 2), .5))]:
            with self.assertRaises(ValueError):
                tune_constrained_biases(y, p, constraints=policy, grid=(0.,))

    def test_cached_run_freezes_before_validation_and_rejects_overwrite(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="recall_test_", dir=ROOT / "tmp") as temporary:
            root = Path(temporary)
            family_fixture(20).to_csv(root / "train.csv", index=False)
            config = ExperimentConfig(train_input_path=root / "train.csv", test_input_path=root / "missing_test.csv",
                                      output_path=root / "run", candidates=(Candidate("reference", "none", 1.),),
                                      folds=2, n_estimators=3, max_depth=2, selector_sample_size=500,
                                      bias_grid=(-1., 0., 1.), bias_passes=1, latency_repeats=1)
            run_development(config)
            original_digest = file_digest(config.output_path / "selection.json")
            destination = root / "constrained"
            from ddos_iot.experiments.constrained import _validation_features

            def after_freeze(*args):
                selection = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
                self.assertFalse(selection["validation_used_for_selection"])
                self.assertTrue((destination / "seed_42/reference/bias.json").exists())
                return _validation_features(*args)

            with patch("ddos_iot.experiments.constrained._validation_features", side_effect=after_freeze), \
                 patch("ddos_iot.experiments.training.fit_partition", side_effect=AssertionError("Retraining forbidden")):
                result = run_constrained_tuning(config.output_path, destination, grid=(-1., 0., 1.), passes=1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["models_trained"], 0)
            self.assertEqual(file_digest(config.output_path / "selection.json"), original_digest)
            self.assertFalse(config.test_input_path.exists())
            self.assertTrue(result["validation_results"])
            from ddos_iot.multiclass import load_multiclass_bundle
            import pandas as pd
            exported = export_constrained_choice(destination)
            bundle = load_multiclass_bundle(exported / "seed_42/model.joblib")
            with np.load(config.output_path / "seed_42/splits.npz") as split:
                ids = split["validation_row_ids"]
            x = pd.read_csv(config.train_input_path).drop(columns="label").iloc[ids]
            with np.load(destination / "seed_42/reference/validation/predictions.npz") as saved:
                np.testing.assert_array_equal(bundle.encoder.transform(bundle.predict(x)), saved["predicted"])
            with self.assertRaises(FileExistsError):
                run_constrained_tuning(config.output_path, destination)
            rows_path = config.output_path / "seed_42/reference/oof/rows.npz"
            with np.load(rows_path) as rows:
                damaged = {name: rows[name].copy() for name in rows.files}
            damaged["row_ids"] = damaged["row_ids"][::-1]
            np.savez(rows_path, **damaged)
            with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                run_constrained_tuning(config.output_path, root / "damaged")


if __name__ == "__main__":
    unittest.main()
