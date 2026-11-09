"""Routing coverage, meta cross-fitting, and frozen multiseed confirmation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.storage import close_memmap, file_digest
from ddos_iot.experiments.study import run_training_study
from ddos_iot.experiments.study_confirmation import confirm_training_study
from ddos_iot.experiments.study_models import log_probability_features, route_specialist
from ddos_iot.experiments.study_specialist import fit_specialist, specialist_group, run_specialist_study
from ddos_iot.experiments.study_stacking import crossfit_meta, fit_meta
from ddos_iot.experiments.study_config import TrainingTrial
import test_training_study as study_fixture


class StudyEnsembleTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_specialist_preserves_mass_and_exposes_gate_misses(self):
        base = np.array([[.6, .3, .1], [.05, .15, .8], [.2, .7, .1]], dtype="float32")
        local = np.array([[.1, .9], [.9, .1]], dtype="float32")
        changed, routed = route_specialist(base, local, [0, 1])
        np.testing.assert_array_equal(routed, [True, False, True])
        np.testing.assert_allclose(changed.sum(axis=1), 1., atol=1e-6)
        np.testing.assert_array_equal(changed[~routed], base[~routed])
        np.testing.assert_array_equal(changed[:, 2], base[:, 2])
        # A true minority sample in row 1 stays missed by the gate. Full-system
        # scoring must include it rather than scoring only routed rows.
        y = np.array([1, 1, 0])
        self.assertEqual(int(np.sum(changed.argmax(axis=1) == y)), 2)
        unchanged, _ = route_specialist(base[[1]], np.empty((0, 2)), [0, 1])
        np.testing.assert_array_equal(unchanged, base[[1]])
        with self.assertRaises(ValueError):
            route_specialist(base, local[:1], [0, 1])

    def test_log_features_keep_order_and_clip_only_zero_probabilities(self):
        a = np.array([[1., 0.], [.4, .6]])
        b = np.array([[.8, .2], [.3, .7]])
        result = log_probability_features([a, b])
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[:, :2], np.log(np.maximum(a, 1e-12)))
        np.testing.assert_allclose(result[:, 2:], np.log(b))
        with self.assertRaises(ValueError):
            log_probability_features([a, b[:1]])

    def test_solver_repair_preserves_the_multinomial_objective(self):
        rng = np.random.default_rng(140)
        ya, yb = np.tile(np.arange(3), 120), np.tile(np.arange(3), 30)
        a, b = rng.normal(size=(len(ya), 6)), rng.normal(size=(len(yb), 6))
        a[:, 0] += 1.1 * ya; b[:, 0] += 1.1 * yb
        original, first = fit_meta(a, ya, b, yb, seed=42, max_iter=1000)
        repaired, second = fit_meta(a, ya, b, yb, seed=42, solver="newton-cg", max_iter=100)
        self.assertFalse(first["hit_iteration_limit"] or second["hit_iteration_limit"])
        self.assertEqual(first["class_counts"], second["class_counts"])
        self.assertEqual(second["solver"], "newton-cg")
        np.testing.assert_allclose(original.predict_proba(b), repaired.predict_proba(b), atol=1e-3, rtol=1e-3)

    def test_scoring_fold_labels_do_not_enter_its_meta_fit(self):
        rng = np.random.default_rng(21)
        a, b = rng.normal(size=(90, 6)), rng.normal(size=(30, 6))
        ya, yb = np.tile(np.arange(3), 30), np.tile(np.arange(3), 10)
        splits = [(np.setdiff1d(np.arange(30), score), score) for score in np.split(np.arange(30), 3)]
        with patch("ddos_iot.experiments.study_stacking.StratifiedKFold.split", return_value=iter(splits)):
            first = crossfit_meta(a, ya, b, yb, seed=42, folder=self.root / "first", folds=3)
        try:
            first_values = first.copy()
        finally:
            close_memmap(first)
        changed = yb.copy(); changed[:10] = (changed[:10] + 1) % 3
        with patch("ddos_iot.experiments.study_stacking.StratifiedKFold.split", return_value=iter(splits)):
            second = crossfit_meta(a, ya, b, changed, seed=42, folder=self.root / "second", folds=3)
        try:
            np.testing.assert_array_equal(first_values[:10], second[:10])
            np.testing.assert_allclose(second.sum(axis=1), 1., atol=1e-6)
        finally:
            close_memmap(second)

    def test_specialist_trains_only_declared_true_classes(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        frame, encoder = fixture.fixture()
        config = fixture.config()
        x, y = frame.drop(columns="label"), encoder.transform(frame.label)
        group = np.array([0, 1, 5, 7])
        model, report = fit_specialist(x, y, group, config, seed=42, power=1., rounds=3)
        self.assertEqual(report["training_rows"], int(np.isin(y, group).sum()))
        self.assertEqual(model.predict_proba(x.iloc[:2]).shape, (2, 4))

    def test_group_expansion_uses_target_mistakes_and_adds_the_whole_family(self):
        classes = ["BenignTraffic", "SqlInjection", "XSS", "DNS_Spoofing", "MITM-ArpSpoofing", "DDoS-UDP_Flood"]
        y = np.array([1, 1, 1, 2, 2, 0, 5])
        predicted = np.array([0, 3, 2, 4, 2, 5, 3])
        original, before = specialist_group(classes, y, predicted, [1, 2], 0)
        expanded, after = specialist_group(classes, y, predicted, [1, 2], 0, expand=True)
        np.testing.assert_array_equal(original, [0, 1, 2])
        np.testing.assert_array_equal(expanded, [0, 1, 2, 3, 4])
        self.assertEqual(after["added_families"], ["Spoofing"])
        self.assertEqual(before["target_errors_inside_routing_group"], 2)
        self.assertEqual(after["target_errors_inside_routing_group"], 4)
        self.assertNotIn("DDoS", after["outside_group_error_counts_by_family"])

    def test_expanded_fixed_specialist_freezes_group_and_capacity_before_validation(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        frame, _ = fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        # Remove the fixture's perfect label proxy so real tuning mistakes
        # supply the expansion evidence, with no fabricated predictions.
        frame["f0"] = np.random.default_rng(711).normal(size=len(frame))
        frame.to_csv(config.source_path, index=False)
        manifest_path = fixture.root / "preparation_report.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["training_subsets"][config.source_path.name]["sha256"] = file_digest(config.source_path)
        manifest_path.write_text(json.dumps(manifest))
        run_training_study(config)
        output = fixture.root / "expanded"
        from ddos_iot.experiments.study_specialist import _score
        def after_freeze(*args):
            if args[4] == "validation":
                frozen = json.loads((output / "selection.json").read_text())
                self.assertFalse(frozen["validation_used_for_selection"])
                self.assertTrue(frozen["expanded_fixed"])
                self.assertTrue(all(v["depth"] == 10 and v["rounds"] == 500 for v in frozen["candidates"].values()))
            return _score(*args)
        with patch("ddos_iot.experiments.study_specialist._score", side_effect=after_freeze):
            report = run_specialist_study(config.output_path, output, expanded_fixed=True, minimum_error_fraction=0.)
        self.assertEqual(report["status"], "complete")
        with np.load(config.output_path / "seed_42/splits.npz") as splits:
            validation = splits["validation"].copy()
        for row in report["validation_results"]:
            bundle = joblib.load(output / row["candidate"] / "model.joblib")
            self.assertEqual(bundle.specialist.max_depth, 10)
            with np.load(output / row["candidate"] / "validation/predictions.npz") as saved:
                np.testing.assert_array_equal(bundle.encoder.transform(bundle.predict(frame.iloc[validation])), saved["predicted"])

    def test_confirmation_reuses_source_and_keeps_parameters_fixed(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        run_training_study(config)
        result = confirm_training_study(config.output_path, fixture.root / "confirmation")
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["hyperparameters_frozen_across_seeds"])
        self.assertEqual({row["seed"] for row in result["results"]}, {42, 123, 2024})
        self.assertEqual(sum(row["reused_source_result"] for row in result["results"]), 1)
        self.assertFalse(result["test_file_read"])
        self.assertEqual(result["selected_candidate"], "paper_fixed")
        selection = config.output_path / "seed_42/selection.json"
        value = json.loads(selection.read_text()); value["best_macro"] = "changed"
        selection.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "selection changed"):
            confirm_training_study(config.output_path, fixture.root / "tampered")


if __name__ == "__main__":
    unittest.main()
