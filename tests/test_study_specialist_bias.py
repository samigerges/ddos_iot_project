"""Routed OOF calibration excludes scoring rows and reuses confirmed hierarchies."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.storage import file_digest
from ddos_iot.experiments.study import run_training_study
from ddos_iot.experiments.study_advanced_confirmation import confirm_advanced_study
from ddos_iot.experiments.study_config import TrainingTrial
from ddos_iot.experiments.study_confirmation import confirm_training_study
from ddos_iot.experiments.study_specialist import fit_specialist, run_specialist_study, _predict_routed
from ddos_iot.experiments.study_specialist_bias import calibrate_specialist, confirm_specialist_decisions
import test_training_study as study_fixture


class SpecialistBiasTests(unittest.TestCase):
    def test_frozen_routed_oof_and_shared_confirmation_do_not_refit_raw_models(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        frame, _ = fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        frame["f0"] = np.random.default_rng(711).normal(size=len(frame))
        frame.to_csv(config.source_path, index=False)
        manifest_path = fixture.root / "preparation_report.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["training_subsets"][config.source_path.name]["sha256"] = file_digest(config.source_path)
        manifest_path.write_text(json.dumps(manifest))
        run_training_study(config)
        raw = fixture.root / "specialist"
        run_specialist_study(config.output_path, raw, expanded_fixed=True, minimum_error_fraction=0.)
        raw_choice = json.loads((raw / "selection.json").read_text())["best_macro"]
        base_confirmation, raw_confirmation = fixture.root / "base_confirmation", fixture.root / "raw_confirmation"
        confirm_training_study(config.output_path, base_confirmation)
        confirm_advanced_study(base_confirmation, raw, raw_confirmation, candidate_override=raw_choice, compare_only=True)
        cached_path = raw_confirmation / "seed_123" / raw_choice / "model.joblib"
        digest = file_digest(cached_path)
        output = fixture.root / "calibration"
        training_ids, scored = set(), set()

        def fitting(x, *args, **kwargs):
            training_ids.clear(); training_ids.update(x.index)
            return fit_specialist(x, *args, **kwargs)

        def scoring(p, model, x, *args):
            self.assertFalse(training_ids & set(x.index))
            self.assertFalse(scored & set(x.index))
            scored.update(x.index)
            return _predict_routed(p, model, x, *args)

        from ddos_iot.experiments.study_specialist_bias import _score
        def validation_after_freeze(*args):
            self.assertTrue((output / "selection.json").exists())
            self.assertFalse(json.loads((output / "selection.json").read_text())["validation_used_for_selection"])
            return _score(*args)

        with patch("ddos_iot.experiments.study_specialist_bias.fit_specialist", side_effect=fitting), \
             patch("ddos_iot.experiments.study_specialist_bias._predict_routed", side_effect=scoring), \
             patch("ddos_iot.experiments.study_specialist_bias._score", side_effect=validation_after_freeze):
            report = calibrate_specialist(config.output_path, raw, output, raw_confirmation=raw_confirmation)
        self.assertEqual(report["status"], "complete")
        with np.load(config.output_path / "seed_42/splits.npz") as split:
            self.assertEqual(scored, set(split["fitting"]))
        confirmed = fixture.root / "decision_confirmation"
        with patch("ddos_iot.experiments.study_specialist_bias.fit_specialist", side_effect=AssertionError("Unexpected refit")), \
             patch("ddos_iot.experiments.study_advanced_confirmation.fit_specialist", side_effect=AssertionError("Unexpected refit")):
            result = confirm_specialist_decisions(base_confirmation, raw_confirmation, output, confirmed)
            generic = confirm_advanced_study(base_confirmation, output, fixture.root / "generic_confirmation",
                candidate_override="specialist_oof_bias", compare_only=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(generic["status"], "complete")
        frozen = json.loads((output / "selection.json").read_text())
        self.assertEqual(len(result["validation_results"]), 3 * len(frozen["candidates"]))
        for row in result["validation_results"]:
            bundle = joblib.load(Path(row["artifact_folder"]) / "model.joblib")
            np.testing.assert_array_equal(bundle.decision_bias, frozen["candidates"][row["candidate"]]["decision_bias"])
        self.assertEqual(file_digest(cached_path), digest)
        self.assertFalse(result["test_file_read"])


if __name__ == "__main__":
    unittest.main()
