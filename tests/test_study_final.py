"""Final-test access is impossible before all frozen models are persisted."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.study import run_training_study
from ddos_iot.experiments.study_config import TrainingTrial
from ddos_iot.experiments.study_confirmation import confirm_training_study
from ddos_iot.experiments.study_final import evaluate_study_test
from ddos_iot.experiments.study_specialist import run_specialist_study
from ddos_iot.experiments.study_stacking import run_stacking_study
import test_training_study as study_fixture


class FinalStudyTests(unittest.TestCase):
    def test_once_only_test_access_follows_all_model_saves(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        frame, _ = fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        run_training_study(config)
        confirmation = fixture.root / "confirmation"
        confirm_training_study(config.output_path, confirmation)
        output, tests = fixture.root / "final", fixture.root / "test_only_after_models"

        def after_all_models(archive, record, destination, classes):
            self.assertEqual(classes, 8)
            for seed in (42, 123, 2024):
                self.assertTrue((output / f"seed_{seed}/paper_fixed/model.joblib").exists())
            self.assertTrue((output / "protocol.json").exists())
            destination.mkdir()
            path = destination / "8class_test_23features.csv"
            frame.to_csv(path, index=False)
            return path

        with patch("ddos_iot.experiments.study_final.extract_frozen_test", side_effect=after_all_models):
            result = evaluate_study_test(confirmation, output, archive=fixture.root / "missing_archive",
                                         record_path=fixture.root / "missing_record", test_directory=tests)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["models_saved"]), 3)
        self.assertEqual(len(result["results"]), 3)
        self.assertTrue(result["test_csv_opened"])
        self.assertFalse(result["test_used_for_selection"])
        with self.assertRaises(FileExistsError):
            evaluate_study_test(confirmation, output, archive=fixture.root / "missing_archive",
                                record_path=fixture.root / "missing_record", test_directory=tests)
        changed = confirmation / "selection.json"
        value = json.loads(changed.read_text()); value["candidate"] = "tampered"
        changed.write_text(json.dumps(value))
        with patch("ddos_iot.experiments.study_final.extract_frozen_test", side_effect=AssertionError("Test accessed")), \
             self.assertRaisesRegex(ValueError, "unchanged multiseed"):
            evaluate_study_test(confirmation, fixture.root / "bad_final", archive=fixture.root / "missing_archive",
                                record_path=fixture.root / "missing_record", test_directory=fixture.root / "other_test")

    def test_complexity_gates_skip_models_without_training_error_evidence(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        run_training_study(config)
        # This separable fixture yields no target mistakes, so additional
        # training must be skipped before any specialist or OOF fitting.
        with patch("ddos_iot.experiments.study_specialist.fit_specialist", side_effect=AssertionError("Unjustified specialist")):
            result = run_specialist_study(config.output_path, fixture.root / "specialist")
        self.assertEqual(result["status"], "skipped")
        with patch("ddos_iot.experiments.study_stacking._base_oof", side_effect=AssertionError("Unjustified stack")):
            result = run_stacking_study(config.output_path, fixture.root / "stack")
        self.assertEqual(result["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
