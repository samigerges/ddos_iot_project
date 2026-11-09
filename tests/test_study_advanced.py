"""Frozen advanced decisions survive confirmation and precede final test access."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.artifacts import save_joblib_atomic
from ddos_iot.experiments.storage import object_digest, write_report
from ddos_iot.experiments.study import run_training_study
from ddos_iot.experiments.study_advanced_confirmation import confirm_advanced_study
from ddos_iot.experiments.study_advanced_final import evaluate_advanced_test
from ddos_iot.experiments.study_config import TrainingTrial
from ddos_iot.experiments.study_confirmation import confirm_training_study
from ddos_iot.multiclass import load_multiclass_bundle
import test_training_study as study_fixture


class AdvancedStudyTests(unittest.TestCase):
    def test_fixed_bias_confirmation_and_all_final_models_before_test_access(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        frame, _ = fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        source_result = run_training_study(config)
        confirmation = fixture.root / "confirmation"
        base_result = confirm_training_study(config.output_path, confirmation)
        # This fixture is perfectly separable. Lower only the synthetic reference
        # summary to exercise promotion; the saved predictions/recall guards and
        # all subsequent model fitting and evaluation remain real.
        for report, path, key in ((source_result, config.output_path / "report.json", "validation_results"),
                                  (base_result, confirmation / "report.json", "results")):
            for row in report[key]: row["macro_f1"] = .8
            write_report(path, report)
        advanced = fixture.root / "bias"
        name = "xgb_oof_protected_bias"
        vector = np.zeros(8)
        frozen = {"base_candidate": "paper_fixed", "single_model_conditions": {name: vector.tolist()},
                  "guarded_macro": name, "validation_used_for_tuning": False}
        write_report(advanced / "selection.json", frozen)
        bundle = load_multiclass_bundle(config.output_path / "seed_42/finalists/paper_fixed/model.joblib")
        bundle.decision_bias = vector
        save_joblib_atomic(advanced / name / "model.joblib", bundle)
        observed = {**source_result["validation_results"][0], "candidate": name, "macro_f1": 1.,
                    "reference_constraints_pass": True}
        write_report(advanced / "report.json", {"status": "complete", "selection_sha256": object_digest(frozen),
                                                "validation_results": [observed], "seconds": 0.})
        promoted = fixture.root / "advanced_confirmation"
        # Bias confirmation must reuse fixed base models, with no new learner or
        # decision search. It still scores the other two real validation splits.
        with patch("ddos_iot.experiments.study_advanced_confirmation.refit_trial", side_effect=AssertionError("Unexpected refit")):
            result = confirm_advanced_study(confirmation, advanced, promoted)
        self.assertTrue(result["advanced_confirmed"])
        self.assertEqual({r["seed"] for r in result["results"]}, {42, 123, 2024})
        for row in result["results"]:
            saved = load_multiclass_bundle(Path(row["artifact_folder"]) / "model.joblib")
            np.testing.assert_array_equal(saved.decision_bias, vector)
        output, test_directory = fixture.root / "final", fixture.root / "test"

        def after_all_models(archive, record, destination, classes):
            self.assertEqual(classes, 8)
            self.assertTrue((output / "protocol.json").exists())
            for seed in (42, 123, 2024):
                for candidate in ("paper_fixed", name):
                    folder = output / f"seed_{seed}" / candidate
                    self.assertTrue((folder / "model.joblib").exists())
                    self.assertEqual(json.loads((folder / "training.json").read_text())["training_rows"], len(frame))
                final = load_multiclass_bundle(output / f"seed_{seed}" / name / "model.joblib")
                np.testing.assert_array_equal(final.decision_bias, vector)
                reference = load_multiclass_bundle(output / f"seed_{seed}/paper_fixed/model.joblib")
                self.assertIsNone(reference.decision_bias)
            destination.mkdir()
            path = destination / "8class_test_23features.csv"
            frame.to_csv(path, index=False)
            return path

        with patch("ddos_iot.experiments.study_advanced_final.extract_frozen_test", side_effect=after_all_models):
            final = evaluate_advanced_test(promoted, output, archive=fixture.root / "unused_archive",
                record_path=fixture.root / "unused_record", test_directory=test_directory)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(len(final["results"]), 6)
        self.assertFalse(final["test_used_for_selection"])
        with self.assertRaises(FileExistsError):
            evaluate_advanced_test(promoted, output, archive=fixture.root / "unused_archive",
                record_path=fixture.root / "unused_record", test_directory=test_directory)
        # A comparison remains useful when a candidate violates a provisional
        # guard, but that evidence must never silently promote the candidate.
        observed["reference_constraints_pass"] = False
        write_report(advanced / "report.json", {"status": "complete", "selection_sha256": object_digest(frozen),
                                                "validation_results": [observed], "seconds": 0.})
        compared = confirm_advanced_study(confirmation, advanced, fixture.root / "comparison",
                                          candidate_override=name, compare_only=True)
        self.assertEqual(len(compared["results"]), 3)
        self.assertTrue(compared["comparison_only"])
        self.assertFalse(compared["advanced_confirmed"])
        self.assertEqual(compared["selected_candidate"], "paper_fixed")


if __name__ == "__main__":
    unittest.main()
