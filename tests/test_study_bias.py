"""Single-model OOF decisions, shared-cache identity, and stale-cache rejection."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.study import run_training_study
from ddos_iot.experiments.study_bias import run_single_bias_study
from ddos_iot.experiments.study_config import TrainingTrial
from ddos_iot.experiments.study_data import load_study_data
from ddos_iot.experiments.study_stacking import ensure_base_oof
import test_training_study as study_fixture


class StudyBiasTests(unittest.TestCase):
    def test_oof_bias_freezes_before_validation_and_shares_verified_cache(self):
        fixture = study_fixture.TrainingStudyTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.fixture()
        config = fixture.config(capacity=(TrainingTrial("paper_fixed"),))
        run_training_study(config)
        output = fixture.root / "bias"
        from ddos_iot.experiments.study_bias import trial_probabilities

        def after_freeze(*args):
            self.assertTrue((output / "selection.json").exists())
            selection = json.loads((output / "selection.json").read_text())
            self.assertFalse(selection["validation_used_for_tuning"])
            return trial_probabilities(*args)

        with patch("ddos_iot.experiments.study_bias.trial_probabilities", side_effect=after_freeze):
            report = run_single_bias_study(config.output_path, output)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["test_file_read"])
        selection = json.loads((config.output_path / "seed_42/selection.json").read_text())
        with np.load(config.output_path / "seed_42/splits.npz") as split:
            fitting = split["fitting"].copy()
        data = load_study_data(config)
        with patch("ddos_iot.experiments.study_stacking._base_oof", side_effect=AssertionError("Duplicate OOF fitting")):
            ensure_base_oof(data, fitting, selection["trials"]["paper_fixed"], config, 42, fixture.root / "second_cache_view")
        # Even a finite, normalized replacement array must fail its fingerprint.
        cached = config.output_path / "oof_cache/seed_42/paper_fixed/probabilities.npy"
        value = np.load(cached)
        value[:] = 1 / value.shape[1]
        np.save(cached, value)
        with self.assertRaisesRegex(ValueError, "modified"):
            ensure_base_oof(data, fitting, selection["trials"]["paper_fixed"], config, 42, fixture.root / "corrupted_view")


if __name__ == "__main__":
    unittest.main()
