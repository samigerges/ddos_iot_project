"""Published-input integrity, disjoint selection, and real model persistence."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.attack_families import FAMILY_NAMES, RAW_TO_FAMILY
from ddos_iot.experiments.storage import file_digest
from ddos_iot.experiments.study import error_diversity, run_training_study
from ddos_iot.experiments.study_config import TrainingStudyConfig, TrainingTrial
from ddos_iot.experiments.study_data import load_study_data, protected_ids, study_split
from ddos_iot.experiments.study_training import fit_trial
from ddos_iot.multiclass import load_multiclass_bundle


class TrainingStudyTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, classes=8):
        labels = FAMILY_NAMES if classes == 8 else tuple(RAW_TO_FAMILY)
        labels = np.repeat(labels, 30)
        encoder = LabelEncoder().fit(labels)
        y = encoder.transform(labels)
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.normal(size=(len(y), 23)).astype("float32"), columns=[f"f{i}" for i in range(23)])
        frame["f0"] = y
        features = list(frame)
        frame["label"] = labels
        path = self.root / f"{classes}class_train_23features.csv"
        frame.to_csv(path, index=False)
        entry = {"sha256": file_digest(path), "features": features, "rows": len(y),
                 "representation": "Synthetic published-format test fixture", "cv_warning": "Upstream selected representation"}
        (self.root / "preparation_report.json").write_text(json.dumps({"status": "complete", "test_csv_parsed": False,
                                                                       "training_subsets": {path.name: entry}}))
        return frame, encoder

    def config(self, **values):
        return TrainingStudyConfig(subset_directory=self.root, output_path=self.root / "run", device="cpu", catboost=False,
                                   full_reference_fit=False, weight_powers=(), latency_repeats=1, **values)

    def test_source_integrity_schema_and_split_coverage_both_tasks(self):
        for classes in (8, 34):
            frame, _ = self.fixture(classes)
            config = self.config(num_classes=classes)
            data = load_study_data(config)
            fitting, tuning, validation = study_split(data.y, config, 42)
            self.assertFalse(set(fitting) & set(tuning) or set(fitting) & set(validation) or set(tuning) & set(validation))
            np.testing.assert_array_equal(np.sort(np.concatenate([fitting, tuning, validation])), np.arange(len(data.y)))
            self.assertEqual(len(protected_ids(data.encoder.classes_, config)), 3 if classes == 8 else 12)
            with config.source_path.open("a") as stream: stream.write("\n")
            with self.assertRaisesRegex(ValueError, "verified manifest"):
                load_study_data(config)

    def test_staged_fit_freezes_selection_before_validation_and_reuses_bundle(self):
        frame, _ = self.fixture()
        config = self.config(capacity=(TrainingTrial("paper_fixed"), TrainingTrial("small", depth=3, rounds=4)))
        from ddos_iot.experiments.study import _cpu_timing
        observed = []
        def after_freeze(bundle, x, cfg):
            selection = json.loads((cfg.output_path / "seed_42/selection.json").read_text())
            self.assertFalse(selection["validation_used_for_selection"])
            observed.append(len(x))
            return _cpu_timing(bundle, x, cfg)
        with patch("ddos_iot.experiments.study._cpu_timing", side_effect=after_freeze):
            result = run_training_study(config)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(observed)
        self.assertFalse(result["test_file_read"])
        with np.load(config.output_path / "seed_42/splits.npz") as split:
            validation = split["validation"]
        for row in result["validation_results"]:
            folder = config.output_path / "seed_42/finalists" / row["candidate"]
            bundle = load_multiclass_bundle(folder / "model.joblib")
            with np.load(folder / "validation/predictions.npz") as saved:
                np.testing.assert_array_equal(bundle.encoder.transform(bundle.predict(frame.iloc[validation])), saved["predicted"])
        with self.assertRaises(FileExistsError):
            run_training_study(config)

    def test_early_stopping_34_columns_and_identity_preprocessing(self):
        frame, encoder = self.fixture(34)
        config = self.config(num_classes=34)
        x, y = frame.drop(columns="label"), encoder.transform(frame.label)
        trial = TrainingTrial("early", depth=3, rounds=4, early_stopping=True)
        bundle, report = fit_trial(x, y, encoder, trial, config, 42, tuning=(x, y))
        self.assertEqual(bundle.predict_proba(x.iloc[:9]).shape, (9, 34))
        np.testing.assert_array_equal(bundle.features.transform(x).to_numpy(), x.to_numpy())
        self.assertGreaterEqual(report["rounds"], 1)
        with self.assertRaises(ValueError):
            fit_trial(x, y, encoder, trial, config, 42)

    def test_diversity_tracks_rescues_and_harm_including_gate_errors(self):
        result = error_diversity(np.array([0, 1, 1, 2]), np.array([0, 0, 1, 0]), np.array([1, 1, 0, 2]), (1, 2), 0)
        self.assertEqual(result["target_rescues"], 2)
        self.assertEqual(result["target_spoiled"], 1)
        self.assertEqual(result["target_to_benign_rescues"], 2)
        self.assertTrue(result["complementarity_present"])

    def test_rejects_changed_reference_and_invalid_trials(self):
        with self.assertRaises(ValueError):
            self.config(capacity=(TrainingTrial("paper_fixed", depth=6),))
        for kwargs in ({"name": "../escape"}, {"name": "x", "depth": 0}, {"name": "x", "weight_power": float("nan")}):
            with self.assertRaises(ValueError):
                TrainingTrial(**kwargs)


if __name__ == "__main__":
    unittest.main()
