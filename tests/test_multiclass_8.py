"""Eight-family mapping, paper sampling, isolation, and real model regressions."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiclass import parse_args
from ddos_iot.attack_families import (
    FAMILY_MEMBERS, FAMILY_NAMES, RAW_TO_FAMILY, map_family_labels,
    prepare_paper_split, proportional_quotas, scan_source_pool, validate_source_paths,
)
from ddos_iot.multiclass import (
    MODEL_NAMES, build_model, class_weights, encode_labels, evaluate, fit_features,
    grid_search, load_raw_split, run_multiclass_training,
)


def family_fixture(rows_per_raw: int = 24) -> pd.DataFrame:
    labels = np.repeat(list(RAW_TO_FAMILY), rows_per_raw)
    family_ids = np.array([FAMILY_NAMES.index(RAW_TO_FAMILY[label]) for label in labels])
    rng = np.random.default_rng(29)
    data = pd.DataFrame(rng.normal(size=(len(labels), 46)).astype("float32"),
                        columns=[f"f{i}" for i in range(46)])
    data["f0"] = family_ids.astype("float32")
    for k in range(8):
        data[f"f{k + 1}"] = (family_ids == k).astype("float32")
    data["f45"] = np.arange(len(labels), dtype="float32")
    data["label"] = labels
    return data


class EightClassTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="eight_class_test_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"

    def args(self, *extra):
        return parse_args(["--train-input-path", str(self.source),
                           "--test-input-path", str(self.root / "must_not_read.csv"),
                           "--output-path", str(self.root / "results"),
                           "--sampling-target", "544", "--min-samples-per-class", "16",
                           "--n-estimators", "3", "--n-jobs", "2", "--log-period", "0",
                           "--no-plots", *extra], default_num_classes=8)

    def test_exact_paper_family_names_and_membership(self):
        self.assertEqual(FAMILY_NAMES, ("Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "WebBased"))
        self.assertEqual(len(RAW_TO_FAMILY), 34)
        self.assertEqual(sum(map(len, FAMILY_MEMBERS.values())), 34)
        self.assertEqual(Counter(RAW_TO_FAMILY.values()), {
            "Benign": 1, "BruteForce": 1, "DDoS": 12, "DoS": 4,
            "Mirai": 3, "Recon": 5, "Spoofing": 2, "WebBased": 6})
        labels = pd.Series([" Backdoor_Malware ", "VulnerabilityScan", "BenignTraffic", "DictionaryBruteForce"])
        self.assertEqual(map_family_labels(labels).tolist(), ["WebBased", "Recon", "Benign", "BruteForce"])
        with self.assertRaisesRegex(ValueError, "Unknown CICIoT2023"):
            map_family_labels(pd.Series(["UnknownAttack", "DDoS-typo"]))
        self.assertTrue(map_family_labels(pd.Series([None, " "])).isna().all())

    def test_eight_defaults_and_34_defaults_remain_distinct(self):
        eight = parse_args([], default_num_classes=8)
        raw = parse_args([])
        self.assertEqual((eight.num_classes, eight.data_protocol, eight.cv_preprocessing), (8, "paper", "train"))
        self.assertEqual(eight.source_input_path, [Path("data/raw/train/train.csv")])
        self.assertEqual(eight.output_path, Path("models/multiclass_8_tuned"))
        self.assertEqual((eight.sampling_target, eight.min_samples_per_class, eight.test_size), (5_000_000, 1000, .2))
        self.assertEqual(eight.cv_folds, 5)
        self.assertEqual(eight.feature_selection, "median")
        self.assertEqual((raw.num_classes, raw.data_protocol, raw.cv_preprocessing), (34, "existing", "fold"))
        self.assertEqual(raw.output_path, Path("models/multiclass_34_tuned"))

    def test_quota_equation_and_impossible_minimum_do_not_oversample(self):
        counts = dict.fromkeys(FAMILY_NAMES, 100)
        counts["DDoS"] = 1000
        quotas = proportional_quotas(counts, 800, 50)
        self.assertEqual(quotas, {k: max(50, n * 800 // 1700) for k, n in counts.items()})
        self.assertNotEqual(sum(quotas.values()), 800)  # Preserve the published equation.
        with self.assertRaisesRegex(ValueError, "Insufficient real samples"):
            proportional_quotas({**counts, "BruteForce": 10}, 800, 50)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            proportional_quotas(counts, 2000, 50)
        with self.assertRaisesRegex(ValueError, "all eight"):
            proportional_quotas({k: n for k, n in counts.items() if k != "Benign"}, 800, 50)

    def test_paper_sampling_is_reproducible_clean_and_disjoint(self):
        data = family_fixture()
        data.loc[0, "f0"] = np.inf
        data.loc[1, "label"] = None
        data.to_csv(self.source, index=False)
        call_args = ([self.source], "label", 79, 42, 544, 16, .2)
        train, test, report = prepare_paper_split(*call_args)
        again, again_test, again_report = prepare_paper_split(*call_args)
        pd.testing.assert_frame_equal(train, again)
        pd.testing.assert_frame_equal(test, again_test)
        self.assertEqual(report, again_report)
        self.assertEqual(report["removed_invalid_rows"], 2)
        self.assertFalse(set(train.index) & set(test.index))
        self.assertFalse(set(train.f45) & set(test.f45))
        counts = pd.concat([train, test]).label.value_counts().to_dict()
        self.assertEqual(counts, report["quotas"])
        self.assertEqual(set(train.label), set(FAMILY_NAMES))
        self.assertEqual(set(test.label), set(FAMILY_NAMES))
        self.assertEqual(len(test), int(np.ceil((len(train) + len(test)) * .2)))
        self.assertFalse(report["sampling_with_replacement"])

    def test_duplicate_sources_and_mismatched_schema_rejected(self):
        family_fixture().to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate source paths"):
            validate_source_paths([self.source, self.source], "label")
        other = self.root / "wrong.csv"
        family_fixture().drop(columns="f1").to_csv(other, index=False)
        with self.assertRaisesRegex(ValueError, "46 numeric features"):
            validate_source_paths([self.source, other], "label")

    def test_existing_split_cap_applies_to_families_not_subtypes(self):
        family_fixture().to_csv(self.source, index=False)
        data, report = load_raw_split(self.source, "label", 10, 51, 42, num_classes=8)
        self.assertEqual(len(data), 80)
        self.assertEqual(report["selected_counts"], dict.fromkeys(FAMILY_NAMES, 10))
        self.assertEqual(report["raw_counts"]["DDoS"], 12 * 24)

    def test_eight_metrics_weights_and_estimator_objectives(self):
        y = np.repeat(np.arange(8), np.arange(10, 18))
        weights = class_weights(y, "balanced", 8)
        np.testing.assert_allclose(np.bincount(y, weights=weights[y]), len(y) / 8)
        self.assertEqual(evaluate(y, y, 8)["macro_f1"], 1.)
        self.assertEqual(evaluate(y, y, 8)["balanced_accuracy"], 1.)
        for name in MODEL_NAMES:
            key = "classes_count" if name == "catboost" else "num_class"
            self.assertEqual(build_model(name, self.args(), 3, False).get_params()[key], 8)
        with self.assertRaisesRegex(ValueError, "paper's exact family names"):
            encode_labels(pd.Series(list("abcdefgh")), pd.Series(list("abcdefgh")), 8)

    def test_counts_only_uses_source_and_never_trains_or_reads_existing_test(self):
        family_fixture().to_csv(self.source, index=False)
        with patch("ddos_iot.multiclass.fit_features") as fit:
            result = run_multiclass_training(self.args("--counts-only"))
        fit.assert_not_called()
        self.assertEqual(set(result["valid_family_counts"]), set(FAMILY_NAMES))
        self.assertTrue((self.root / "results" / "raw_to_family_mapping.json").exists())

    def test_eight_fold_local_option_excludes_scoring_rows(self):
        from sklearn.model_selection import StratifiedKFold

        data = family_fixture()
        labels = map_family_labels(data.label)
        _, y, _ = encode_labels(labels, labels, 8)
        x = data.drop(columns="label")
        args = self.args("--cv-preprocessing", "fold", "--no-refit-full")
        with patch("ddos_iot.multiclass.fit_features", wraps=fit_features) as spy:
            report = grid_search("lightgbm", x, y, args, {"n_estimators": [3]}, self.root / "fold_search")
        self.assertEqual(spy.call_count, 5)
        self.assertEqual(report["cv_preprocessing"], "fold")
        folds = StratifiedKFold(5, shuffle=True, random_state=args.random_state)
        for call, (train, score) in zip(spy.call_args_list, folds.split(x, y)):
            self.assertTrue(set(call.args[0].index) <= set(train))
            self.assertFalse(set(call.args[0].index) & set(score))

    def test_all_three_models_paper_cv_saved_artifacts_and_untouched_test(self):
        source_data = family_fixture()
        source_data.to_csv(self.source, index=False)
        with patch("ddos_iot.multiclass.fit_features", wraps=fit_features) as spy:
            result = run_multiclass_training(self.args())
        self.assertEqual(spy.call_count, 1)  # Paper preprocessing once, reused across models/folds.
        output = self.root / "results"
        data_report = json.loads((output / "data_report.json").read_text())
        self.assertEqual(len(spy.call_args.args[0]), data_report["train"]["selected_rows"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["class_count"], 8)
        self.assertEqual(set(json.loads((output / "label_mapping.json").read_text()).values()), set(FAMILY_NAMES))
        self.assertFalse((self.root / "must_not_read.csv").exists())
        train, test, _ = prepare_paper_split([self.source], "label", 250_000, 42, 544, 16, .2)
        self.assertTrue(set(spy.call_args.args[0].f45) <= set(train.f45))
        self.assertFalse(set(spy.call_args.args[0].f45) & set(test.f45))
        for row in result["results"]:
            folder = output / row["model"]
            bundle = joblib.load(folder / "model.joblib")
            self.assertEqual(set(bundle.encoder.classes_), set(FAMILY_NAMES))
            self.assertTrue(set(bundle.predict(source_data.drop(columns="label"))) <= set(FAMILY_NAMES))
            saved_predictions = pd.read_csv(folder / "test_predictions.csv")
            np.testing.assert_array_equal(bundle.predict(test.drop(columns="label")), saved_predictions.predicted_label)
            matrix = pd.read_csv(folder / "confusion_matrix.csv", index_col=0)
            self.assertEqual(matrix.shape, (8, 8))
            self.assertEqual(matrix.to_numpy().sum(), row["test_rows"])
            report = pd.read_csv(folder / "classification_report_test.csv", index_col=0)
            self.assertAlmostEqual(report.loc["macro avg", "f1-score"], row["macro_f1"])
            search = json.loads((folder / "grid_search_report.json").read_text())
            self.assertEqual(search["class_count"], 8)
            self.assertEqual(search["cv_preprocessing"], "train")
            self.assertEqual(len(search["candidates"]), 4)
            self.assertTrue(all(len(c["fold_scores"]) == 5 for c in search["candidates"]))


if __name__ == "__main__":
    unittest.main()
