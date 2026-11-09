"""Paired uncertainty preserves identity and a known fixed-class improvement."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddos_iot.experiments.study_uncertainty import paired_bootstrap


class StudyUncertaintyTests(unittest.TestCase):
    def test_identical_predictions_have_zero_paired_delta_in_every_draw(self):
        y = np.repeat(np.arange(3), 30)
        prediction = y.copy(); prediction[:5] = 1
        result = paired_bootstrap(y, prediction, prediction, ["a", "b", "c"], repetitions=200)
        self.assertEqual(result["macro_f1"]["delta_ci95"], [0., 0.])
        self.assertTrue(all(row["delta"] == 0. for row in result["per_class"]))

    def test_recovered_errors_produce_a_positive_interval_and_exact_point_metric(self):
        y = np.repeat(np.arange(3), 100)
        reference = y.copy(); reference[:50] = 1
        result = paired_bootstrap(y, reference, y, ["a", "b", "c"], repetitions=400)
        self.assertAlmostEqual(result["macro_f1"]["reference"], (2/3 + .8 + 1) / 3)
        self.assertEqual(result["macro_f1"]["candidate"], 1.)
        self.assertGreater(result["macro_f1"]["delta_ci95"][0], 0.)
        self.assertEqual([r["support"] for r in result["per_class"]], [100, 100, 100])
        with self.assertRaises(ValueError):
            paired_bootstrap(y, reference[:-1], y, ["a", "b", "c"])

    def test_compact_34_class_integer_ids_do_not_overflow_joint_indices(self):
        y = np.repeat(np.arange(34, dtype=np.uint8), 3)
        result = paired_bootstrap(y, y, y, [str(i) for i in range(34)], repetitions=100)
        self.assertEqual(result["macro_f1"]["reference"], 1.)
        self.assertEqual(result["macro_f1"]["delta_ci95"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
