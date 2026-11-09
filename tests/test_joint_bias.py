"""Joint moves solve feasible improvements blocked by single-coordinate limits."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.decisions import predict_with_bias
from ddos_iot.experiments.joint_bias import tune_joint_biases
from ddos_iot.experiments.recall_constraints import reference_constraints, tune_constrained_biases


class JointBiasTests(unittest.TestCase):
    def test_escapes_recall_fpr_coordinate_trap_without_changing_inference(self):
        y = np.array([0] * 10 + [1] * 10 + [2] * 100)
        p = np.asarray([[.35, .1, .55]] * 3 + [[.6, .3, .1]] * 3 + [[.99, .005, .005]] * 4 +
                       [[.4, .5, .1]] * 5 + [[.6, .3, .1]] * 5 + [[.01, .01, .98]] * 100, dtype="float32")
        constraints = reference_constraints(y, p.argmax(axis=1), 3, benign_id=0, protected_ids=(1,))
        coordinate, initial = tune_constrained_biases(y, p, constraints=constraints, grid=(0., 1., 2.))
        joint, report = tune_joint_biases(y, p, constraints, coordinate, grid=(0., 1., 2.))
        self.assertTrue(report["tuned"]["feasible"])
        self.assertGreater(report["tuned"]["macro_f1"], initial["tuned"]["macro_f1"] + .1)
        ids = predict_with_bias(p, joint)
        self.assertGreaterEqual(float(np.mean(ids[y == 1] == 1)), constraints.minimum_recall[1])
        self.assertLessEqual(float(np.mean(ids[y == 0] != 0)), constraints.max_benign_fpr + 1e-12)
        self.assertFalse(report["test_used_for_tuning"])
        self.assertFalse(report["validation_used_for_tuning"])

    def test_two_class_competition_handles_empty_outside_group(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
        constraints = reference_constraints(y, p.argmax(axis=1), 2, benign_id=0, protected_ids=(1,))
        bias, report = tune_joint_biases(y, p, constraints, np.zeros(2), grid=(-1., 0., 1.))
        np.testing.assert_array_equal(predict_with_bias(p, bias), y)
        self.assertTrue(report["tuned"]["feasible"])


if __name__ == "__main__":
    unittest.main()
