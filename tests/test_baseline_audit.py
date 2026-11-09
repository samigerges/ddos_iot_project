"""Published settings are inspected as syntax, never executed as project code."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.experiments.baseline_audit import inspect_author_script


class BaselineAuditTests(unittest.TestCase):
    def test_inspects_literals_and_dynamic_expressions_without_execution(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            path = Path(folder) / "author.py"
            path.write_text('''raise RuntimeError("This file must never execute")
def parse_arguments():
    parser.add_argument('--target_rows', default=5000000)
def run_ablation():
    xgb_params_all = {"max_depth": 14, "random_state": random_state}
    xgb_params_selected = {"max_depth": 10, "n_estimators": 500}
def perform_feature_selection():
    fs_params = {"objective": "multiclass", "num_class": n_classes}
''', encoding="utf-8")
            result = inspect_author_script(path)
            self.assertFalse(result["executed"])
            self.assertEqual(result["argument_defaults"]["--target_rows"], 5000000)
            self.assertEqual(result["dictionaries"]["xgb_params_all"]["values"]["max_depth"], 14)
            self.assertEqual(result["dictionaries"]["xgb_params_selected"]["values"]["max_depth"], 10)
            self.assertEqual(result["dictionaries"]["fs_params"]["values"]["num_class"],
                             {"source_expression": "n_classes"})
            self.assertEqual(len(result["sha256"]), 64)

    def test_rejects_unrecognized_source_instead_of_inventing_defaults(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            path = Path(folder) / "author.py"
            path.write_text("x = 1", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings are missing"):
                inspect_author_script(path)


if __name__ == "__main__":
    unittest.main()
