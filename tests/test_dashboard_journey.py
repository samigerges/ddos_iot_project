"""Journey navigation and provenance stay safe with complete, missing or partial reports."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddos_iot.journey import load_binary_journey, load_journey
from ddos_iot import dashboard_journey


class JourneyFixture(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        previous = tempfile.tempdir
        tempfile.tempdir = str(ROOT / "tmp")
        self.addCleanup(setattr, tempfile, "tempdir", previous)
        temporary = tempfile.TemporaryDirectory(prefix="journey_test_", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def fixture(self, selected=.84):
        base = "outputs/studies"
        self.write(f"{base}/final_protocol_v1/report.json", {"status": "complete"})
        self.write(f"{base}/final_results_v1/report.json", {"status": "complete",
            "selected_34_test_recall_budget_pass": False,
            "macro_f1_uncertainty": {"reference": .8, "candidate": selected, "delta_ci95": [.02, .06]}})
        rows = [{"classes": k, "candidate": name, "seeds": 3, "macro_f1_mean": score,
                 "weighted_f1_mean": .99, "benign_false_positive_rate_mean": .03}
                for k, name, score in ((8, "paper_fixed", .81), (34, "paper_fixed", .8), (34, "specialist", selected))]
        pd.DataFrame(rows).to_csv(self.root / f"{base}/final_results_v1/test_summary.csv", index=False)
        pd.DataFrame([{"classes": 8, "macro_f1": .82, "gpu_macro_f1": .82},
                      {"classes": 34, "macro_f1": .81, "gpu_macro_f1": .81}]).to_csv(
            self.root / f"{base}/final_results_v1/pinned_paper_references.csv", index=False)
        pd.DataFrame([{"classes": 34, "class": "DictionaryBruteForce", "support": 200,
            "reference_f1_mean": .6, "selected_f1_mean": .65, "selected_recall_mean": .52,
            "recall_delta_pp": -2.}]).to_csv(self.root / f"{base}/final_results_v1/minority_test_metrics.csv", index=False)
        self.write(f"{base}/34class_v1/recall_policy_confirmation_v1/selection.json",
            {"recall_policy": {"scope": "mean", "maximum_recall_loss": .005}})
        for k in (8, 34):
            self.write(f"{base}/{k}class_v1/confirmation_v1/report.json", {"status": "complete", "results": [
                {"seed": seed, "candidate": "paper_fixed", "split": "validation", "macro_f1": score}
                for seed, score in ((42, .78), (123, .8), (2024, .82))]})
        self.write(f"{base}/34class_v1/selected_paired_timing_v1/report.json", {"status": "complete", "results": [
            {"candidate": name, "inference_us_median": cost, "inference_us_std": .1}
            for name, cost in (("paper_fixed", 10.), ("specialist", 11.))]})
        path = self.root / "docs/MULTICLASS_FINAL_RESULTS.md"
        path.parent.mkdir()
        path.write_text("# Fixture final results\n", encoding="utf-8")


class JourneyDataTests(JourneyFixture):
    def test_binary_comparison_preserves_order_and_excludes_failed_or_invalid_scores(self):
        self.write("models/requested_model_comparison/model_comparison_report.json", {
            "best_model": {"model": "lightgbm", "status": "success", "f1": .93},
            "models": [
                {"model": "random_forest", "status": "success", "f1": .8},
                {"model": "lightgbm", "status": "success", "f1": .93},
                {"model": "naive_bayes", "status": "success", "f1": .6},
                {"model": "xgboost", "status": "failed", "f1": .99},
                {"model": "hist_gradient_boosting", "status": "success", "f1": float("nan")},
            ]})
        with patch("joblib.load", side_effect=AssertionError("No model loads")), \
             patch("pandas.read_csv", side_effect=AssertionError("No dataset reads")):
            data = load_binary_journey(self.root)
        self.assertEqual(data.comparisons.model.tolist(), ["random_forest", "lightgbm", "naive_bayes"])
        self.assertEqual(data.comparisons.f1.tolist(), [.8, .93, .6])
        self.assertEqual(data.comparisons.selected.tolist(), [False, True, False])
        self.assertTrue(any("invalid F1" in issue for issue in data.issues))

    def test_duplicate_binary_models_hide_ambiguous_comparison(self):
        self.write("models/requested_model_comparison/model_comparison_report.json", {"models": [
            {"model": "lightgbm", "status": "success", "f1": .8},
            {"model": "lightgbm", "status": "success", "f1": .9}]})
        data = load_binary_journey(self.root)
        self.assertTrue(data.comparisons.empty)
        self.assertTrue(any("duplicate" in issue for issue in data.issues))

    def test_multiclass_model_counts_require_valid_saved_results_for_the_task(self):
        self.write("models/multiclass_8_tuned/model_comparison_report.json", {"class_count": 8, "results": [
            {"model": "xgboost", "macro_f1": .8}, {"model": "catboost", "macro_f1": .7},
            {"model": "xgboost", "macro_f1": .9}, {"model": "lightgbm", "macro_f1": True}]})
        self.write("models/multiclass_34/model_comparison_report.json", {"class_count": 8,
            "results": [{"model": "xgboost", "macro_f1": .8}]})
        data = load_journey(self.root)
        self.assertEqual(data.model_families[8], ("xgboost", "catboost"))
        self.assertEqual(data.model_families[34], ())

    def test_binary_journey_uses_saved_evidence_without_loading_models_or_datasets(self):
        base = "models/requested_model_comparison"
        self.write("data/processed/preprocessing_report.json", {"selected_features": ["feature"]})
        self.write(f"{base}/model_comparison_report.json", {
            "best_model": {"model": "lightgbm", "status": "success", "f1": .93}})
        bundle = self.root / base / "lightgbm/model.joblib"
        bundle.parent.mkdir()
        bundle.write_bytes(b"This is not a pickle and must never be loaded.")
        self.write(f"{base}/lightgbm/shap/shap_report.json", {"method": "tree", "explained_rows": 10})
        with patch("joblib.load", side_effect=AssertionError("No model loads")), \
             patch("pandas.read_csv", side_effect=AssertionError("No dataset reads")):
            result = load_binary_journey(self.root)
        self.assertEqual(result.f1, .93)
        self.assertEqual(result.best_model, "lightgbm")
        self.assertTrue(all(step.complete for step in result.milestones))

    def test_binary_missing_partial_and_failed_reports_do_not_invent_progress(self):
        result = load_binary_journey(self.root)
        self.assertIsNone(result.f1)
        self.assertFalse(any(step.complete for step in result.milestones))
        base = "models/requested_model_comparison"
        self.write("data/processed/preprocessing_report.json", {"status": "running"})
        self.write(f"{base}/lightgbm/shap/shap_report.json", {"method": "tree", "explained_rows": True})
        for score, status, name in ((float("nan"), "success", "lightgbm"), (1.2, "success", "lightgbm"),
                                     (.9, "failed", "lightgbm"), (True, "success", "lightgbm"),
                                     (.9, "success", "../../external")):
            self.write(f"{base}/model_comparison_report.json", {
                "best_model": {"model": name, "status": status, "f1": score}})
            result = load_binary_journey(self.root)
            self.assertIsNone(result.f1)
            self.assertFalse(any(step.complete for step in result.milestones))

    def test_three_seed_validation_and_final_scores_remain_separate(self):
        self.fixture()
        with patch("joblib.load", side_effect=AssertionError("Browsing must not load models")):
            data = load_journey(self.root)
        self.assertEqual(data.development.macro_f1.tolist(), [.8, .8])
        self.assertAlmostEqual(data.development["std"].iloc[0], .016329931618554536)
        self.assertAlmostEqual(data.final.loc[data.final.candidate == "specialist", "macro_f1_mean"].iloc[0], .84)
        self.assertEqual(data.recall_budget_pp, .5)
        self.assertIs(data.recall_budget_pass, False)
        self.assertEqual(data.interval, (.02, .06))

    def test_partial_or_duplicate_or_test_rows_never_become_validation_progress(self):
        self.fixture()
        path = "outputs/studies/34class_v1/confirmation_v1/report.json"
        original = json.loads((self.root / path).read_text())
        for kind in ("partial", "duplicate", "test", "malformed"):
            report = json.loads(json.dumps(original))
            if kind == "partial": report["status"] = "confirming"
            if kind == "duplicate": report["results"][1]["seed"] = 42
            if kind == "test": report["results"][0]["split"] = "test"
            if kind == "malformed": report["results"] = None
            self.write(path, report)
            self.assertFalse((load_journey(self.root).development.classes == 34).any(), kind)

    def test_incomplete_global_evaluation_hides_final_tables_even_if_csv_exists(self):
        self.fixture()
        self.write("outputs/studies/final_protocol_v1/report.json", {"status": "running"})
        data = load_journey(self.root)
        self.assertFalse(data.final_complete)
        self.assertTrue(data.final.empty)
        self.assertTrue(data.paper.empty)
        self.assertIsNone(data.interval)

    def test_missing_and_corrupt_reports_are_unavailable_not_fake_completions(self):
        data = load_journey(self.root)
        self.assertEqual(sum(m.complete for m in data.milestones), 0)
        self.assertTrue(data.final.empty)
        self.assertEqual(data.issues, [])
        path = self.write("outputs/experiments/8class_v1/experiment_report.json", {})
        path.write_text('{"status":', encoding="utf-8")
        data = load_journey(self.root)
        self.assertFalse(data.milestones[1].complete)
        self.assertEqual(len(data.issues), 1)

    def test_stale_uncertainty_is_not_attached_to_different_final_scores(self):
        self.fixture()
        path = "outputs/studies/final_results_v1/report.json"
        report = json.loads((self.root / path).read_text())
        report["macro_f1_uncertainty"]["candidate"] = .99
        self.write(path, report)
        data = load_journey(self.root)
        self.assertIsNone(data.interval)
        self.assertTrue(any("does not match" in issue for issue in data.issues))

    def test_task_journeys_use_only_their_own_evidence_and_steps(self):
        self.write("outputs/experiments/34class_pilot_v1/experiment_report.json", {"status": "complete"})
        self.write("outputs/experiments/8class_v1/experiment_report.json", {"status": "complete"})
        data = load_journey(self.root)
        self.assertTrue(data.task_milestones[34][0].complete)
        self.assertFalse(data.task_milestones[8][0].complete)
        self.assertTrue(data.task_milestones[8][1].complete)
        self.assertEqual(len(data.task_milestones[8]), 9)
        self.assertEqual(len(data.task_milestones[34]), 8)
        for count, other in ((8, 34), (34, 8)):
            self.assertFalse(any(f"/{other}class" in path for step in data.task_milestones[count] for path in step.evidence))
        self.assertFalse(any("eight-class ablations" in step.title for step in data.task_milestones[34]))

    def test_earlier_graph_data_keeps_protocols_and_score_roles_separate(self):
        def row(score, split="validation", seed=42):
            return {"candidate": "example", "decision": "argmax", "macro_f1": score, "split": split, "seed": seed}
        self.write("outputs/experiments/8class_pilot_v1/experiment_report.json",
                   {"status": "complete", "results": [row(.7), row(.99, "test"), row(.98, seed=123)]})
        self.write("outputs/experiments/34class_pilot_v1/experiment_report.json",
                   {"status": "running", "results": [row(.9)]})
        self.write("outputs/studies/8class_v1/report.json", {"status": "complete",
                   "results": [row(.8, "tuning"), row(.99, "test")], "validation_results": [row(.75)]})
        values = load_journey(self.root).exploration
        self.assertEqual(values.classes.tolist(), [8, 8, 8])
        self.assertEqual(values.phase.tolist(), ["Local pilot", "Published training search", "Published model trials"])
        self.assertEqual(values.macro_f1.tolist(), [.7, .8, .75])
        self.assertEqual(values.split.tolist(), ["validation", "tuning", "validation"])
        self.assertEqual(values.seed.tolist(), [42, 42, 42])


class JourneyPageTests(JourneyFixture):
    def test_navigation_from_all_workflows_and_refresh_uses_saved_results_only(self):
        from streamlit.testing.v1 import AppTest
        self.fixture()
        self.write("models/requested_model_comparison/model_comparison_report.json", {
            "best_model": {"model": "lightgbm", "status": "success", "f1": .93},
            "models": [{"model": "random_forest", "status": "success", "f1": .8},
                       {"model": "lightgbm", "status": "success", "f1": .93}]})
        with patch.object(dashboard_journey, "PROJECT_ROOT", self.root), \
             patch("joblib.load", side_effect=AssertionError("Journey must not load models")), \
             patch("ddos_iot.dashboard.run_command", side_effect=AssertionError("Journey must not run a pipeline")), \
             patch("ddos_iot.dashboard_multiclass.start_training_job", side_effect=AssertionError("Journey must not train")), \
             patch("ddos_iot.dashboard_multiclass.discover_runs", return_value=[]):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            self.assertEqual(app.radio(key="dashboard_navigation").value, "Overview")
            self.assertFalse(any(s.label == "Classification problem" for s in app.selectbox))
            self.assertFalse(any(m.label == "Raw CSV files" for m in app.metric))
            self.assertTrue(any(h.value == "Overview" for h in app.header))
            self.assertEqual(app.metric[0].label, "Binary F1")
            self.assertEqual(app.metric[0].value, "0.9300")
            self.assertEqual(app.metric[1].label, "8-class Macro-F1")
            self.assertEqual(app.metric[1].value, "0.8100")
            self.assertEqual(app.metric[2].value, "0.8400")
            self.assertEqual(app.metric[2].delta, "+0.0400 vs XGBoost")
            self.assertTrue(any("recall limit" in w.value for w in app.warning))
            self.assertEqual(len(app.tabs), 0)
            self.assertEqual(len(app.get("progress")), 0)
            self.assertEqual([h.value for h in app.subheader], ["Binary classification", "8-class classification", "34-class classification"])
            self.assertEqual(len(app.get("vega_lite_chart")), 3)
            self.assertTrue(any("**2 models**" in m.value for m in app.markdown))
            self.assertFalse(any(s.label == "Experiment stage" for s in app.selectbox))
            self.fixture_update(.85)
            app.button(key="journey_refresh").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[2].value, "0.8500")
            self.assertEqual(app.metric[2].delta, "+0.0500 vs XGBoost")
            for index, count in ((1, 8), (2, 34)):
                app.radio(key="dashboard_navigation").set_value("Workflows").run()
                app.selectbox(key="classification_problem").select_index(index).run()
                self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
                self.assertNotIn("Research Journey", app.radio(key=f"mc_{count}_page").options)
                app.radio(key="dashboard_navigation").set_value("Overview").run()
                self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
                self.assertEqual([m.label for m in app.metric], ["Binary F1", "8-class Macro-F1", "34-class Macro-F1"])
                self.assertFalse(any(s.label == "Classification problem" for s in app.selectbox))
                self.assertFalse(any(s.label == "Experiment" for s in app.selectbox))
                self.assertEqual(len(app.tabs), 0)

    def fixture_update(self, score):
        path = self.root / "outputs/studies/final_results_v1/test_summary.csv"
        rows = pd.read_csv(path)
        rows.loc[rows.candidate == "specialist", "macro_f1_mean"] = score
        rows.to_csv(path, index=False)
        report_path = "outputs/studies/final_results_v1/report.json"
        report = json.loads((self.root / report_path).read_text())
        report["macro_f1_uncertainty"]["candidate"] = score
        self.write(report_path, report)

    def test_empty_checkout_renders_the_journey_without_data_or_models(self):
        from streamlit.testing.v1 import AppTest
        with patch.object(dashboard_journey, "PROJECT_ROOT", self.root), \
             patch("joblib.load", side_effect=AssertionError("No model loads")):
            app = AppTest.from_string("from ddos_iot.dashboard_journey import render_journey\nrender_journey()", default_timeout=30).run()
            self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
            self.assertTrue(all(m.value == "Unavailable" for m in app.metric))
            self.assertTrue(any("No saved research reports" in i.value for i in app.info))


if __name__ == "__main__":
    unittest.main()
