"""Dashboard routing, safe job launch, progress and prediction-only integration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiclass import parse_args
from dashboard_worker import run_job
from ddos_iot.dashboard_runs import (
    build_training_command, discover_runs, read_report, saved_models,
    start_training_job, tail_log, write_job,
)


class DashboardRunTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="dashboard_test_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_command_preserves_cli_defaults_for_each_task(self):
        for count in (8, 34):
            command = build_training_command(count, [])
            args = parse_args(command[4:])
            self.assertEqual(args.num_classes, count)
            self.assertEqual(args.cv_folds, 5)
            self.assertEqual(args.feature_selection, "median")
            self.assertEqual(args.data_protocol, "paper" if count == 8 else "existing")
            self.assertEqual(args.cv_preprocessing, "train" if count == 8 else "fold")
        with self.assertRaises(ValueError):
            build_training_command(8, ["--num-classes=34"])

    def test_discovery_reads_legacy_and_new_reports_without_loading_pickle(self):
        for name in ("legacy34", "family8", "queued8", "unrelated", "partial"):
            (self.root / name).mkdir()
        write_job(self.root / "legacy34" / "label_mapping.json", {str(i): f"label{i}" for i in range(34)})
        write_job(self.root / "family8" / "data_report.json", {"class_count": 8})
        write_job(self.root / "queued8" / "dashboard_job.json", {"class_count": 8, "status": "queued"})
        (self.root / "partial" / "data_report.json").write_text('{"class_count":')
        model = self.root / "legacy34" / "xgboost" / "model.joblib"
        model.parent.mkdir()
        model.write_bytes(b"not a pickle; discovery must not load this")
        with patch("joblib.load", side_effect=AssertionError("Do not unpickle during discovery")):
            self.assertEqual(discover_runs(34, self.root), [self.root / "legacy34"])
            self.assertEqual({p.name for p in discover_runs(8, self.root)}, {"family8", "queued8"})
            self.assertEqual(saved_models(self.root / "legacy34"), [model])
        self.assertEqual(read_report(self.root / "partial" / "data_report.json"), {})
        log = self.root / "test.log"
        log.write_text("a" * 100 + "latest")
        self.assertEqual(tail_log(log, 6), "latest")

    def test_background_launch_reserves_new_directory_and_never_overwrites(self):
        folder = self.root / "fresh"
        command = build_training_command(34, ["--output-path", str(folder)])
        with patch("ddos_iot.dashboard_runs.subprocess.Popen") as launch:
            self.assertEqual(start_training_job(command, models_root=self.root), folder)
            self.assertEqual(read_report(folder / "dashboard_job.json")["status"], "queued")
            call = launch.call_args
            self.assertNotIn("shell", call.kwargs)
            self.assertIn("dashboard_worker.py", call.args[0][3])
            with self.assertRaises(FileExistsError):
                start_training_job(command, models_root=self.root)
            launch.assert_called_once()
            outside = build_training_command(8, ["--output-path", str(self.root / "nested" / "run")])
            with self.assertRaises(ValueError):
                start_training_job(outside, models_root=self.root)

    def test_launch_failure_and_worker_status_are_recorded(self):
        folder = self.root / "failed_launch"
        command = build_training_command(34, ["--output-path", str(folder)])
        with patch("ddos_iot.dashboard_runs.subprocess.Popen", side_effect=OSError("launch unavailable")):
            with self.assertRaises(OSError):
                start_training_job(command, models_root=self.root)
        self.assertEqual(read_report(folder / "dashboard_job.json")["status"], "failed")
        for error in (None, RuntimeError("fit failure"), KeyboardInterrupt()):
            with self.subTest(error=error):
                folder = self.root / f"worker_{type(error).__name__}"
                folder.mkdir()
                command = build_training_command(8, ["--output-path", str(folder)])
                write_job(folder / "dashboard_job.json", {"status": "queued", "arguments": command[4:]})
                with patch("dashboard_worker.configure_logging"), \
                     patch("ddos_iot.multiclass.run_multiclass_training", side_effect=error), \
                     patch("traceback.print_exc"):
                    code = run_job(folder)
                job = read_report(folder / "dashboard_job.json")
                self.assertEqual(job["returncode"], code)
                self.assertEqual(job["status"], "complete" if error is None else
                                 "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
                with self.assertRaises(ValueError):
                    run_job(folder)

    def test_real_background_worker_saves_tiny_cpu_model(self):
        from test_multiclass_8 import family_fixture

        train, test = self.root / "train.csv", self.root / "test.csv"
        family_fixture(12).to_csv(train, index=False)
        family_fixture(3).to_csv(test, index=False)
        folder = self.root / "real_worker"
        command = build_training_command(8, [
            "--data-protocol", "existing", "--train-input-path", str(train),
            "--test-input-path", str(test), "--output-path", str(folder),
            "--models", "xgboost", "--device", "cpu", "--cv-folds", "0",
            "--feature-selection", "none", "--n-estimators", "2", "--n-jobs", "1", "--no-plots",
        ])
        processes = []
        original_popen = subprocess.Popen

        def capture(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch("ddos_iot.dashboard_runs.subprocess.Popen", side_effect=capture):
            start_training_job(command, models_root=self.root)
        process = processes[0]
        try:
            code = process.wait(timeout=60)
        finally:
            if process.poll() is None:
                # Only this test's own child; never an existing training job.
                process.terminate()
                process.wait(timeout=10)
        self.assertEqual(code, 0, tail_log(folder / "dashboard_training.log"))
        self.assertEqual(read_report(folder / "dashboard_job.json")["status"], "complete")
        self.assertTrue((folder / "xgboost" / "model.joblib").is_file())
        self.assertEqual(read_report(folder / "xgboost" / "model_save_report.json")["evaluation_status"], "complete")


class DashboardPageTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        # Streamlit creates its own temporary scripts; keep them in the same
        # disposable workspace area, not in models/ or a sandbox system temp.
        self.previous_temp = tempfile.tempdir
        tempfile.tempdir = str(ROOT / "tmp")
        self.addCleanup(setattr, tempfile, "tempdir", self.previous_temp)
        self.temp = tempfile.TemporaryDirectory(prefix="dashboard_pages_", dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = {}
        for count in (8, 34):
            run = self.root / f"classes_{count}"
            model = run / "xgboost"
            model.mkdir(parents=True)
            (model / "model.joblib").write_bytes(b"not loaded while browsing")
            write_job(run / "data_report.json", {"class_count": count})
            write_job(run / "model_comparison_report.json", {"class_count": count, "best_model": "xgboost",
                       "selection_metric": "cv_macro_f1_mean", "status": "complete"})
            write_job(model / "metrics.json", {"macro_f1": 0.8})
            pd.DataFrame({"class": ["Benign"], "train_selected": [100], "test_selected": [20]}).to_csv(run / "class_counts.csv", index=False)
            pd.DataFrame({"model": ["xgboost"], "accuracy": [0.9], "macro_f1": [0.8]}).to_csv(run / "model_comparison.csv", index=False)
            self.runs[count] = [run]

    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def assert_no_exception(self, app):
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def test_both_tasks_all_pages_and_binary_switch_without_training(self):
        with patch("ddos_iot.dashboard_multiclass.discover_runs", side_effect=lambda count: self.runs[count]), \
             patch("ddos_iot.dashboard_multiclass.start_training_job", side_effect=AssertionError("Browsing must not train")), \
             patch("joblib.load", side_effect=AssertionError("Browsing must not load models")):
            app = self.app()
            self.assert_no_exception(app)
            for index, count in ((1, 8), (2, 34)):
                app.selectbox(key="classification_problem").select_index(index).run()
                for page in ("Overview", "Dataset", "Model Comparison", "SHAP", "Predict", "Run Pipeline"):
                    app.radio(key=f"mc_{count}_page").set_value(page).run()
                    self.assert_no_exception(app)
                cv = next(c for c in app.checkbox if c.label == "Five-fold grid-search tuning")
                self.assertTrue(cv.value)
                self.assertEqual(next(s for s in app.selectbox if s.label == "Training device").value, "gpu")
                self.assertEqual(next(s for s in app.selectbox if s.label == "LightGBM device").value, "cpu")
            app.selectbox(key="classification_problem").select_index(0).run()
            self.assert_no_exception(app)

    def test_no_models_and_start_requires_confirmation(self):
        with patch("ddos_iot.dashboard_multiclass.discover_runs", return_value=[]), \
             patch("ddos_iot.dashboard_multiclass.start_training_job") as launch:
            app = self.app()
            app.selectbox(key="classification_problem").select_index(1).run()
            app.radio(key="mc_8_page").set_value("Predict").run()
            self.assert_no_exception(app)
            self.assertTrue(any("No saved model" in m.value for m in app.info))
            app.radio(key="mc_8_page").set_value("Run Pipeline").run()
            next(b for b in app.button if b.label == "Start multiclass training").click().run()
            self.assert_no_exception(app)
            launch.assert_not_called()
            self.assertTrue(any("Confirm" in m.value for m in app.warning))

    def test_saved_8_and_34_class_metrics_and_confusion_tables_without_loading_models(self):
        for count in (8, 34):
            folder = self.runs[count][0] / "xgboost"
            labels = [f"class_{i:02d}" for i in range(count)]
            matrix = np.eye(count, dtype=int) * 10
            matrix[0, 0], matrix[0, 1] = 8, 2
            pd.DataFrame(matrix, index=labels, columns=labels).to_csv(folder / "confusion_matrix.csv", index_label="true_label")
            rows = [{"class": label, "precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 10}
                    for label in labels]
            rows.append({"class": "macro avg", "precision": 0.75, "recall": 0.9, "f1-score": 0.8, "support": count * 10})
            pd.DataFrame(rows).to_csv(folder / "classification_report_test.csv", index=False)
        with patch("ddos_iot.dashboard_multiclass.discover_runs", side_effect=lambda count: self.runs[count]), \
             patch("ddos_iot.dashboard_multiclass.start_training_job", side_effect=AssertionError("Metrics must not train")), \
             patch("joblib.load", side_effect=AssertionError("Metrics must not load models")):
            app = self.app()
            for index, count in ((1, 8), (2, 34)):
                app.selectbox(key="classification_problem").select_index(index).run()
                app.radio(key=f"mc_{count}_page").set_value("Model Comparison").run()
                self.assert_no_exception(app)
                tables = [element.value for element in app.dataframe]
                scores = next(table for table in tables if "macro_precision" in table and "macro_recall" in table)
                self.assertAlmostEqual(scores.iloc[0]["macro_precision"], 0.75)
                counts = next(table for table in tables if table.index.name == "Actual class")
                self.assertEqual(counts.shape, (count, count))
                self.assertEqual(counts.iloc[0, 1], 2)
                app.radio(key=f"mc_classes_{count}_xgboost_matrix_values").set_value("Row percentages").run()
                self.assert_no_exception(app)
                normalized = next(element.value for element in app.dataframe if element.value.index.name == "Actual class")
                self.assertAlmostEqual(normalized.iloc[0, 1], 0.2)

    def test_prediction_page_calls_inference_not_training_and_retains_export(self):
        output_root = self.root / "predictions"
        raw = self.root / "raw.csv"
        raw.write_text("feature\n1\n2\n")

        def predict(model, input_file, output_file, chunk_size, **kwargs):
            self.assertEqual(input_file, raw)
            self.assertEqual(kwargs["expected_num_classes"], 34)
            output_file.parent.mkdir(exist_ok=True)
            output_file.write_text("predicted_label\nBenignTraffic\nDDoS-SYN_Flood\n")
            return {"rows": 2, "model": "xgboost", "class_count": 34,
                    "model_path": str(model), "output_file": str(output_file)}

        with patch("ddos_iot.dashboard_multiclass.discover_runs", side_effect=lambda count: self.runs[count]), \
             patch("ddos_iot.dashboard_multiclass.PREDICTIONS_DIR", output_root), \
             patch("ddos_iot.multiclass_inference.run_multiclass_prediction", side_effect=predict) as inference, \
             patch("ddos_iot.multiclass.run_multiclass_training", side_effect=AssertionError("Prediction must not train")):
            app = self.app()
            app.selectbox(key="classification_problem").select_index(2).run()
            app.radio(key="mc_34_page").set_value("Predict").run()
            app.text_input(key="mc_34_predict_input").set_value(str(raw)).run()
            app.button(key="mc_34_predict_submit").click().run()
            self.assert_no_exception(app)
            inference.assert_called_once()
            self.assertTrue(any("Predicted 2 rows" in m.value for m in app.success))
            app.run()
            inference.assert_called_once()
            self.assertEqual(len(list(output_root.glob("*.csv"))), 1)

    def test_confirmed_form_launches_selected_task_with_five_fold_defaults(self):
        raw = self.root / "raw.csv"
        raw.write_text("feature,label\n1,BenignTraffic\n")
        with patch("ddos_iot.dashboard_multiclass.discover_runs", return_value=[]), \
             patch("ddos_iot.dashboard_multiclass.start_training_job", return_value=self.root / "launched") as launch:
            app = self.app()
            for index, count in ((1, 8), (2, 34)):
                app.selectbox(key="classification_problem").select_index(index).run()
                app.radio(key=f"mc_{count}_page").set_value("Run Pipeline").run()
                for field in app.text_input:
                    if field.label in ("Raw training/source CSV", "Raw test CSV (existing-split protocol only)"):
                        field.set_value(str(raw))
                next(c for c in app.checkbox if c.label.startswith("I understand")).check()
                next(b for b in app.button if b.label == "Start multiclass training").click().run()
                self.assert_no_exception(app)
                self.assertTrue(any("Started launched" in m.value for m in app.success))
                args = parse_args(launch.call_args.args[0][4:])
                self.assertEqual(args.num_classes, count)
                self.assertEqual(args.cv_folds, 5)
                self.assertEqual(args.device, "gpu")
                self.assertEqual(args.lightgbm_device, "cpu")
                self.assertEqual(args.feature_selection, "median")
                self.assertEqual(args.data_protocol, "paper" if count == 8 else "existing")

    def test_multiclass_shap_generation_and_class_row_navigation_never_train(self):
        def generate(model, input_file, **kwargs):
            count = kwargs["expected_num_classes"]
            folder = model.parent / "shap" / "generated"
            folder.mkdir(parents=True)
            rng = np.random.default_rng(31)
            values = rng.normal(size=(3, 4, count))
            outputs = values.sum(axis=1)
            probabilities = np.exp(outputs - outputs.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            classes = np.asarray([f"class_{i}" for i in range(count)])
            np.savez_compressed(folder / "shap_values.npz", values=values,
                base_values=np.zeros((3, count)), outputs=outputs, probabilities=probabilities,
                row_indices=np.array([4, 9, 20]), raw_values=rng.normal(size=(3, 4)),
                feature_names=np.asarray([f"feature_{i}" for i in range(4)]), classes=classes)
            report = {"status": "complete", "model": "xgboost", "class_count": count,
                      "method": "xgboost_native_tree_shap", "output_space": "raw_margin",
                      "explained_rows": 3, "classes": classes.tolist(), "output_path": str(folder),
                      "sampling": {"sampling_scope": "prefix"}}
            write_job(folder / "shap_report.json", report)
            pd.DataFrame({"class": classes, "feature": "feature_0", "mean_abs_shap": 1}).to_csv(
                folder / "shap_class_importance.csv", index=False)
            return report

        with patch("ddos_iot.dashboard_multiclass.discover_runs", side_effect=lambda count: self.runs[count]), \
             patch("ddos_iot.multiclass_shap.run_multiclass_shap", side_effect=generate) as generate_call, \
             patch("joblib.load", side_effect=AssertionError("Viewing must not load model")), \
             patch("ddos_iot.multiclass.run_multiclass_training", side_effect=AssertionError("SHAP must not train")):
            app = self.app()
            for index, count in ((1, 8), (2, 34)):
                app.selectbox(key="classification_problem").select_index(index).run()
                app.radio(key=f"mc_{count}_page").set_value("SHAP").run()
                app.button(key=f"mc_{count}_shap_generate").click().run()
                self.assert_no_exception(app)
                self.assertEqual(len(app.error), 0, [x.value for x in app.error])
                app.selectbox(key=f"mc_{count}_shap_class").set_value(count - 1).run()
                app.selectbox(key=f"mc_{count}_shap_row").set_value(1).run()
                self.assert_no_exception(app)
                self.assertEqual(len(app.error), 0, [x.value for x in app.error])
                self.assertTrue(any("NOT directly" in m.value for m in app.info))
            self.assertEqual(generate_call.call_count, 2)


if __name__ == "__main__":
    unittest.main()
