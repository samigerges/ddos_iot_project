"""Streamlit dashboard for the IoT DDoS detection project."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ddos_iot.dashboard_metrics import render_evaluation_tables, render_metric_table
from ddos_iot.paths import (
    MODEL_COMPARISON_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RAW_TEST_FILE,
    RAW_TRAIN_FILE,
)


TARGET_COLUMN = "target"
PIPELINE_PREPROCESS_RESULT_KEY = "pipeline_preprocess_result"
PIPELINE_TRAINING_RESULT_KEY = "pipeline_training_result"
PIPELINE_TRAINING_CONTEXT_KEY = "pipeline_training_context"
PIPELINE_TRAIN_SAMPLE_SIZE_KEY = "pipeline_train_sample_size"


st.set_page_config(
    page_title="IoT Intrusion Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render the dashboard."""

    st.title("IoT Intrusion Detection")
    st.caption(
        "Preprocess traffic flows, compare models, inspect SHAP explanations, and run predictions."
    )

    problem = st.sidebar.selectbox(
        "Classification problem",
        ("Binary — Benign vs DDoS", "8 classes — Attack families", "34 classes — Original labels"),
        key="classification_problem",
    )
    if problem != "Binary — Benign vs DDoS":
        from ddos_iot.dashboard_multiclass import render_multiclass

        render_multiclass(8 if problem.startswith("8 ") else 34)
        return

    page = st.sidebar.radio(
        "View",
        ("Overview", "Run Pipeline", "Dataset", "Model Comparison", "SHAP", "Predict"),
    )

    if page == "Overview":
        render_overview()
    elif page == "Run Pipeline":
        render_pipeline_runner()
    elif page == "Dataset":
        render_dataset_page()
    elif page == "Model Comparison":
        render_model_comparison_page()
    elif page == "SHAP":
        render_shap_page()
    else:
        render_predict_page()


def render_overview() -> None:
    """Render project status cards."""

    raw_files = list(RAW_DATA_DIR.rglob("*.csv")) if RAW_DATA_DIR.exists() else []
    comparison_file = MODEL_COMPARISON_DIR / "model_comparison.csv"
    best_report = read_json(MODEL_COMPARISON_DIR / "model_comparison_report.json")

    columns = st.columns(2)
    columns[0].metric("Raw CSV files", len(raw_files))
    columns[1].metric("Best model", best_model_name(best_report))

    if not raw_files and any(MODEL_COMPARISON_DIR.glob("*/model.joblib")):
        st.info(
            "Saved binary models are available. Training datasets are stored separately. "
            "To reuse a model, open Predict and upload a processed CSV."
        )

    st.subheader("Project Outputs")
    output_rows = [
        {
            "artifact": "Raw train data",
            "path": short_path(RAW_TRAIN_FILE),
            "status": path_status(RAW_TRAIN_FILE),
        },
        {
            "artifact": "Raw test data",
            "path": short_path(RAW_TEST_FILE),
            "status": path_status(RAW_TEST_FILE),
        },
        {
            "artifact": "Processed data",
            "path": short_path(PROCESSED_DATA_DIR),
            "status": path_status(PROCESSED_DATA_DIR),
        },
        {
            "artifact": "Model comparison",
            "path": short_path(MODEL_COMPARISON_DIR),
            "status": path_status(MODEL_COMPARISON_DIR),
        },
        {
            "artifact": "Prediction outputs",
            "path": short_path(PREDICTIONS_DIR),
            "status": path_status(PREDICTIONS_DIR),
        },
    ]
    st.dataframe(pd.DataFrame(output_rows), use_container_width=True, hide_index=True)

    if comparison_file.exists():
        st.subheader("Latest Comparison")
        st.dataframe(pd.read_csv(comparison_file), use_container_width=True, hide_index=True)
    else:
        st.info("No model comparison is available yet. Run the pipeline to generate results.")


def render_pipeline_runner() -> None:
    """Render controls for running the pipeline."""

    st.subheader("Run Pipeline")
    st.caption("Training uses the raw train CSV, and evaluation uses the raw test CSV.")
    st.caption("Step 1 creates processed train/test files. Step 2 trains on an allowed number of processed train rows.")

    left, right = st.columns(2)
    train_input_path = left.text_input(
        "Train input path",
        value=str(RAW_TRAIN_FILE.relative_to(PROJECT_ROOT)),
    )
    test_input_path = left.text_input(
        "Test input path",
        value=str(RAW_TEST_FILE.relative_to(PROJECT_ROOT)),
    )
    processed_path = left.text_input(
        "Processed path",
        value=str(PROCESSED_DATA_DIR.relative_to(PROJECT_ROOT)),
    )
    output_path = left.text_input(
        "Models output path",
        value=str(MODEL_COMPARISON_DIR.relative_to(PROJECT_ROOT)),
    )
    top_features = left.number_input("Top features", min_value=1, max_value=100, value=25, step=1)
    rebalance = left.selectbox("Training rebalance", ("undersample", "none", "oversample"))

    explain_shap = right.checkbox("Generate SHAP", value=True)
    shap_for_all = right.checkbox("SHAP for all models", value=False)
    no_plots = right.checkbox("Skip plots", value=False)
    max_rows = right.number_input("Max rows per raw file", min_value=0, value=0, step=1000)
    verbose = right.checkbox("Verbose logs", value=False)
    preprocess_state = st.session_state.get(PIPELINE_PREPROCESS_RESULT_KEY)
    training_state = st.session_state.get(PIPELINE_TRAINING_RESULT_KEY)

    st.markdown("**Step 1: Preprocess Data**")
    st.caption("This creates the processed train and test files that training will use.")
    if st.button("Run Preprocessing", key="run_preprocessing_button"):
        preprocess_command = [
            sys.executable,
            "src/preprocess.py",
            "--train-input-path",
            train_input_path,
            "--test-input-path",
            test_input_path,
            "--output-path",
            processed_path,
            "--top-features",
            str(top_features),
            "--rebalance",
            rebalance,
        ]
        if max_rows > 0:
            preprocess_command.extend(["--max-rows-per-file", str(max_rows)])
        if verbose:
            preprocess_command.append("--verbose")

        preprocess_result = run_command(preprocess_command)
        st.session_state[PIPELINE_PREPROCESS_RESULT_KEY] = snapshot_command_result(preprocess_result)
        if preprocess_result.returncode == 0:
            clear_dashboard_caches()
            st.session_state.pop(PIPELINE_TRAINING_RESULT_KEY, None)
            st.session_state.pop(PIPELINE_TRAINING_CONTEXT_KEY, None)
        preprocess_state = st.session_state.get(PIPELINE_PREPROCESS_RESULT_KEY)
        training_state = st.session_state.get(PIPELINE_TRAINING_RESULT_KEY)

    processed_summary = read_processed_split_summary(resolve_dashboard_path(processed_path))
    available_train_rows = processed_summary["train_rows"] if processed_summary else None
    available_test_rows = processed_summary["test_rows"] if processed_summary else None
    available_feature_count = processed_summary["feature_count"] if processed_summary else None

    if isinstance(preprocess_state, dict):
        preprocess_summary_metrics: dict[str, str] | None = None
        if available_train_rows is not None and available_test_rows is not None:
            preprocess_summary_metrics = {
                "Processed train rows": f"{available_train_rows:,}",
                "Processed test rows": f"{available_test_rows:,}",
                "Allowed training samples": format_allowed_training_samples(available_train_rows),
            }
        render_command_summary(
            "Preprocessing Summary",
            restore_command_result(preprocess_state),
            extra_metrics=preprocess_summary_metrics,
            key_prefix="pipeline_preprocess",
        )

    st.markdown("**Step 2: Train And Compare Models**")
    if available_train_rows is None or available_test_rows is None:
        st.info("Run preprocessing first so the dashboard can detect the allowed training sample range.")
        return

    if available_train_rows < 2:
        st.error("The processed training split must contain at least 2 rows before training can run.")
        return

    requested_sample_size = int(st.session_state.get(PIPELINE_TRAIN_SAMPLE_SIZE_KEY, available_train_rows))
    st.session_state[PIPELINE_TRAIN_SAMPLE_SIZE_KEY] = min(max(requested_sample_size, 2), available_train_rows)
    train_sample_step = 1000 if available_train_rows >= 1000 else max(1, available_train_rows // 10 or 1)
    train_sample_size = st.number_input(
        "Training samples after preprocessing",
        min_value=2,
        max_value=available_train_rows,
        step=train_sample_step,
        key=PIPELINE_TRAIN_SAMPLE_SIZE_KEY,
        help="This value is limited to the processed training rows currently available.",
    )

    if st.button("Run Training And Comparison", key="run_training_button"):
        training_command = [
            sys.executable,
            "src/compare_models.py",
            "--processed-path",
            processed_path,
            "--output-path",
            output_path,
            "--train-sample-size",
            str(train_sample_size),
        ]
        if explain_shap:
            training_command.append("--explain-shap")
        if shap_for_all:
            training_command.append("--shap-for-all-models")
        if no_plots:
            training_command.append("--no-plots")
        if verbose:
            training_command.append("--verbose")

        training_result = run_command(training_command)
        training_state = snapshot_command_result(training_result)
        training_state["train_sample_size"] = int(train_sample_size)
        training_state["feature_count"] = int(available_feature_count or 0)
        st.session_state[PIPELINE_TRAINING_RESULT_KEY] = training_state
        if training_result.returncode == 0:
            clear_dashboard_caches()
            training_context_path = resolve_dashboard_path(output_path) / "training_sample_summary.json"
            training_context = read_json(training_context_path)
            if isinstance(training_context, dict):
                st.session_state[PIPELINE_TRAINING_CONTEXT_KEY] = training_context

    if isinstance(training_state, dict):
        render_command_summary(
            "Training Summary",
            restore_command_result(training_state),
            extra_metrics={
                "Training samples": f"{int(training_state.get('train_sample_size', 0)):,}",
                "Features": f"{int(training_state.get('feature_count', 0)):,}",
            },
            key_prefix="pipeline_training",
        )


def render_dataset_page() -> None:
    """Render processed dataset and preprocessing summaries."""

    st.subheader("Dataset")
    st.caption("Processed train and test files come from the raw train/test CSVs, not an internal split.")
    st.caption("Train-side stats reflect the latest sampled training data when a training run is available.")
    report = read_json(PROCESSED_DATA_DIR / "preprocessing_report.json")
    processed_summary = read_processed_split_summary(PROCESSED_DATA_DIR)
    if processed_summary is None:
        st.info("Processed train/test files are missing. Run preprocessing first.")
        return

    training_summary = resolve_training_summary(
        processed_summary["train_rows"],
        processed_summary["feature_count"],
        processed_summary["train_class_counts"],
    )

    raw_sources = extract_raw_sources(report)
    if raw_sources:
        st.subheader("Raw Sources")
        st.dataframe(pd.DataFrame(raw_sources), use_container_width=True, hide_index=True)

    columns = st.columns(4)
    columns[0].metric("Train rows", f"{training_summary['train_rows_used']:,}")
    columns[1].metric("Test rows", f"{processed_summary['test_rows']:,}")
    columns[2].metric("Features", f"{training_summary['feature_count']:,}")
    columns[3].metric("Target", TARGET_COLUMN)

    st.subheader("Class Distribution")
    class_counts = pd.DataFrame(
        {
            "split": ["train", "test"],
            "benign": [
                int(training_summary["class_counts"].get("0", 0)),
                int(processed_summary["test_class_counts"].get("0", 0)),
            ],
            "ddos": [
                int(training_summary["class_counts"].get("1", 0)),
                int(processed_summary["test_class_counts"].get("1", 0)),
            ],
        }
    )
    st.dataframe(class_counts, use_container_width=True, hide_index=True)
    st.bar_chart(class_counts.set_index("split")[["benign", "ddos"]])

    st.subheader("Selected Features")
    selected_features = read_json(PROCESSED_DATA_DIR / "selected_features.json") or []
    if selected_features:
        st.dataframe(pd.DataFrame({"feature": selected_features}), use_container_width=True, hide_index=True)

    if report:
        st.subheader("Preprocessing Report")
        st.json(report, expanded=False)


def render_model_comparison_page() -> None:
    """Render model comparison metrics and per-model artifacts."""

    st.subheader("Model Comparison")
    comparison_path = MODEL_COMPARISON_DIR / "model_comparison.csv"
    report = read_json(MODEL_COMPARISON_DIR / "model_comparison_report.json")

    if not comparison_path.exists():
        st.info("No model comparison file exists yet. Run model comparison first.")
        return

    comparison = pd.read_csv(comparison_path)
    training_summary = read_latest_training_summary()
    summary_cols = st.columns(3)
    summary_cols[0].metric("Best model", best_model_name(report))
    summary_cols[1].metric(
        "Training rows",
        format_count_or_missing(training_summary.get("train_rows_used") if isinstance(training_summary, dict) else None),
    )
    summary_cols[2].metric(
        "Features",
        format_count_or_missing(training_summary.get("feature_count") if isinstance(training_summary, dict) else None),
    )
    st.subheader("Test metrics by model")
    st.caption("Scores range from 0 to 1. F1, precision, and recall use DDoS as the positive class.")
    render_metric_table(
        comparison,
        ("model", "status", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
         "false_positive_rate", "false_negative_rate"),
        sort_by=report.get("selected_metric", "f1") if isinstance(report, dict) else "f1",
    )
    st.subheader("Timing comparison")
    render_metric_table(comparison, ("model", "train_time_seconds", "inference_time_seconds", "inference_time_per_row_ms"))

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "train_time_seconds",
        "inference_time_seconds",
    ]
    available_metrics = [column for column in metric_columns if column in comparison.columns]
    if available_metrics:
        selected_metric = st.selectbox(
            "Chart metric",
            available_metrics,
            index=available_metrics.index("f1") if "f1" in available_metrics else 0,
        )
        st.bar_chart(comparison.set_index("model")[selected_metric])

    model_dirs = available_model_dirs()
    if model_dirs:
        st.subheader("Model Details")
        winner = best_model_name(report)
        default_model = next((index for index, folder in enumerate(model_dirs) if folder.name == winner), 0)
        model_dir = st.selectbox("Model", model_dirs, index=default_model, format_func=lambda path: path.name)
        render_model_detail(model_dir)


def render_model_detail(model_dir: Path) -> None:
    """Render artifacts for one model directory."""

    metrics = read_json(model_dir / "metrics.json")
    metadata = read_json(model_dir / "model_metadata.json")
    if isinstance(metrics, dict):
        rows = [{**values, "split": split.title()} for split, values in metrics.items()
                if split in ("train", "test") and isinstance(values, dict)]
        st.subheader("Train and test metrics")
        render_metric_table(pd.DataFrame(rows), ("split", "accuracy", "precision", "recall", "f1",
                                               "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"))
        splits = [split for split in ("test", "train") if isinstance(metrics.get(split), dict)]
    else:
        splits = []
    if isinstance(metadata, dict):
        metadata_cols = st.columns(2)
        metadata_cols[0].metric("Training rows", format_count_or_missing(metadata.get("train_rows_used")))
        metadata_cols[1].metric("Features", format_count_or_missing(metadata.get("feature_count")))

    split = st.radio("Evaluation split", splits, format_func=str.title, horizontal=True,
                     key=f"binary_{model_dir.name}_split") if splits else "test"
    render_evaluation_tables(model_dir, metrics=metrics.get(split) if splits else None,
                             split=split, key=f"binary_{model_dir.name}_{split}")
    render_images(
        [
            model_dir / "roc_curve.png",
            model_dir / "precision_recall_curve.png",
            model_dir / "score_distribution.png",
        ],
        columns=2,
    )

    feature_importance = model_dir / "feature_importance.csv"
    if feature_importance.exists():
        st.subheader("Feature Importance")
        st.dataframe(read_csv(feature_importance), use_container_width=True, hide_index=True)


def render_shap_page() -> None:
    """Render SHAP global and local explanation artifacts."""

    st.subheader("SHAP Explanations")
    shap_dirs = [path / "shap" for path in available_model_dirs() if (path / "shap").exists()]
    if not shap_dirs:
        st.info("No SHAP outputs exist yet. Run the pipeline with --explain-shap.")
        return

    shap_dir = st.selectbox("Model explanations", shap_dirs, format_func=lambda path: path.parent.name)
    report = read_json(shap_dir / "shap_report.json")
    metadata = read_json(shap_dir.parent / "model_metadata.json")
    if report:
        cols = st.columns(4)
        cols[0].metric("Method", str(report.get("method", "unknown")))
        cols[1].metric("Explained rows", report.get("explained_rows", 0))
        cols[2].metric("Background rows", report.get("background_rows", 0))
        cols[3].metric("Local rows", report.get("local_explanation_rows", 0))
    if isinstance(metadata, dict):
        metadata_cols = st.columns(2)
        metadata_cols[0].metric("Training rows", format_count_or_missing(metadata.get("train_rows_used")))
        metadata_cols[1].metric("Features", format_count_or_missing(metadata.get("feature_count")))

    importance_path = shap_dir / "shap_global_importance.csv"
    if importance_path.exists():
        st.subheader("Global Importance")
        importance = read_csv(importance_path)
        st.dataframe(importance, use_container_width=True, hide_index=True)
        if {"feature", "mean_abs_shap"}.issubset(importance.columns):
            st.bar_chart(importance.set_index("feature")["mean_abs_shap"])

    st.subheader("Global Plots")
    render_images([shap_dir / "shap_summary_bar.png", shap_dir / "shap_beeswarm.png"], columns=2)

    st.subheader("Local Explanations")
    local_table = shap_dir / "shap_local_top_features.csv"
    if local_table.exists():
        st.dataframe(read_csv(local_table), use_container_width=True, hide_index=True)
    waterfall_images = sorted(shap_dir.glob("shap_local_waterfall_*.png"))
    if waterfall_images:
        selected = st.selectbox("Waterfall plot", waterfall_images, format_func=lambda path: path.stem)
        st.image(str(selected))


def render_predict_page() -> None:
    """Render prediction controls for processed CSV files."""

    st.subheader("Predict")
    model_files = sorted(MODEL_COMPARISON_DIR.glob("*/model.joblib"))
    if not model_files:
        st.info("No trained model exists yet. Run training or model comparison first.")
        return

    left, right = st.columns(2)
    model_path = left.selectbox("Model", model_files, format_func=lambda path: path.parent.name)
    uploaded_file = left.file_uploader("Processed CSV", type=["csv"])
    threshold = left.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    explain_local_shap = right.checkbox("Generate local SHAP", value=True)
    shap_waterfall_count = right.number_input("Waterfall plots", min_value=1, max_value=50, value=10, step=1)
    shap_max_display = right.number_input("Top SHAP features", min_value=1, max_value=50, value=15, step=1)
    verbose = right.checkbox("Verbose logs", value=False)
    prediction_state = st.session_state.get("predict_page_result")

    if st.button("Predict", disabled=uploaded_file is None):
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        upload_path = PREDICTIONS_DIR / "uploaded_prediction_input.csv"
        output_path = PREDICTIONS_DIR / "ui_predictions.csv"
        shap_output_dir = PREDICTIONS_DIR / "ui_predictions_shap"
        upload_path.write_bytes(uploaded_file.getbuffer())
        prediction_command = [
            sys.executable,
            "src/predict.py",
            "--model-path",
            str(model_path),
            "--input-file",
            str(upload_path),
            "--output-file",
            str(output_path),
            "--threshold",
            str(threshold),
        ]
        if verbose:
            prediction_command.append("--verbose")

        prediction_result = run_command(
            prediction_command,
            spinner_text="Running prediction...",
            show_command=False,
        )
        clear_dashboard_caches()
        predictions = (
            read_csv(output_path)
            if prediction_result.returncode == 0 and output_path.exists()
            else None
        )
        metrics = read_json(output_path.with_suffix(".metrics.json")) if prediction_result.returncode == 0 else None
        prediction_state = {
            "prediction_returncode": prediction_result.returncode,
            "predictions": predictions,
            "metrics": metrics,
            "output_path": str(output_path),
            "shap_requested": bool(explain_local_shap),
            "shap_output_dir": str(shap_output_dir),
            "shap_report": None,
            "shap_error": None,
        }
        st.session_state["predict_page_result"] = prediction_state

        if prediction_result.returncode == 0 and output_path.exists():
            assert predictions is not None
            st.subheader("Predictions")
            st.dataframe(predictions.head(1000), use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions",
                data=output_path.read_bytes(),
                file_name="predictions.csv",
                mime="text/csv",
            )

        render_prediction_summary(
            result=prediction_result,
            predictions=predictions,
            metrics=metrics,
        )

        shap_report = None
        shap_error = None
        if prediction_result.returncode == 0 and output_path.exists() and explain_local_shap:
            shap_status = st.empty()
            shap_status.info("Generating local SHAP...")
            shap_command = [
                *prediction_command,
                "--explain-local-shap",
                "--shap-output-dir",
                str(shap_output_dir),
                "--shap-background-size",
                "200",
                "--shap-waterfall-count",
                str(shap_waterfall_count),
                "--shap-max-display",
                str(shap_max_display),
            ]
            shap_result = run_command(
                shap_command,
                spinner_text="Generating local SHAP...",
                show_command=False,
            )
            clear_dashboard_caches()
            shap_report = read_json(shap_output_dir / "shap_report.json") if shap_result.returncode == 0 else None
            shap_error = read_json(shap_output_dir / "shap_error.json")
            prediction_state["shap_report"] = shap_report
            prediction_state["shap_error"] = shap_error
            st.session_state["predict_page_result"] = prediction_state
            if shap_result.returncode == 0 and isinstance(shap_report, dict):
                shap_status.success("Local SHAP is ready.")
            else:
                shap_status.warning("Local SHAP could not be generated.")
            render_prediction_local_shap(shap_output_dir, shap_report, shap_error)

        return

    if not prediction_state:
        return

    predictions = prediction_state.get("predictions")
    metrics = prediction_state.get("metrics")
    output_path = Path(str(prediction_state.get("output_path", "")))
    prediction_returncode = int(prediction_state.get("prediction_returncode", 1))
    shap_requested = bool(prediction_state.get("shap_requested", False))
    shap_output_dir = Path(str(prediction_state.get("shap_output_dir", "")))
    shap_report = prediction_state.get("shap_report")
    shap_error = prediction_state.get("shap_error")

    if prediction_returncode == 0 and output_path.exists():
        assert isinstance(predictions, pd.DataFrame)
        st.subheader("Predictions")
        st.dataframe(predictions.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions",
            data=output_path.read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
        )

    render_prediction_summary(
        result=build_completed_process(prediction_returncode),
        predictions=predictions if isinstance(predictions, pd.DataFrame) else None,
        metrics=metrics if isinstance(metrics, (dict, list)) else None,
    )

    if prediction_returncode == 0 and output_path.exists() and shap_requested:
        render_prediction_local_shap(
            shap_output_dir,
            shap_report if isinstance(shap_report, (dict, list)) else None,
            shap_error if isinstance(shap_error, (dict, list)) else None,
        )


def run_command(
    command: list[str],
    *,
    spinner_text: str = "Running command...",
    show_command: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a project command and return the captured result."""

    if show_command:
        st.code(" ".join(command), language="powershell")
    with st.spinner(spinner_text):
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )


def build_completed_process(returncode: int) -> subprocess.CompletedProcess[str]:
    """Build a minimal completed-process shape for stored UI state."""

    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr="")


def snapshot_command_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Store the useful parts of a completed process in session state."""

    return {
        "returncode": int(result.returncode),
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def restore_command_result(state: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    """Rebuild a completed-process-like object from stored session state."""

    return subprocess.CompletedProcess(
        args=[],
        returncode=int(state.get("returncode", 1)),
        stdout=str(state.get("stdout", "")),
        stderr=str(state.get("stderr", "")),
    )


def render_command_summary(
    title: str,
    result: subprocess.CompletedProcess[str],
    *,
    extra_metrics: dict[str, str] | None = None,
    key_prefix: str | None = None,
) -> None:
    """Render a compact command summary with captured logs."""

    st.subheader(title)
    summary_metrics: list[tuple[str, str | int]] = [
        ("Status", "Success" if result.returncode == 0 else "Failed"),
    ]
    if extra_metrics:
        summary_metrics.extend(extra_metrics.items())

    cols = st.columns(len(summary_metrics))
    for column, (label, value) in zip(cols, summary_metrics):
        column.metric(label, value)

    widget_prefix = key_prefix or title.lower().replace(" ", "_")
    if result.stdout:
        st.text_area("Output", result.stdout, height=200, key=f"{widget_prefix}_output")


def render_prediction_summary(
    result: subprocess.CompletedProcess[str],
    predictions: pd.DataFrame | None,
    metrics: dict[str, Any] | list[Any] | None,
) -> None:
    """Render prediction metrics in one summary block."""

    st.subheader("Summary")
    has_ground_truth = isinstance(metrics, dict) and "accuracy" in metrics
    cols = st.columns(3)
    cols[0].metric("Prediction rows", f"{len(predictions):,}" if predictions is not None else "Missing")
    if isinstance(metrics, dict):
        cols[1].metric("Inference", format_duration_seconds(metrics.get("inference_time_seconds")))
        cols[2].metric(
            "Inference / row",
            format_duration_milliseconds(metrics.get("inference_time_per_row_ms")),
        )
    else:
        cols[1].metric("Inference", "Missing")
        cols[2].metric("Inference / row", "Missing")

    if has_ground_truth and isinstance(metrics, dict):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", format_metric(metrics.get("accuracy")))
        metric_cols[1].metric("Precision", format_metric(metrics.get("precision")))
        metric_cols[2].metric("Recall", format_metric(metrics.get("recall")))
        metric_cols[3].metric("F1", format_metric(metrics.get("f1")))
    elif result.returncode != 0:
        st.error(f"Prediction command failed with exit code {result.returncode}.")


def render_prediction_local_shap(
    shap_output_dir: Path,
    shap_report: dict[str, Any] | list[Any] | None,
    shap_error: dict[str, Any] | list[Any] | None,
) -> None:
    """Render prediction-time local SHAP outputs when available."""

    st.subheader("Local SHAP")
    if isinstance(shap_error, dict) and not isinstance(shap_report, dict):
        st.warning(f"Local SHAP failed: {shap_error.get('error', 'Unknown error')}")
        return

    local_table = shap_output_dir / "shap_local_top_features.csv"
    if not local_table.exists():
        st.info("No local SHAP outputs are available for this prediction run.")
        return

    if isinstance(shap_report, dict):
        cols = st.columns(4)
        cols[0].metric("Method", str(shap_report.get("method", "unknown")))
        cols[1].metric("Background rows", shap_report.get("background_rows", 0))
        cols[2].metric("Explained rows", shap_report.get("explained_rows", 0))
        cols[3].metric("Waterfall plots", shap_report.get("waterfall_plot_rows", 0))

    st.dataframe(read_csv(local_table), use_container_width=True, hide_index=True)
    waterfall_images = sorted(shap_output_dir.glob("shap_local_waterfall_*.png"))
    if waterfall_images:
        selected = st.selectbox(
            "Prediction waterfall plot",
            waterfall_images,
            format_func=lambda path: path.stem,
            key="predict_waterfall_plot",
        )
        st.image(str(selected))


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Read current report contents, including files added after the first visit."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with Streamlit caching."""

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def count_csv_rows(path: Path) -> int:
    """Count CSV data rows without loading the full file into a dataframe."""

    with path.open("r", encoding="utf-8", newline="") as file:
        row_count = sum(1 for _ in file)
    return max(row_count - 1, 0)


@st.cache_data(show_spinner=False)
def count_processed_feature_columns(path: Path) -> int:
    """Count processed feature columns from a CSV header."""

    columns = pd.read_csv(path, nrows=0).columns
    return max(len(columns) - (1 if TARGET_COLUMN in columns else 0), 0)


@st.cache_data(show_spinner=False)
def read_target_class_counts(path: Path) -> dict[str, int]:
    """Read binary target counts from a processed CSV."""

    target = pd.read_csv(path, usecols=[TARGET_COLUMN])[TARGET_COLUMN].astype(int)
    counts = target.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def read_processed_split_summary(processed_dir: Path) -> dict[str, Any] | None:
    """Read lightweight processed train/test summary values."""

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return None

    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": count_csv_rows(train_path),
        "test_rows": count_csv_rows(test_path),
        "feature_count": count_processed_feature_columns(train_path),
        "train_class_counts": read_target_class_counts(train_path),
        "test_class_counts": read_target_class_counts(test_path),
    }


def available_model_dirs() -> list[Path]:
    """Return model directories containing artifacts."""

    if not MODEL_COMPARISON_DIR.exists():
        return []
    return sorted(path for path in MODEL_COMPARISON_DIR.iterdir() if path.is_dir())


def render_images(paths: list[Path], columns: int) -> None:
    """Render existing images in a responsive grid."""

    existing = [path for path in paths if path.exists()]
    if not existing:
        st.info("No plots are available for this section.")
        return

    cols = st.columns(columns)
    for index, path in enumerate(existing):
        cols[index % columns].image(str(path), caption=path.name, use_container_width=True)


def best_model_name(report: dict[str, Any] | list[Any] | None) -> str:
    """Return the selected best model name from a comparison report."""

    if not isinstance(report, dict):
        return "Missing"
    best = report.get("best_model")
    if not isinstance(best, dict):
        return "Missing"
    return str(best.get("model", "Missing"))


def resolve_training_summary(
    default_rows: int,
    default_feature_count: int,
    default_class_counts: dict[str, int],
) -> dict[str, Any]:
    """Merge the latest sampled training context with processed-data defaults."""

    summary = read_latest_training_summary()
    if not isinstance(summary, dict):
        return {
            "train_rows_used": default_rows,
            "feature_count": default_feature_count,
            "class_counts": default_class_counts,
        }

    train_rows_used = summary.get("train_rows_used")
    feature_count = summary.get("feature_count")
    class_counts = summary.get("class_counts")
    return {
        "train_rows_used": min(int(train_rows_used), default_rows)
        if isinstance(train_rows_used, int) and train_rows_used > 0
        else default_rows,
        "feature_count": int(feature_count)
        if isinstance(feature_count, int) and feature_count > 0
        else default_feature_count,
        "class_counts": normalize_class_counts(class_counts, default_class_counts),
    }


def read_latest_training_summary() -> dict[str, Any] | None:
    """Read the saved summary for the latest sampled training run, when available."""

    training_context = st.session_state.get(PIPELINE_TRAINING_CONTEXT_KEY)
    if isinstance(training_context, dict):
        return training_context

    metadata = read_latest_model_metadata()
    if not isinstance(metadata, dict):
        return None

    summary = metadata.get("training_summary")
    if isinstance(summary, dict):
        return summary

    summary_path = metadata.get("training_summary_path")
    if isinstance(summary_path, str):
        loaded_summary = read_json(resolve_dashboard_path(summary_path))
        if isinstance(loaded_summary, dict):
            return loaded_summary

    train_rows_used = metadata.get("train_rows_used")
    feature_count = metadata.get("feature_count")
    class_counts = metadata.get("train_class_counts")
    if isinstance(train_rows_used, int) and isinstance(feature_count, int):
        return {
            "train_rows_used": train_rows_used,
            "feature_count": feature_count,
            "class_counts": class_counts if isinstance(class_counts, dict) else {},
        }

    training_args = metadata.get("training_args")
    if isinstance(training_args, dict):
        train_sample_size = training_args.get("train_sample_size")
        if isinstance(train_sample_size, int) and train_sample_size > 0:
            return {
                "train_rows_used": train_sample_size,
                "feature_count": int(metadata.get("feature_count", 0) or 0),
                "class_counts": class_counts if isinstance(class_counts, dict) else {},
            }
    return None


def normalize_class_counts(value: Any, fallback: dict[str, int]) -> dict[str, int]:
    """Normalize binary class-count dictionaries and fall back when needed."""

    if not isinstance(value, dict):
        return fallback

    normalized = {
        str(label): int(count)
        for label, count in value.items()
        if str(label) in {"0", "1"}
    }
    return normalized if normalized else fallback


def read_latest_model_metadata() -> dict[str, Any] | None:
    """Read the saved metadata for the latest selected best model, when available."""

    report = read_json(MODEL_COMPARISON_DIR / "model_comparison_report.json")
    if not isinstance(report, dict):
        return None

    best_model = report.get("best_model")
    if not isinstance(best_model, dict):
        return None

    # The recorded model_dir can use Windows paths from the training computer.
    model_dir = next((folder for folder in available_model_dirs() if folder.name == best_model.get("model")), None)
    if model_dir is None:
        return None
    metadata_path = model_dir / "model_metadata.json"
    metadata = read_json(metadata_path)
    return metadata if isinstance(metadata, dict) else None


def format_count_or_missing(value: Any) -> str:
    """Format a whole-number KPI or return Missing."""

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "Missing"


def format_allowed_training_samples(train_rows: int) -> str:
    """Format the allowed sampled-training range."""

    return f"2 to {train_rows:,}" if train_rows >= 2 else "Unavailable"


def path_status(path: Path) -> str:
    """Return a compact status for a result path."""

    if not path.exists():
        return "Missing"
    if path.is_file():
        return "Ready"
    return "Ready" if any(path.iterdir()) else "Empty"


def short_path(path: Path) -> str:
    """Return a path relative to the project root when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_dashboard_path(path: Path | str) -> Path:
    """Resolve dashboard-entered paths against the project root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def clear_dashboard_caches() -> None:
    """Clear cached file readers after preprocessing or training outputs change."""

    read_csv.clear()
    count_csv_rows.clear()
    count_processed_feature_columns.clear()
    read_target_class_counts.clear()


def format_metric(value: Any) -> str:
    """Format numeric metric values for cards."""

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "Missing"


def format_duration_seconds(value: Any) -> str:
    """Format seconds for compact KPI display."""

    try:
        return f"{float(value):.4f}s"
    except (TypeError, ValueError):
        return "Missing"


def format_duration_milliseconds(value: Any) -> str:
    """Format milliseconds for compact KPI display."""

    try:
        return f"{float(value):.4f} ms"
    except (TypeError, ValueError):
        return "Missing"


def extract_raw_sources(report: dict[str, Any] | list[Any] | None) -> list[dict[str, str]]:
    """Convert preprocessing input files into a simple table for display."""

    if not isinstance(report, dict):
        return []

    input_files = report.get("input_files")
    if not isinstance(input_files, dict):
        return []

    rows: list[dict[str, str]] = []
    for split_name in ("train", "test"):
        files = input_files.get(split_name, [])
        if not isinstance(files, list):
            continue
        for file in files:
            rows.append({"split": split_name, "path": str(file)})
    return rows
