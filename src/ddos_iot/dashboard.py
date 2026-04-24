"""Streamlit dashboard for the IoT DDoS detection project."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ddos_iot.paths import (
    MODEL_COMPARISON_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)


TARGET_COLUMN = "target"


st.set_page_config(
    page_title="IoT DDoS Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render the dashboard."""

    st.title("IoT DDoS Detection")
    st.caption(
        "Preprocess traffic flows, compare models, inspect SHAP explanations, and run predictions."
    )

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
    processed_train = PROCESSED_DATA_DIR / "train.csv"
    comparison_file = MODEL_COMPARISON_DIR / "model_comparison.csv"
    best_report = read_json(MODEL_COMPARISON_DIR / "model_comparison_report.json")

    columns = st.columns(4)
    columns[0].metric("Raw CSV files", len(raw_files))
    columns[1].metric("Processed train", "Ready" if processed_train.exists() else "Missing")
    columns[2].metric("Model comparison", "Ready" if comparison_file.exists() else "Missing")
    columns[3].metric("Best model", best_model_name(best_report))

    st.subheader("Project Outputs")
    output_rows = [
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
        st.dataframe(read_csv(comparison_file), use_container_width=True, hide_index=True)
    else:
        st.info("No model comparison is available yet. Run the pipeline to generate results.")


def render_pipeline_runner() -> None:
    """Render controls for running the pipeline."""

    st.subheader("Run Pipeline")
    with st.form("pipeline_form"):
        left, right = st.columns(2)
        input_path = left.text_input("Input path", value=str(RAW_DATA_DIR.relative_to(PROJECT_ROOT)))
        processed_path = left.text_input(
            "Processed path",
            value=str(PROCESSED_DATA_DIR.relative_to(PROJECT_ROOT)),
        )
        output_path = left.text_input(
            "Models output path",
            value=str(MODEL_COMPARISON_DIR.relative_to(PROJECT_ROOT)),
        )
        top_features = left.number_input("Top features", min_value=1, max_value=100, value=25, step=1)
        test_size = left.slider("Test size", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
        rebalance = left.selectbox("Training rebalance", ("undersample", "none", "oversample"))

        explain_shap = right.checkbox("Generate SHAP", value=True)
        shap_for_all = right.checkbox("SHAP for all models", value=False)
        skip_preprocess = right.checkbox("Skip preprocessing", value=False)
        no_plots = right.checkbox("Skip plots", value=False)
        max_rows = right.number_input("Max rows per raw file", min_value=0, value=0, step=1000)
        verbose = right.checkbox("Verbose logs", value=False)

        submitted = st.form_submit_button("Run")

    if submitted:
        command = [
            sys.executable,
            "run.py",
            "--input-path",
            input_path,
            "--processed-path",
            processed_path,
            "--models-output-path",
            output_path,
            "--top-features",
            str(top_features),
            "--test-size",
            str(test_size),
            "--rebalance",
            rebalance,
        ]
        if explain_shap:
            command.append("--explain-shap")
        if shap_for_all:
            command.append("--shap-for-all-models")
        if skip_preprocess:
            command.append("--skip-preprocess")
        if no_plots:
            command.append("--no-plots")
        if max_rows > 0:
            command.extend(["--max-rows-per-file", str(max_rows)])
        if verbose:
            command.append("--verbose")

        run_command(command)


def render_dataset_page() -> None:
    """Render processed dataset and preprocessing summaries."""

    st.subheader("Dataset")
    report = read_json(PROCESSED_DATA_DIR / "preprocessing_report.json")
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        st.info("Processed train/test files are missing. Run preprocessing first.")
        return

    train = read_csv(train_path)
    test = read_csv(test_path)
    columns = st.columns(4)
    columns[0].metric("Train rows", f"{len(train):,}")
    columns[1].metric("Test rows", f"{len(test):,}")
    columns[2].metric("Features", train.shape[1] - 1)
    columns[3].metric("Target", TARGET_COLUMN)

    st.subheader("Class Distribution")
    class_counts = pd.DataFrame(
        {
            "split": ["train", "test"],
            "benign": [int((train[TARGET_COLUMN] == 0).sum()), int((test[TARGET_COLUMN] == 0).sum())],
            "ddos": [int((train[TARGET_COLUMN] == 1).sum()), int((test[TARGET_COLUMN] == 1).sum())],
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

    comparison = read_csv(comparison_path)
    st.metric("Best model", best_model_name(report))
    st.dataframe(comparison, use_container_width=True, hide_index=True)

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
            index=available_metrics.index("f1"),
        )
        st.bar_chart(comparison.set_index("model")[selected_metric])

    model_dirs = available_model_dirs()
    if model_dirs:
        st.subheader("Model Details")
        model_dir = st.selectbox("Model", model_dirs, format_func=lambda path: path.name)
        render_model_detail(model_dir)


def render_model_detail(model_dir: Path) -> None:
    """Render artifacts for one model directory."""

    metrics = read_json(model_dir / "metrics.json")
    if metrics:
        test_metrics = metrics.get("test", {})
        cols = st.columns(4)
        cols[0].metric("F1", format_metric(test_metrics.get("f1")))
        cols[1].metric("Recall", format_metric(test_metrics.get("recall")))
        cols[2].metric("Precision", format_metric(test_metrics.get("precision")))
        cols[3].metric("Inference", f"{test_metrics.get('inference_time_seconds', 0):.3f}s")

    render_images(
        [
            model_dir / "confusion_matrix.png",
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
    if report:
        cols = st.columns(4)
        cols[0].metric("Method", str(report.get("method", "unknown")))
        cols[1].metric("Explained rows", report.get("explained_rows", 0))
        cols[2].metric("Background rows", report.get("background_rows", 0))
        cols[3].metric("Local rows", report.get("local_explanation_rows", 0))

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

    model_path = st.selectbox("Model", model_files, format_func=lambda path: path.parent.name)
    uploaded_file = st.file_uploader("Processed CSV", type=["csv"])
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if st.button("Predict", disabled=uploaded_file is None):
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        upload_path = PREDICTIONS_DIR / "uploaded_prediction_input.csv"
        output_path = PREDICTIONS_DIR / "ui_predictions.csv"
        upload_path.write_bytes(uploaded_file.getbuffer())
        command = [
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
        run_command(command)
        if output_path.exists():
            predictions = read_csv(output_path)
            st.dataframe(predictions.head(1000), use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions",
                data=output_path.read_bytes(),
                file_name="predictions.csv",
                mime="text/csv",
            )


def run_command(command: list[str]) -> None:
    """Run a project command and stream captured output into the UI."""

    st.code(" ".join(command), language="powershell")
    with st.spinner("Running command..."):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.stdout:
        st.text_area("Output", result.stdout, height=240)
    if result.stderr:
        st.text_area("Logs", result.stderr, height=240)
    if result.returncode == 0:
        st.success("Command finished successfully.")
    else:
        st.error(f"Command failed with exit code {result.returncode}.")


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Read JSON if it exists."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with Streamlit caching."""

    return pd.read_csv(path)


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


def format_metric(value: Any) -> str:
    """Format numeric metric values for cards."""

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "Missing"
