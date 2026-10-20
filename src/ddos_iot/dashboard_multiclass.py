"""Dashboard views for the shared eight-family and 34-label training pipeline."""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from ddos_iot.attack_families import FAMILY_MEMBERS
from ddos_iot.dashboard_metrics import add_multiclass_scores, render_evaluation_tables, render_metric_table
from ddos_iot.dashboard_runs import (
    MODEL_NAMES, build_training_command, discover_runs, read_report,
    saved_models, start_training_job, tail_log,
)
from ddos_iot.paths import MODELS_DIR, PREDICTIONS_DIR, PROJECT_ROOT, RAW_TEST_FILE, RAW_TRAIN_FILE


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def show_csv(path: Path, *, limit: int | None = None) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, nrows=limit)
    except (OSError, ValueError, pd.errors.ParserError):
        st.info(f"Not available yet: {path.name}. Refresh after the next training checkpoint.")
        return pd.DataFrame()
    st.dataframe(frame, use_container_width=True, hide_index=True)
    return frame


def render_multiclass(num_classes: int) -> None:
    """Task-specific navigation; binary pages never consume multiclass artifacts."""
    st.caption(f"{num_classes}-class workflow · XGBoost, LightGBM, CatBoost · saved models are reusable")
    page = st.sidebar.radio("View", ("Overview", "Run Pipeline", "Dataset", "Model Comparison", "SHAP", "Predict"),
                            key=f"mc_{num_classes}_page")
    runs = discover_runs(num_classes)
    run = st.sidebar.selectbox("Experiment", runs, format_func=lambda p: p.name,
                               key=f"mc_{num_classes}_run") if runs else None
    st.sidebar.button("Refresh reports", key=f"mc_{num_classes}_refresh")
    st.sidebar.caption("Reports are read fresh on refresh. A reported 'running' status is not a live process check.")
    if page == "Run Pipeline":
        render_training(num_classes)
    elif page == "Predict":
        render_prediction(num_classes, runs)
    elif page == "Dataset":
        render_dataset(num_classes, run)
    elif page == "Model Comparison":
        render_comparison(run)
    elif page == "SHAP":
        from ddos_iot.dashboard_multiclass_shap import render_shap_page

        render_shap_page(num_classes, run)
    else:
        render_overview(num_classes, run, runs)


def render_overview(num_classes: int, run: Path | None, runs: list[Path]) -> None:
    st.subheader(f"{num_classes}-Class Overview")
    columns = st.columns(3)
    columns[0].metric("Experiments", len(runs))
    columns[1].metric("Saved models", sum(len(saved_models(p)) for p in runs))
    columns[2].metric("Classes", num_classes)
    st.markdown("Raw CSV → class mapping/sampling → gain above the median of all 46 features → "
                "scaling → five-fold grid search → final fit/refit → **save model** → held-out evaluation")
    if num_classes == 8:
        st.info("Default: paper-style proportional sampling with family minimums from the training source, "
                "then a new 80/20 split. The existing test CSV is not used. Preprocessing is fitted once "
                "on the training split before CV; fold-local preprocessing is also available.")
    else:
        st.info("Default: use all valid rows from the existing raw train/test split; fit preprocessing "
                "inside CV folds. The held-out test set is never used for tuning.")
    st.caption("Older experiments can have different settings. Inspect their stored configuration, not current defaults. "
               "Paper-equivalent scores are not guaranteed. Use SHAP to explain a saved model without retraining.")
    if run is None:
        st.info("No experiment exists yet. Use Run Pipeline to train the first models.")
        return
    render_run_status(run)


def render_run_status(run: Path) -> None:
    st.subheader(f"Experiment: {run.name}")
    comparison = read_report(run / "model_comparison_report.json")
    job = read_report(run / "dashboard_job.json")
    st.write(f"Last reported status: {job.get('status', comparison.get('status', 'counts/preparation only'))}")
    st.caption(str(run))
    if job.get("error"):
        st.error(job["error"])
    if comparison.get("failures"):
        st.warning("Some models failed. Any successfully saved model remains available for prediction.")
        st.json(comparison["failures"])
    rows = []
    for name in MODEL_NAMES:
        folder = run / name
        search = read_report(folder / "grid_search_report.json")
        save = read_report(folder / "model_save_report.json")
        metrics = read_report(folder / "metrics.json")
        if not folder.exists():
            continue
        candidates = search.get("candidates", [])
        completed = sum(len(c.get("fold_scores", [])) for c in candidates)
        rows.append({"model": name, "saved_model": (folder / "model.joblib").is_file(),
                     "evaluation": save.get("evaluation_status", "complete" if metrics else "not reported"),
                     "CV status": search.get("status", "not used / not started"),
                     "CV evaluations recorded": f"{completed}/{len(candidates) * 5}" if candidates else "—",
                     "current fold": search.get("current_fold"),
                     "current candidate": search.get("current_candidate")})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No final model is saved yet. Data preparation and CV do not produce a reusable final model.")
    log = tail_log(run / "dashboard_training.log")
    if log:
        with st.expander("Training log (latest lines)"):
            st.code(log, language="text")
    if (run / "model_comparison.csv").exists():
        show_csv(run / "model_comparison.csv")


def render_training(num_classes: int) -> None:
    st.subheader(f"Train {num_classes} Classes")
    st.caption("Launching is explicit. Existing experiments are never overwritten or automatically retrained. "
               "A background worker keeps running when you switch dashboard pages.")
    prefix = f"mc_{num_classes}_train"
    with st.form(prefix):
        models = st.multiselect("Models", MODEL_NAMES, default=list(MODEL_NAMES))
        device = st.selectbox("Training device", ("gpu", "cpu"))
        lightgbm_device = st.selectbox("LightGBM device", ("cpu", "gpu"),
                                      help="CPU avoids the OpenCL failure previously seen on this system.")
        st.caption("GPU: XGBoost/CatBoost fits use GPU; XGBoost CV prediction uses CUDA. "
                   "CatBoost multiclass scoring, feature selection, CSV work, and final inference use CPU.")
        protocol = st.selectbox("Data protocol", ("paper", "existing") if num_classes == 8 else ("existing",))
        train_path = st.text_input("Raw training/source CSV", display_path(RAW_TRAIN_FILE))
        test_path = st.text_input("Raw test CSV (existing-split protocol only)", display_path(RAW_TEST_FILE))
        if num_classes == 8:
            st.caption("Paper mode reads only the training/source CSV and creates a new 80/20 split. "
                       "To use the supplied train/test split, choose existing. Form changes apply on submit.")
            sampling_target = st.number_input("Paper sampling target", min_value=8, value=5_000_000, step=1000)
            minimum = st.number_input("Paper minimum samples per family", min_value=1, value=1000, step=100)
        cap = st.number_input("Training cap per class (existing protocol; 0 = all)", min_value=0, value=0, step=1000)
        cv = st.checkbox("Five-fold grid-search tuning", value=True)
        cv_preprocessing = st.selectbox("CV preprocessing", ("train", "fold") if num_classes == 8 else ("fold", "train"))
        st.caption("train follows the paper but can give optimistic CV scores; fold fits preprocessing within each fold.")
        estimators = st.number_input("Maximum boosting rounds", min_value=1, value=500, step=50)
        grid = st.text_input("Optional parameter-grid JSON path", "")
        gpu_id = st.number_input("GPU ID", min_value=0, value=0, step=1)
        no_plots = st.checkbox("Skip confusion-matrix plots", value=False)
        run_name = st.text_input("New experiment folder name", f"multiclass_{num_classes}_{datetime.now():%Y%m%d_%H%M%S}")
        acknowledged = st.checkbox("I understand this starts a new training run and may compete with any running job.")
        submitted = st.form_submit_button("Start multiclass training")
    if not submitted:
        return
    if not acknowledged:
        st.warning("Confirm the new training run before starting.")
        return
    if not models:
        st.error("Select at least one model.")
        return
    if not run_name.strip() or Path(run_name).name != run_name or any(c in run_name for c in '/\\:') or run_name in (".", ".."):
        st.error("Use a folder name, not a path. Runs are stored directly under models/.")
        return
    options = ["--models", *models, "--device", device, "--lightgbm-device", lightgbm_device,
               "--gpu-id", str(gpu_id), "--data-protocol", protocol,
               "--train-input-path", str(resolve_path(train_path)), "--test-input-path", str(resolve_path(test_path)),
               "--output-path", str(MODELS_DIR / run_name), "--cv-folds", "5" if cv else "0",
               "--cv-preprocessing", cv_preprocessing, "--n-estimators", str(estimators)]
    if protocol == "paper":
        if cap:
            st.error("Training caps apply only to existing-split mode; use the paper sampling target instead.")
            return
        options += ["--sampling-target", str(sampling_target), "--min-samples-per-class", str(minimum)]
    else:
        options += ["--max-train-per-class", str(cap)]
    if grid.strip():
        options += ["--param-grid", str(resolve_path(grid.strip()))]
    if no_plots:
        options.append("--no-plots")
    try:
        if not resolve_path(train_path).is_file() or (protocol == "existing" and not resolve_path(test_path).is_file()):
            raise ValueError("The required raw CSV input files must exist.")
        if grid.strip() and (not cv or not resolve_path(grid.strip()).is_file()):
            raise ValueError("A custom grid requires five-fold tuning and an existing JSON file.")
        command = build_training_command(num_classes, options)
        folder = start_training_job(command)
    except SystemExit:
        st.error("Invalid training settings. Check sampling minimums and the selected protocol.")
        return
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    st.success(f"Started {folder.name}. Select the experiment in Overview and refresh to inspect progress.")
    st.code(subprocess.list2cmdline(command), language="powershell")


def render_dataset(num_classes: int, run: Path | None) -> None:
    st.subheader(f"{num_classes}-Class Dataset")
    if num_classes == 8:
        with st.expander("Exact family mapping", expanded=True):
            st.dataframe(pd.DataFrame([{"family": family, "raw_label": raw}
                                      for family, members in FAMILY_MEMBERS.items() for raw in members]),
                         use_container_width=True, hide_index=True)
    if run is None:
        st.info("Class counts and selected features appear after data preparation.")
        return
    counts = run / "class_counts.csv"
    if not counts.exists():
        counts = run / "class_counts_raw.csv"
    st.markdown("**Samples per class**")
    show_csv(counts)
    selection = read_report(run / "feature_selection_report.json")
    if selection:
        st.write(f"Selected {selection.get('selected_feature_count')} of {selection.get('original_feature_count')} features")
        st.caption(f"Gain threshold: {selection.get('gain_threshold')} · rule: {selection.get('comparison')}")
    if (run / "feature_selection_scores.csv").exists():
        show_csv(run / "feature_selection_scores.csv")
    for filename in ("data_report.json", "source_inventory.json", "feature_selection_report.json"):
        report = read_report(run / filename)
        if report:
            with st.expander(filename):
                st.json(report)


def render_comparison(run: Path | None) -> None:
    st.subheader("Multiclass Model Comparison")
    if run is None:
        st.info("Train an experiment first to compare models.")
        return
    report = read_report(run / "model_comparison_report.json")
    st.caption(f"Best model: {report.get('best_model') or 'not selected yet'} · "
               f"selection criterion: {report.get('selection_metric', 'see experiment metadata')}")
    try:
        frame = add_multiclass_scores(pd.read_csv(run / "model_comparison.csv"), run)
    except (OSError, ValueError, pd.errors.ParserError):
        frame = pd.DataFrame()
    st.subheader("Test metrics by model")
    st.caption("Scores range from 0 to 1. Macro metrics weight each class equally; weighted F1 uses class support.")
    render_metric_table(frame, ("model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                                "balanced_accuracy", "cv_macro_f1_mean", "cv_macro_f1_std"),
                        sort_by=report.get("selection_metric", "macro_f1"))
    st.subheader("Timing comparison")
    render_metric_table(frame, ("model", "training_seconds", "cv_seconds", "total_training_seconds",
                                "inference_seconds", "inference_us_per_sample"))
    if not frame.empty:
        with st.expander("All recorded model metrics"):
            st.dataframe(frame, use_container_width=True, hide_index=True)
    metrics = [c for c in ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1",
                           "cv_macro_f1_mean", "training_seconds", "inference_us_per_sample")
               if c in frame.columns and frame[c].notna().any()]
    if metrics and "model" in frame:
        metric = st.selectbox("Chart metric", metrics, key="mc_chart_metric")
        st.bar_chart(frame.set_index("model")[metric])
    names = [name for name in MODEL_NAMES if (run / name).is_dir()]
    if not names:
        st.info("No model reports yet.")
        return
    winner = report.get("best_model")
    name = st.selectbox("Model details", names, index=names.index(winner) if winner in names else 0,
                        key="mc_model_details")
    folder = run / name
    st.write(f"Saved bundle: {'available' if (folder / 'model.joblib').is_file() else 'not yet available'}")
    render_evaluation_tables(folder, key=f"mc_{run.name}_{name}")
    for filename in ("grid_search_results.csv",):
        if (folder / filename).is_file():
            with st.expander(filename):
                show_csv(folder / filename)
    for filename in ("metrics.json", "model_metadata.json", "model_save_report.json", "grid_search_report.json"):
        report = read_report(folder / filename)
        if report:
            with st.expander(filename):
                st.json(report)


def render_prediction(num_classes: int, runs: list[Path]) -> None:
    st.subheader(f"Predict with a Saved {num_classes}-Class Model")
    choices = [model for run in runs for model in saved_models(run)]
    if not choices:
        st.info("No saved model is available for this task yet. Train once, then return here; prediction never trains.")
        return
    prefix = f"mc_{num_classes}_predict"
    model = st.selectbox("Saved model", choices, format_func=lambda p: f"{p.parent.parent.name} / {p.parent.name}",
                         key=f"{prefix}_model")
    st.caption("Load only trusted local model.joblib files. Provide raw numeric features, not binary processed data. "
               "The saved columns, scaler, and label encoder are reused on CPU. An input label column is ignored.")
    source = st.radio("Input source", ("Local CSV path", "Upload CSV"), key=f"{prefix}_source")
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Raw feature CSV", type=["csv"], key=f"{prefix}_upload")
        input_path = None
    else:
        input_path = resolve_path(st.text_input("Raw prediction CSV path", display_path(RAW_TEST_FILE), key=f"{prefix}_input"))
    chunk = st.number_input("Prediction chunk size", min_value=1, value=100_000, step=10_000, key=f"{prefix}_chunk")
    if st.button("Predict without retraining", key=f"{prefix}_submit",
                 disabled=source == "Upload CSV" and uploaded is None):
        st.session_state.pop(f"{prefix}_result", None)
        output = PREDICTIONS_DIR / f"multiclass_{num_classes}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.csv"
        try:
            from ddos_iot.multiclass_inference import run_multiclass_prediction

            with st.spinner("Loading the saved model and predicting; no training or CV..."):
                if uploaded is not None:
                    temp_root = PROJECT_ROOT / "tmp"
                    temp_root.mkdir(exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix="dashboard_upload_", dir=temp_root) as temporary:
                        upload_path = Path(temporary) / "input.csv"
                        upload_path.write_bytes(uploaded.getbuffer())
                        result = run_multiclass_prediction(model, upload_path, output, int(chunk), expected_num_classes=num_classes)
                else:
                    result = run_multiclass_prediction(model, input_path, output, int(chunk), expected_num_classes=num_classes)
            st.session_state[f"{prefix}_result"] = result
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    result = st.session_state.get(f"{prefix}_result")
    if result:
        st.success(f"Predicted {result['rows']:,} rows using {result['model']} ({result['class_count']} classes).")
        st.caption(f"Model used: {result['model_path']}")
        output = Path(result["output_file"])
        st.caption(f"Saved: {output}")
        show_csv(output, limit=1000)
        st.caption("Preview shows up to 1,000 rows. The export contains every row, in input order.")
        if output.is_file() and output.stat().st_size <= 50_000_000:
            st.download_button("Download multiclass predictions", output.read_bytes(), file_name=output.name,
                               mime="text/csv", key=f"{prefix}_download")
        else:
            st.info("Large exports are available at the saved path above.")
