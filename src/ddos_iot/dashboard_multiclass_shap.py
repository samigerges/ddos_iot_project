"""Class-aware SHAP generation and viewing, independent of training pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ddos_iot.dashboard_multiclass import display_path, resolve_path, show_csv
from ddos_iot.dashboard_runs import read_report, saved_models
from ddos_iot.paths import RAW_TEST_FILE


def render_shap_page(num_classes: int, run: Path | None) -> None:
    st.subheader(f"{num_classes}-Class SHAP Explanations")
    choices = saved_models(run) if run is not None else []
    if not choices:
        st.info("No saved model is available for this experiment. Train once, then generate SHAP without retraining.")
        return
    prefix = f"mc_{num_classes}_shap"
    model = st.selectbox("Model to explain", choices, format_func=lambda p: p.parent.name, key=f"{prefix}_model")
    st.caption("Only trusted local model bundles. SHAP reuses the saved features, scaler, and class order; "
               "no training, tuning, or GPU work is performed.")
    with st.expander("Generate new explanations", expanded=True):
        input_value = st.text_input("Raw CSV to explain", display_path(RAW_TEST_FILE), key=f"{prefix}_input")
        sample_size = st.number_input("SHAP sample rows", min_value=1, max_value=5000, value=200, key=f"{prefix}_size")
        scan = st.number_input("Rows scanned per CSV (0 = full file)", min_value=0, value=50_000, step=10_000, key=f"{prefix}_scan")
        st.caption("The default samples valid rows from the first 50,000 data rows, not the whole dataset. "
                   "Rare classes may be absent. Set the scan limit to 0 for a full-file random sample; reading may take longer. "
                   "These explanations describe the chosen input, not a new held-out evaluation.")
        method = st.selectbox("SHAP method", ("auto", "native", "permutation"), key=f"{prefix}_method")
        background = st.text_input("Optional separate training-background CSV", "", key=f"{prefix}_background")
        st.caption("Native TreeSHAP uses training tree statistics. A training-background CSV is needed only for "
                   "permutation SHAP or fallback; never substitute the explained/test CSV.")
        with st.expander("Advanced SHAP settings"):
            bg_size = st.number_input("Background rows", min_value=1, max_value=500, value=50, key=f"{prefix}_bgsize")
            fallback_size = st.number_input("Permutation maximum rows", min_value=1, max_value=200, value=20, key=f"{prefix}_fallback")
            local = st.number_input("Exported local waterfall plots", min_value=0, max_value=50, value=5, key=f"{prefix}_local")
            display = st.number_input("Top displayed features", min_value=1, max_value=46, value=15, key=f"{prefix}_display")
            threads = st.number_input("SHAP CPU threads", min_value=1, max_value=32, value=2, key=f"{prefix}_threads")
        if st.button("Generate SHAP without retraining", key=f"{prefix}_generate"):
            try:
                from ddos_iot.multiclass_shap import run_multiclass_shap

                with st.spinner("Computing sampled multiclass SHAP on CPU..."):
                    report = run_multiclass_shap(model, resolve_path(input_value),
                        sample_size=int(sample_size), max_scan_rows=int(scan), method=method,
                        background_file=resolve_path(background.strip()) if background.strip() else None,
                        background_size=int(bg_size), fallback_size=int(fallback_size), local_count=int(local),
                        max_display=int(display), n_jobs=int(threads), expected_num_classes=num_classes)
                st.session_state[f"{prefix}_run"] = Path(report["output_path"])
                st.success(f"Saved SHAP for {report['explained_rows']} rows and all {num_classes} classes.")
            except Exception as exc:
                st.error(f"SHAP failed: {exc}. Saved models and training results were not changed.")
    root = model.parent / "shap"
    folders = sorted((p for p in root.iterdir() if p.is_dir()) , reverse=True) if root.exists() else []
    reports = {p: read_report(p / "shap_report.json") for p in folders}
    complete = [p for p in folders if reports[p].get("status") == "complete" and reports[p].get("class_count") == num_classes]
    for folder in folders:
        if reports[folder].get("status") in ("failed", "interrupted"):
            st.warning(f"{folder.name}: {reports[folder].get('error', 'SHAP did not finish')}")
    if not complete:
        st.info("No completed SHAP report for this model yet. Generate one above; existing models do not need retraining.")
        return
    selected = st.selectbox("Saved SHAP run", complete, format_func=lambda p: p.name, key=f"{prefix}_run")
    render_saved_shap(selected, reports[selected], prefix)


def render_saved_shap(folder: Path, report: dict, prefix: str) -> None:
    from ddos_iot.multiclass_shap_plots import (
        PLOT_LOCK, beeswarm_figure, importance_figure, load_arrays, plt, waterfall_figure,
    )

    st.caption(f"{report['model']} · {report['method']} · {report['explained_rows']} explained rows · "
               f"maximum additivity error: {report.get('max_additivity_error', 0):.3g}")
    space = report["output_space"]
    if space == "raw_margin":
        st.info("SHAP contributions add to each class's raw score (logit), NOT directly to its probability. "
                "Softmax across all class scores produces probabilities. Positive values raise the selected class score.")
    else:
        st.info("These permutation contributions add to the selected class probability. "
                "Their magnitudes are not directly comparable with raw-score TreeSHAP values.")
    if report.get("sampling", {}).get("sampling_scope") == "prefix":
        st.warning("This report samples a CSV prefix, not the full dataset. Interpret overall importance as sample-specific.")
    try:
        data = load_arrays(folder)
        class_id = st.selectbox("Class to explain", list(range(len(data["classes"]))),
                                 format_func=lambda i: str(data["classes"][i]), key=f"{prefix}_class")
        st.subheader("Global Feature Importance")
        st.caption("All-class summary averages |SHAP| over sampled rows and output classes; it is not class-conditional importance.")
        with PLOT_LOCK:
            figures = []
            try:
                for figure in (importance_figure(data, output_space=space),
                               importance_figure(data, class_id, output_space=space),
                               beeswarm_figure(data, class_id, output_space=space)):
                    figures.append(figure)
                    st.pyplot(figure)
            finally:
                for figure in figures:
                    plt.close(figure)
        st.subheader("Individual Prediction")
        row = st.selectbox("Sample row (zero-based CSV data row)", list(range(len(data["row_indices"]))),
                           format_func=lambda i: str(data["row_indices"][i]), key=f"{prefix}_row")
        probabilities = data["probabilities"][row]
        predicted = int(probabilities.argmax())
        st.write(f"Predicted: {data['classes'][predicted]} ({probabilities[predicted]:.2%}). "
                 f"Selected class: {data['classes'][class_id]} ({probabilities[class_id]:.2%}).")
        st.caption("The waterfall explains the selected class—even when it is not the predicted class. "
                   "Displayed feature values are in the original raw units.")
        with PLOT_LOCK:
            figure = waterfall_figure(data, row, class_id, output_space=space)
            try:
                st.pyplot(figure)
            finally:
                plt.close(figure)
        with st.expander("Selected-class contribution table"):
            table = pd.DataFrame({"feature": data["feature_names"], "raw_value": data["raw_values"][row],
                                  "shap_value": data["values"][row, :, class_id]})
            table = table.iloc[np.argsort(np.abs(table.shap_value.to_numpy()))[::-1]]
            st.dataframe(table, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not display SHAP arrays/plots: {exc}")
    with st.expander("Tables and report"):
        show_csv(folder / "shap_class_importance.csv")
        st.json(report)
        for filename in ("shap_global_importance.csv", "shap_class_importance.csv", "shap_local_top_features.csv", "shap_report.json", "shap_values.npz"):
            path = folder / filename
            if path.is_file() and path.stat().st_size <= 50_000_000:
                st.download_button(f"Download {filename}", path.read_bytes(), file_name=filename, key=f"{prefix}_download_{filename}")
