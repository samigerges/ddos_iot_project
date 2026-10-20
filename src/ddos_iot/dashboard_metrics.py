"""Tables and confusion matrices from saved evaluation reports, without training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure


SCORE_COLUMNS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate", "balanced_accuracy",
    "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
    "cv_macro_f1_mean", "cv_macro_f1_std", "f1-score",
)
TIMING_COLUMNS = (
    "train_time_seconds", "inference_time_seconds", "inference_time_per_row_ms",
    "training_seconds", "cv_seconds", "total_training_seconds",
    "inference_seconds", "inference_us_per_sample",
)
LABELS = {
    "model": "Model", "split": "Split", "class": "Class", "status": "Status",
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1", "f1-score": "F1", "support": "Support",
    "roc_auc": "ROC AUC", "pr_auc": "PR AUC",
    "false_positive_rate": "False positive rate", "false_negative_rate": "False negative rate",
    "balanced_accuracy": "Balanced accuracy", "macro_precision": "Macro precision",
    "macro_recall": "Macro recall", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1",
    "cv_macro_f1_mean": "CV macro F1 (mean)", "cv_macro_f1_std": "CV macro F1 (std)",
    "train_time_seconds": "Training (s)", "inference_time_seconds": "Inference (s)",
    "inference_time_per_row_ms": "Inference (ms/row)", "training_seconds": "Training (s)",
    "cv_seconds": "Cross-validation (s)", "total_training_seconds": "Total training (s)",
    "inference_seconds": "Inference (s)", "inference_us_per_sample": "Inference (us/row)",
}


def render_metric_table(frame: pd.DataFrame, columns: tuple[str, ...], *, sort_by: str | None = None) -> None:
    """Show readable numeric scores while preserving the recorded values."""
    selected = [column for column in columns if column in frame.columns]
    if not selected or frame.empty:
        st.info("No saved metrics are available yet.")
        return
    table = frame.copy()
    numeric = set(SCORE_COLUMNS + TIMING_COLUMNS + ("support",))
    for column in table.columns.intersection(list(numeric)):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False, na_position="last", kind="stable")
    config = {column: st.column_config.NumberColumn(LABELS.get(column, column),
                                                   format="%.6f" if column in SCORE_COLUMNS else "%.4f", width="small")
              if column in numeric else st.column_config.TextColumn(LABELS.get(column, column),
                                                                     width="medium" if column in ("model", "class") else "small")
              for column in selected}
    if "support" in selected:
        config["support"] = st.column_config.NumberColumn("Support", format="%d")
    st.dataframe(table[selected], column_config=config, hide_index=True, use_container_width=True)


def load_classification_table(folder: Path, split: str = "test", metrics: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load per-class precision/recall/F1, omitting sklearn's scalar accuracy row."""
    report = (metrics or {}).get("classification_report")
    if isinstance(report, dict):
        frame = pd.DataFrame.from_dict({label: row for label, row in report.items()
                                       if isinstance(row, dict)}, orient="index")
    else:
        path = folder / f"classification_report_{split}.csv"
        if not path.is_file():
            return pd.DataFrame()
        frame = pd.read_csv(path, index_col=0)
    frame = frame.drop(index="accuracy", errors="ignore")
    frame.index.name = "class"
    return frame.reset_index()


def add_multiclass_scores(frame: pd.DataFrame, run: Path) -> pd.DataFrame:
    """Add saved macro precision and recall to the model comparison."""
    frame = frame.copy()
    if "model" not in frame:
        return frame
    for index, name in frame["model"].items():
        if str(name) not in ("xgboost", "lightgbm", "catboost"):
            continue
        report = load_classification_table(run / str(name))
        if report.empty:
            continue
        macro = report.loc[report["class"] == "macro avg"]
        if not macro.empty:
            for metric in ("precision", "recall"):
                if metric in macro:
                    frame.loc[index, f"macro_{metric}"] = macro.iloc[0][metric]
    return frame


def load_confusion_table(folder: Path, metrics: dict[str, Any] | None = None, *, split: str = "test") -> pd.DataFrame:
    """Use binary TN/FP/FN/TP counts or a labeled multiclass CSV."""
    counts = (metrics or {}).get("confusion_matrix")
    if isinstance(counts, dict) and all(key in counts for key in ("tn", "fp", "fn", "tp")):
        frame = pd.DataFrame([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
                             index=["Benign", "DDoS"], columns=["Benign", "DDoS"])
    else:
        path = folder / "confusion_matrix.csv"
        if split != "test" or not path.is_file():
            return pd.DataFrame()
        frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates or set(frame.index) != set(frame.columns):
        raise ValueError("Saved confusion matrix must have matching actual and predicted class labels.")
    frame = frame.loc[frame.columns].apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Saved confusion matrix must contain nonnegative whole-number counts.")
    frame = frame.astype("int64")
    frame.index.name = "Actual class"
    frame.columns.name = "Predicted class"
    return frame


def normalize_confusion_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize by actual-class support; classes with no samples remain zero."""
    return frame.div(frame.sum(axis=1).replace(0, 1), axis=0)


def confusion_figure(frame: pd.DataFrame, normalized: bool) -> Figure:
    """Label both axes explicitly and keep large class sets readable."""
    shown = normalize_confusion_table(frame) if normalized else frame
    count = len(frame)
    size = max(5, min(17, 4 + count * 0.35))
    figure = Figure(figsize=(size + 1, size), layout="constrained")
    ax = figure.subplots()
    plot = ax.imshow(shown.to_numpy(), cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set_xticks(range(count), labels=frame.columns, rotation=90 if count > 10 else 45, ha="right")
    ax.set_yticks(range(count), labels=frame.index)
    ax.tick_params(labelsize=7 if count > 10 else 9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion matrix (row-normalized)" if normalized else "Confusion matrix (counts)")
    figure.colorbar(plot, ax=ax, label="Share of actual class" if normalized else "Number of flows", shrink=0.8)
    if count <= 10:
        threshold = float(shown.to_numpy().max()) / 2
        for row in range(count):
            for column in range(count):
                value = shown.iloc[row, column]
                ax.text(column, row, f"{value:.1%}" if normalized else f"{value:,}",
                        ha="center", va="center", fontsize=8,
                        color="white" if value > threshold else "black")
    return figure


def render_evaluation_tables(folder: Path, *, key: str, metrics: dict[str, Any] | None = None,
                             split: str = "test") -> None:
    """Display stored class metrics, a heatmap, and its exact count table."""
    st.subheader(f"Per-class metrics ({split})")
    try:
        report = load_classification_table(folder, split, metrics)
        render_metric_table(report, ("class", "precision", "recall", "f1-score", "support"))
        matrix = load_confusion_table(folder, metrics, split=split)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Unable to read the saved evaluation report: {error}")
        return
    st.subheader(f"Confusion matrix ({split})")
    if matrix.empty:
        st.info("No saved confusion matrix is available for this model yet.")
        return
    normalized = st.radio("Matrix values", ("Counts", "Row percentages"), horizontal=True,
                          key=f"{key}_matrix_values") == "Row percentages"
    st.caption("Rows are actual classes; columns are predicted classes. Row percentages show recall within each actual class.")
    st.pyplot(confusion_figure(matrix, normalized), clear_figure=True)
    shown = normalize_confusion_table(matrix) if normalized else matrix
    config = {column: st.column_config.NumberColumn(column, format="%d")
              for column in shown.columns}
    # Percentage formatting uses a Styler so underlying 0..1 values stay intact.
    display = shown.style.format("{:.2%}") if normalized else shown
    st.dataframe(display, column_config=None if normalized else config, use_container_width=True)
    st.download_button("Download confusion counts (CSV)", matrix.to_csv().encode("utf-8"),
                       file_name=f"{folder.parent.name}_{folder.name}_{split}_confusion_matrix.csv",
                       mime="text/csv", key=f"{key}_matrix_download")
