"""SHAP explanation helpers for fitted DDoS classifiers."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

from ddos_iot.paths import configure_runtime_environment

configure_runtime_environment()
matplotlib.use("Agg")

from matplotlib import pyplot as plt

from ddos_iot.utils import predict_scores, write_json


LOGGER = logging.getLogger(__name__)


TREE_MODEL_NAMES = {
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "HistGradientBoostingClassifier",
}


def save_shap_explanations(
    model: Any,
    model_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    background_size: int,
    local_count: int,
    max_display: int,
    model_agnostic_sample_size: int,
    random_state: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Create and save SHAP global and local explanations for one fitted model."""

    import shap

    output_dir = output_path / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    feature_columns = list(x_train.columns)
    background = sample_frame(x_train, background_size, random_state)
    explain_data = sample_frame(x_test, sample_size, random_state)

    explanation, method, explain_data = build_shap_explanation(
        shap=shap,
        model=model,
        x_background=background,
        x_explain=explain_data,
        feature_columns=feature_columns,
        model_agnostic_sample_size=model_agnostic_sample_size,
        random_state=random_state,
    )
    shap_values = class_one_values(explanation.values)
    base_values = class_one_base_values(explanation.base_values)

    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_columns):
        raise ValueError(f"Unexpected SHAP value shape: {shap_values.shape}")

    save_global_shap_outputs(
        shap=shap,
        shap_values=shap_values,
        x_explain=explain_data,
        output_dir=output_dir,
        max_display=max_display,
    )
    save_local_shap_outputs(
        shap=shap,
        model=model,
        shap_values=shap_values,
        base_values=base_values,
        x_explain=explain_data,
        output_dir=output_dir,
        local_count=local_count,
        max_display=max_display,
        threshold=threshold,
    )

    report = {
        "model": model_name,
        "method": method,
        "feature_count": len(feature_columns),
        "background_rows": int(len(background)),
        "explained_rows": int(len(explain_data)),
        "local_explanation_rows": int(min(local_count, len(explain_data))),
        "elapsed_seconds": float(time.perf_counter() - start),
        "outputs": [
            "shap_global_importance.csv",
            "shap_values_sample.csv",
            "shap_summary_bar.png",
            "shap_beeswarm.png",
            "shap_local_top_features.csv",
            "shap_local_waterfall_*.png",
        ],
    }
    write_json(output_dir / "shap_report.json", report)
    LOGGER.info("Saved SHAP explanations for %s to %s", model_name, output_dir)
    return report


def save_prediction_local_shap_explanations(
    model: Any,
    model_name: str,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    max_display: int,
    model_agnostic_sample_size: int,
    random_state: int,
    target: pd.Series | None = None,
    waterfall_count: int | None = None,
) -> dict[str, Any]:
    """Create and save local SHAP explanations for prediction-time inputs."""

    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    feature_columns = list(x_background.columns)
    if feature_columns != list(x_explain.columns):
        raise ValueError("Background and explain-data feature columns do not match for prediction SHAP.")

    effective_sample_size = (
        len(x_explain) if model_agnostic_sample_size <= 0 else model_agnostic_sample_size
    )
    explanation, method, explain_data = build_shap_explanation(
        shap=shap,
        model=model,
        x_background=x_background,
        x_explain=x_explain,
        feature_columns=feature_columns,
        model_agnostic_sample_size=effective_sample_size,
        random_state=random_state,
    )
    shap_values = class_one_values(explanation.values)
    base_values = class_one_base_values(explanation.base_values)
    aligned_target = target.loc[explain_data.index] if target is not None else None
    plotted_rows = save_local_shap_outputs(
        shap=shap,
        model=model,
        shap_values=shap_values,
        base_values=base_values,
        x_explain=explain_data,
        output_dir=output_dir,
        local_count=len(explain_data),
        max_display=max_display,
        threshold=threshold,
        target=aligned_target,
        plot_count=waterfall_count,
    )

    requested_rows = int(len(x_explain))
    explained_rows = int(len(explain_data))
    waterfall_rows = explained_rows if waterfall_count is None or waterfall_count <= 0 else min(
        int(waterfall_count),
        explained_rows,
    )
    report = {
        "model": model_name,
        "method": method,
        "feature_count": len(feature_columns),
        "background_rows": int(len(x_background)),
        "requested_rows": requested_rows,
        "explained_rows": explained_rows,
        "local_explanation_rows": explained_rows,
        "waterfall_plot_rows": waterfall_rows,
        "has_ground_truth": bool(target is not None),
        "elapsed_seconds": float(time.perf_counter() - start),
        "outputs": [
            "shap_local_top_features.csv",
            "shap_local_waterfall_*.png",
        ],
    }
    write_json(output_dir / "shap_report.json", report)
    LOGGER.info(
        "Saved prediction-time local SHAP explanations for %s to %s (%s rows)",
        model_name,
        output_dir,
        len(plotted_rows["row_position"].unique()) if not plotted_rows.empty else 0,
    )
    return report


def build_shap_explanation(
    shap: Any,
    model: Any,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
    feature_columns: list[str],
    model_agnostic_sample_size: int,
    random_state: int,
) -> tuple[Any, str, pd.DataFrame]:
    """Build SHAP values, preferring fast tree explainers and falling back safely."""

    if model.__class__.__name__ in TREE_MODEL_NAMES:
        try:
            explainer = shap.Explainer(model, x_background)
            return explainer(x_explain), "tree", x_explain
        except Exception:
            LOGGER.exception("Tree SHAP failed; falling back to model-agnostic SHAP.")

    capped_explain = sample_frame(x_explain, model_agnostic_sample_size, random_state)

    def score_ddos(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=feature_columns)
        return predict_scores(model, frame)

    explainer = shap.Explainer(score_ddos, x_background, algorithm="permutation")
    max_evals = max(2 * len(feature_columns) + 1, 100)
    return explainer(capped_explain, max_evals=max_evals), "model_agnostic_permutation", capped_explain


def sample_frame(data: pd.DataFrame, size: int, random_state: int) -> pd.DataFrame:
    """Return a deterministic sample without changing all rows when size is non-positive."""

    if size <= 0 or len(data) <= size:
        return data.copy()
    return data.sample(n=size, random_state=random_state).sort_index()


def class_one_values(values: Any) -> np.ndarray:
    """Extract class-1 SHAP values from common binary-classification shapes."""

    if isinstance(values, list):
        selected = values[1] if len(values) > 1 else values[0]
        return np.asarray(selected, dtype=float)

    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            return array[:, :, 1]
        if array.shape[0] == 2:
            return array[1, :, :]
    return array


def class_one_base_values(base_values: Any) -> np.ndarray | float:
    """Extract class-1 SHAP base values from common binary-classification shapes."""

    array = np.asarray(base_values, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return float(array[1]) if array.shape[0] == 2 else array
    if array.ndim == 2 and array.shape[1] == 2:
        return array[:, 1]
    return array


def save_global_shap_outputs(
    shap: Any,
    shap_values: np.ndarray,
    x_explain: pd.DataFrame,
    output_dir: Path,
    max_display: int,
) -> None:
    """Save global SHAP importance tables and summary plots."""

    importance = pd.DataFrame(
        {
            "feature": x_explain.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.insert(0, "rank", range(1, len(importance) + 1))
    importance.to_csv(output_dir / "shap_global_importance.csv", index=False)

    shap_sample = pd.DataFrame(shap_values, columns=x_explain.columns)
    shap_sample.insert(0, "sample_index", x_explain.index.to_numpy())
    shap_sample.to_csv(output_dir / "shap_values_sample.csv", index=False)

    shap.summary_plot(shap_values, x_explain, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, x_explain, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_local_shap_outputs(
    shap: Any,
    model: Any,
    shap_values: np.ndarray,
    base_values: np.ndarray | float,
    x_explain: pd.DataFrame,
    output_dir: Path,
    local_count: int,
    max_display: int,
    threshold: float = 0.5,
    target: pd.Series | None = None,
    plot_count: int | None = None,
) -> pd.DataFrame:
    """Save row-level SHAP explanations and waterfall plots."""

    local_rows = min(local_count, len(x_explain))
    if local_rows <= 0:
        empty = pd.DataFrame(
            columns=[
                "sample_index",
                "row_position",
                "y_true",
                "y_score",
                "y_pred",
                "feature",
                "feature_value",
                "shap_value",
                "abs_shap_value",
            ]
        )
        empty.to_csv(output_dir / "shap_local_top_features.csv", index=False)
        return empty

    waterfall_rows = (
        local_rows if plot_count is None or plot_count <= 0 else min(int(plot_count), local_rows)
    )
    scores = predict_scores(model, x_explain.iloc[:local_rows])
    target_values = target.to_numpy(dtype=int)[:local_rows] if target is not None else None
    records: list[dict[str, Any]] = []

    for row_position in range(local_rows):
        row_values = shap_values[row_position]
        top_positions = np.argsort(np.abs(row_values))[::-1][:max_display]
        sample_index = x_explain.index[row_position]
        for feature_position in top_positions:
            feature = x_explain.columns[feature_position]
            record = {
                "sample_index": int(sample_index),
                "row_position": int(row_position),
                "y_score": float(scores[row_position]),
                "y_pred": int(scores[row_position] >= threshold),
                "feature": feature,
                "feature_value": float(x_explain.iloc[row_position, feature_position]),
                "shap_value": float(row_values[feature_position]),
                "abs_shap_value": float(abs(row_values[feature_position])),
            }
            if target_values is not None:
                record["y_true"] = int(target_values[row_position])
            records.append(record)

        if row_position < waterfall_rows:
            base_value = row_base_value(base_values, row_position)
            explanation = shap.Explanation(
                values=row_values,
                base_values=base_value,
                data=x_explain.iloc[row_position].to_numpy(),
                feature_names=list(x_explain.columns),
            )
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(
                output_dir / f"shap_local_waterfall_{row_position:03d}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close()

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "shap_local_top_features.csv", index=False)
    return frame


def row_base_value(base_values: np.ndarray | float, row_position: int) -> float:
    """Return one scalar base value for a local explanation."""

    if isinstance(base_values, float):
        return base_values
    array = np.asarray(base_values, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        if len(array) == 1:
            return float(array[0])
        return float(array[row_position])
    return float(np.ravel(array)[0])
