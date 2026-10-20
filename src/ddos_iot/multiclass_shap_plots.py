"""Plots from saved multiclass SHAP arrays; no estimator loading or computation."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

import matplotlib
import numpy as np

from ddos_iot.paths import configure_runtime_environment

configure_runtime_environment()
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Matplotlib's stateful SHAP plots must not interleave across dashboard sessions.
PLOT_LOCK = RLock()


def load_arrays(folder: Path) -> dict[str, np.ndarray]:
    with np.load(folder / "shap_values.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    n, f, k = len(data["row_indices"]), len(data["feature_names"]), len(data["classes"])
    if data["values"].shape != (n, f, k) or data["base_values"].shape != (n, k):
        raise ValueError("Invalid saved multiclass SHAP array dimensions.")
    return data


def importance_figure(data: dict[str, np.ndarray], class_id: int | None = None,
                      max_display: int = 15, output_space: str = "raw_margin"):
    values = data["values"]
    scores = np.abs(values).mean(axis=(0, 2)) if class_id is None else np.abs(values[:, :, class_id]).mean(axis=0)
    order = np.argsort(scores)[-max_display:]
    figure, axis = plt.subplots(figsize=(9, max(3, 0.35 * len(order) + 1.5)))
    axis.barh(data["feature_names"][order], scores[order], color="#377eb8")
    label = "All classes (equal output weighting)" if class_id is None else str(data["classes"][class_id])
    axis.set_title(f"SHAP feature importance — {label}")
    axis.set_xlabel(f"Mean |SHAP| ({output_space})")
    figure.tight_layout()
    return figure


def beeswarm_figure(data: dict[str, np.ndarray], class_id: int, max_display: int = 15,
                    output_space: str = "raw_margin"):
    import shap

    explanation = shap.Explanation(values=data["values"][:, :, class_id],
        base_values=data["base_values"][:, class_id], data=data["raw_values"],
        feature_names=data["feature_names"].tolist())
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    figure = plt.gcf()
    figure.axes[0].set_title(f"{data['classes'][class_id]} — {output_space}")
    figure.axes[0].set_xlabel(f"SHAP contribution ({output_space}); colors use raw feature values")
    figure.tight_layout()
    return figure


def waterfall_figure(data: dict[str, np.ndarray], row: int, class_id: int,
                      max_display: int = 15, output_space: str = "raw_margin"):
    import shap

    explanation = shap.Explanation(values=data["values"][row, :, class_id],
        base_values=float(data["base_values"][row, class_id]), data=data["raw_values"][row],
        feature_names=data["feature_names"].tolist())
    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    figure = plt.gcf()
    figure.axes[0].set_title(f"Row {data['row_indices'][row]} · {data['classes'][class_id]} · {output_space}", pad=32)
    figure.tight_layout()
    return figure


def save_plots(folder: Path, class_id: int, local_count: int = 5, max_display: int = 15) -> None:
    data = load_arrays(folder)
    report = json.loads((folder / "shap_report.json").read_text(encoding="utf-8"))
    space = report["output_space"]
    with PLOT_LOCK:
        try:
            for name, figure in (
                ("shap_summary_bar.png", importance_figure(data, max_display=max_display, output_space=space)),
                ("shap_class_bar.png", importance_figure(data, class_id, max_display, space)),
                ("shap_class_beeswarm.png", beeswarm_figure(data, class_id, max_display, space)),
            ):
                figure.savefig(folder / name, dpi=140, bbox_inches="tight")
                plt.close(figure)
            for row in range(min(local_count, len(data["row_indices"]))):
                predicted = int(data["probabilities"][row].argmax())
                figure = waterfall_figure(data, row, predicted, max_display, space)
                figure.savefig(folder / f"shap_local_waterfall_{row:03d}_class_{predicted:02d}.png", dpi=140, bbox_inches="tight")
                plt.close(figure)
        finally:
            plt.close("all")
