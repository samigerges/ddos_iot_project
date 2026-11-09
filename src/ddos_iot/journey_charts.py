"""Small charts for saved journey reports; protocols remain separate."""

from __future__ import annotations

import altair as alt
import pandas as pd


ACCENT = "#0f766e"


def development_chart(frame: pd.DataFrame, *, selected_stage: str | None = None):
    values = frame.sort_values("order").copy()
    values["label"] = values.stage.replace({"Reference": "XGBoost", "OOF bias": "Thresholds"})
    values["selected"] = values.stage.eq(selected_stage)
    axis = alt.X("label:N", sort=values.label.tolist(), title=None,
                 axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=120,
                               labelExpr="split(datum.label, ' ')", labelLineHeight=12))
    scale = alt.Scale(domain=[max(0, float(values.lower.min()) - .003),
                             min(1, float(values.upper.max()) + .003)], zero=False)
    base = alt.Chart(values).encode(x=axis, tooltip=[
        alt.Tooltip("label:N", title="Stage"),
        alt.Tooltip("macro_f1:Q", title="Mean Macro-F1", format=".5f"),
        alt.Tooltip("std:Q", title="Split std", format=".5f"),
        alt.Tooltip("seeds:Q", title="Seeds", format=".0f"),
        alt.Tooltip("selected:N", title="Final choice"),
    ])
    line = base.mark_line(color=ACCENT, point=alt.OverlayMarkDef(size=80), strokeWidth=3).encode(
        y=alt.Y("macro_f1:Q", title="Macro-F1", scale=scale))
    labels = base.mark_text(dy=-14, color=ACCENT, fontWeight="bold", fontSize=11).encode(
        y="macro_f1:Q", text=alt.Text("macro_f1:Q", format=".4f"))
    selected = base.transform_filter(alt.datum.selected).mark_point(
        filled=True, size=160, color=ACCENT, stroke="white", strokeWidth=2).encode(y="macro_f1:Q")
    return ((line + labels + selected).properties(height=190, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(domain=False, tickSize=0, labelColor="#64748b", titleColor="#64748b",
                            gridColor="#e8eef1", labelPadding=12, titlePadding=14))


def binary_comparison_chart(frame: pd.DataFrame):
    """Compare independent models on the same split, in the report's saved order."""
    values = frame.sort_values("order").copy()
    low, high = float(values.f1.min()), float(values.f1.max())
    padding = max(.0005, (high - low) * .25)
    scale = alt.Scale(domain=[max(0, low - padding), min(1, high + padding)], zero=False)
    base = alt.Chart(values).encode(
        x=alt.X("label:N", sort=values.label.tolist(), title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=100, labelOverlap=False,
                              labelExpr="split(datum.label, ' ')", labelFontSize=10, labelLineHeight=12)),
        y=alt.Y("f1:Q", title="Test F1", scale=scale, axis=alt.Axis(format=".3f", tickCount=4)),
        tooltip=[alt.Tooltip("label:N", title="Model"), alt.Tooltip("f1:Q", title="Test F1", format=".5f"),
                 alt.Tooltip("selected:N", title="Selected model")],
    )
    line = base.mark_line(color=ACCENT, strokeWidth=2.5,
                          point=alt.OverlayMarkDef(size=65, filled=True))
    labels = base.mark_text(dy=-15, fontSize=11, color=ACCENT).encode(text=alt.Text("f1:Q", format=".5f"))
    selected = base.transform_filter(alt.datum.selected).mark_point(
        filled=True, size=160, color=ACCENT, stroke="white", strokeWidth=2)
    return ((line + labels + selected).properties(height=190, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(domain=False, tickSize=0, labelColor="#64748b", titleColor="#64748b",
                            gridColor="#e8eef1", labelPadding=10, titlePadding=14))
