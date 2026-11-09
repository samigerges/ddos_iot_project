"""One-page model stories and progress charts from saved results."""

from __future__ import annotations

import streamlit as st

from ddos_iot.dashboard_style import page_header
from ddos_iot.journey import BinaryJourney, Journey, load_binary_journey, load_journey
from ddos_iot.journey_charts import binary_comparison_chart, development_chart
from ddos_iot.paths import PROJECT_ROOT


def _pair(data: Journey):
    frame = data.final.loc[data.final.classes == 34]
    reference = frame.loc[frame.candidate == "paper_fixed"]
    selected = frame.loc[frame.candidate != "paper_fixed"]
    return (reference.iloc[0], selected.iloc[0]) if len(reference) == len(selected) == 1 else None


def _snapshot_markdown(data: Journey, count: int | None = None) -> str:
    title = f"{count}-class Research Journey" if count else "Research Journey"
    lines = [f"# {title}", "", "Progress from local experiments to the frozen official-test evaluation.", ""]
    milestones = data.task_milestones[count] if count else data.milestones
    for number, milestone in enumerate(milestones, 1):
        lines += [f"## {number}. {milestone.title}", "", f"Track: {milestone.phase}", "",
                  milestone.finding if milestone.complete else "Complete saved evidence is not available in this checkout.", ""]
    final = data.final.loc[data.final.classes == count] if count else data.final
    if not final.empty:
        lines += ["## Official test results", "", "| Classes | Candidate | Macro-F1 |", "|---:|---|---:|"]
        for row in final.itertuples():
            lines.append(f"| {int(row.classes)} | {row.candidate} | {row.macro_f1_mean:.5f} |")
    if data.recall_budget_pass is False and count != 8:
        lines += ["", "The 34-class specialist's F1 gain did not preserve the selected recall budget on the official test."]
    goals = {8: "Demonstrate a reliable gain over XGBoost.", 34: "Keep the F1 gain while recovering minority-class recall."}
    lines += ["", f"Next: {goals[count]}" if count else "Next: demonstrate an 8-class gain and recover 34-class recall.",
              "Further candidates need a new untouched holdout. Local 46-feature and published 23-feature results use different protocols."]
    return "\n".join(lines) + "\n"


def _render_binary(binary: BinaryJourney) -> None:
    with st.container(border=True, key="panel_binary_journey"):
        title, score = st.columns([4, 1.5])
        title.subheader("Binary classification")
        score.metric("Binary F1", f"{binary.f1:.4f}" if binary.f1 is not None else "Unavailable")
        selected = binary.comparisons.loc[binary.comparisons.selected] if not binary.comparisons.empty else None
        winner = selected.label.iloc[0] if selected is not None and len(selected) == 1 else binary.best_model
        if not binary.comparisons.empty:
            count = len(binary.comparisons)
            st.markdown(f"We trained and compared **{count} {'model' if count == 1 else 'models'}** for Benign vs DDoS. "
                        + (f"**{winner}** achieved the best saved F1." if winner else "A final model has not been selected yet."))
            st.altair_chart(binary_comparison_chart(binary.comparisons), use_container_width=True)
            st.caption("Saved comparison order · test F1 on the same split · the selected model has a larger dot.")
        else:
            st.info("Saved binary model-comparison scores are not available yet.")


def _render_task(data: Journey, count: int, score: float | None, delta: str | None = None) -> None:
    with st.container(border=True, key=f"panel_{count}_journey"):
        title, result = st.columns([4, 1.5])
        title.subheader(f"{count}-class classification")
        result.metric(f"{count}-class Macro-F1", f"{score:.4f}" if score is not None else "Unavailable", delta=delta)
        families = data.model_families[count]
        prefix = (f"We compared **{len(families)} model families**, then tested threshold tuning and specialists. "
                  if families else "We tested threshold tuning and specialist models. ")
        development = data.development.loc[data.development.classes == count]
        if development.empty:
            st.info("Saved development scores are not available yet.")
        else:
            outcome = ("**XGBoost was retained** because the gains were not reliable across splits."
                       if count == 8 else "The **tuned specialist was selected** after improving Macro-F1.")
            st.markdown(prefix + (outcome if score is not None else "The final model choice is pending."))
            selected = ("Reference" if count == 8 else "Tuned specialist") if score is not None else None
            st.altair_chart(development_chart(development, selected_stage=selected), use_container_width=True)
            st.caption("Line: validation Macro-F1 averaged across three matched splits. The score above is the separate final test result.")
        if count == 34 and data.recall_budget_pass is False:
            st.warning("F1 improved, but the recall limit was missed.")


def render_journey() -> None:
    """Show all three model journeys together, without workflow selectors."""
    page_header("Overview", "What we trained, what improved, and how we chose the final models.",
                eyebrow="The model journey")
    data = load_journey(PROJECT_ROOT)
    binary = load_binary_journey(PROJECT_ROOT)
    pair = _pair(data)
    eight = data.final.loc[(data.final.classes == 8) & (data.final.candidate == "paper_fixed")]
    eight_score = eight.iloc[0].macro_f1_mean if len(eight) == 1 else None

    complete = sum(m.complete for m in data.milestones)
    if not complete and data.final.empty and not any(m.complete for m in binary.milestones):
        st.info("No saved research reports are available in this checkout.")
    _render_binary(binary)
    _render_task(data, 8, eight_score)
    _render_task(data, 34, pair[1].macro_f1_mean if pair else None,
                 f"{pair[1].macro_f1_mean - pair[0].macro_f1_mean:+.4f} vs XGBoost" if pair else None)

    st.button("Refresh saved progress", key="journey_refresh")
    with st.expander("Full reports"):
        st.caption("Test scores above use binary F1 or multiclass Macro-F1, with separate datasets and protocols.")
        report = PROJECT_ROOT / "docs/MULTICLASS_FINAL_RESULTS.md"
        if data.final_complete and report.is_file() and report.stat().st_size <= 2_000_000:
            st.download_button("Download full results", report.read_text(encoding="utf-8"),
                               file_name=report.name, mime="text/markdown", key="journey_final_download")
        for count in (8, 34):
            st.download_button(f"Download {count}-class full journey", _snapshot_markdown(data, count),
                               file_name=f"{count}_class_journey.md", mime="text/markdown", key=f"journey_download_{count}")
        for issue in [*binary.issues, *data.issues]:
            st.caption(issue)
