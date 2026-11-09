"""Shared presentation for every dashboard workflow; no data or training access."""

from __future__ import annotations

import streamlit as st


STYLES = """
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 1440px; padding: 2.25rem 2.75rem 4rem;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #fff; border-right: 1px solid #e2e8ee;
    min-width: 244px !important; max-width: 244px !important;
}
[data-testid="stSidebarUserContent"] { padding: 0 1.1rem 2rem; }
[data-testid="stSidebarHeader"] { height: 2.5rem; padding: .5rem 1rem; }
.workspace-brand { display: flex; gap: 11px; align-items: center; margin: 0 0 1.5rem; }
.workspace-mark {
    width: 36px; height: 40px; display: grid; place-items: center;
    color: #0f766e; background: #e7f3f0; border-radius: 10px; flex: 0 0 36px;
}
.workspace-brand strong { display: block; color: #1e293b; font-size: 15px; letter-spacing: -.3px; }
.workspace-brand small { display: block; color: #64748b; font-size: 11px; margin-top: 3px; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #64748b; font-size: 11px; font-weight: 600; letter-spacing: .055em;
}
[data-testid="stSidebar"] [data-testid="stRadio"] { width: 100%; }
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) { width: 100% !important; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    padding: 7px 12px; margin: 0; border-radius: 7px; width: 100%; min-height: 36px;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display: none; }
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:last-child { margin-left: 0; }
[data-testid="stSidebar"] label[data-baseweb="radio"] p { font-size: 13px; }
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
    background: #e7f3f0; color: #0f766e; font-weight: 600;
}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover { background: #f1f6f5; }
[data-testid="stSidebar"] label[data-baseweb="radio"]:focus-within { outline: 2px solid #0f766e; outline-offset: 2px; }
[data-testid="stSidebar"] hr { margin: .5rem 0 1rem; border-color: #e2e8ee; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: 11px; line-height: 1.6; }
h1, h2, h3 { color: #172b3a; letter-spacing: -.035em; }
h1, h2 { font-size: 1.9rem !important; font-weight: 650 !important; padding: 0 0 .25rem !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; padding-top: .65rem !important; }
h4 { font-size: .92rem !important; font-weight: 600 !important; padding: 0 0 .65rem !important; }
[data-testid="stCaptionContainer"] p { color: #64748b; line-height: 1.65; }
[data-testid="stCaptionContainer"] { opacity: 1; }
.page-eyebrow { color: #0f766e; font-size: 10px; font-weight: 650; letter-spacing: .13em; text-transform: uppercase; margin-bottom: -.5rem; }
[data-testid="stMetric"] {
    background: #fff; border: 1px solid #dfe6eb; border-radius: 10px;
    padding: 18px 20px; min-height: 120px;
}
[data-testid="stMetricLabel"] p { color: #64748b; font-size: 12px; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: clamp(1.35rem, 2.5vw, 2rem); font-weight: 600; letter-spacing: -.045em; }
[data-testid="stMetricValue"] > div { overflow-wrap: anywhere; white-space: normal; }
[data-testid="stMetricDelta"] { font-size: 11px; }
[data-testid="stForm"], [data-testid="stExpander"],
[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #dfe6eb; border-radius: 10px; }
[data-testid="stForm"] { padding: 1.4rem; }
[data-testid="stExpander"] summary { font-size: 13px; }
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid #dfe6eb; background: #fff; }
[data-testid="stVegaLiteChart"], [data-testid="stImage"] { border-radius: 10px; background: #fff; padding: 12px; }
[data-testid="stWidgetLabel"] p { font-size: 12px; font-weight: 500; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    border-radius: 7px; border-color: #dfe6eb;
}
[data-testid="stFileUploaderDropzone"] { background: #fff; border: 1px dashed #b4c8cc; border-radius: 8px; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button { border-radius: 7px; font-size: 12px; min-height: 38px; font-weight: 500; }
[data-testid="stTabs"] [role="tablist"] { gap: 1.3rem; border-bottom: 1px solid #dfe6eb; margin-bottom: 1.1rem; }
[data-testid="stTabs"] [role="tab"] { font-size: 13px; padding: .6rem .1rem; }
[data-testid="stAlert"] { border-radius: 8px; font-size: 13px; }
[data-testid="stAlert"] p { font-size: 12px; line-height: 1.65; }
[class*="st-key-panel_"][class*="_journey"] [data-testid="stMetric"] {
    padding: 0; min-height: 0; background: transparent; border: 0;
}
[class*="st-key-panel_"][class*="_journey"] [data-testid="stHorizontalBlock"] { align-items: center; }
[class*="st-key-panel_"] { background: #fff; border-radius: 10px; }
@media (max-width: 1100px) {
    [data-testid="stMainBlockContainer"] { padding: 2rem 1.5rem 3rem; }
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 1.6rem 1rem 3rem; }
    h1, h2 { font-size: 1.65rem !important; }
    [data-testid="stMetric"] { min-height: 100px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
}
</style>
"""


def apply_dashboard_style() -> None:
    """Style native Streamlit controls without replacing their interactions."""
    st.markdown(STYLES, unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="workspace-brand"><div class="workspace-mark" aria-hidden="true">'
        '<svg width="22" height="24" viewBox="0 0 24 26" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M12 2 21 6v7c0 5-4 9-9 11-5-2-9-6-9-11V6l9-4Z" stroke="currentColor" stroke-width="1.7"/>'
        '<path d="m6 13 4 0 2-5 3 9 2-4h2" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>'
        '</svg></div><div><strong>IoT Intrusion</strong><small>Detection workspace</small></div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, description: str, *, eyebrow: str = "Binary workflow") -> None:
    """Give every page the same compact title and context hierarchy."""
    from html import escape

    st.markdown(f'<div class="page-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.header(title)
    st.caption(description)
