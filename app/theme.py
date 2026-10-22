"""Shared look-and-feel for the EcoPulse dashboard.

Everything visual lives here so that all seven pages look like one product:
the colour palette, the injected stylesheet, the metric cards, the notice
boxes and the Plotly defaults.

Nothing in this module computes statistics. It only formats what the tested
analysis modules have already computed.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from ecopulse.config import COLUMN_METADATA

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Brand colours. One blue for the primary series, teal and amber for
#: comparison series, red reserved for threshold breaches.
PRIMARY = "#1f6feb"
SECONDARY = "#0d9488"
ACCENT = "#f59e0b"
DANGER = "#dc2626"
VIOLET = "#7c3aed"
MUTED = "#94a3b8"
INK = "#0f172a"
SUBTLE = "#5b6b7f"
GRID = "#e2e8f0"
SURFACE = "#ffffff"
CANVAS = "#f6f8fb"

#: The deep navy used by the sidebar and the overview hero.
NAVY = "#0b1f33"
NAVY_SOFT = "#102a43"

#: A categorical sequence that stays distinguishable for the most common
#: forms of colour blindness (blue / teal / amber / violet / red).
SEQUENCE = [PRIMARY, SECONDARY, ACCENT, VIOLET, DANGER, "#0891b2", "#65a30d", "#db2777"]

#: One colour per season, used by every page that breaks data down by season.
SEASON_COLORS = {
    "Winter": "#2563eb",
    "Spring": "#16a34a",
    "Summer": "#f59e0b",
    "Autumn": "#c2410c",
}

STYLESHEET = """
<style>
/* ---- Canvas and typography ------------------------------------------- */
.stApp { background-color: #f6f8fb; }
.block-container {
    padding: 2.5rem 2.6rem 3.5rem;
    max-width: 1580px;
}
@media (max-width: 1280px) {
    .block-container { padding: 2.1rem 1.5rem 3rem; }
}

h1, h2, h3, h4, h5 { color: #0f172a; letter-spacing: -0.015em; }
h1 { font-size: 1.9rem; font-weight: 750; margin-bottom: 0.15rem; }
h2 {
    font-size: 1.24rem; font-weight: 700; margin-top: 2.1rem;
    margin-bottom: 0.5rem;
}
h3 { font-size: 1.03rem; font-weight: 650; }
p, li, td, th { color: #24303f; }
a { color: #1d4ed8; text-decoration: none; }
a:hover { text-decoration: underline; }
::selection { background: #cceee9; color: #0b1f33; }

/* ---- Reusable content blocks ----------------------------------------- */
.ep-eyebrow {
    text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem;
    font-weight: 750; color: #1d4ed8; margin-bottom: 0.3rem;
}
.ep-lead { color: #475569; font-size: 1rem; line-height: 1.62; max-width: 68rem; }
.ep-muted { color: #5b6b7f; font-size: 0.86rem; line-height: 1.58; }

/* Hero used on the Overview page. */
.ep-hero {
    background: linear-gradient(135deg, #0b1f33 0%, #11324d 58%, #0d4f4a 100%);
    color: #ffffff; border-radius: 18px; padding: 1.6rem 1.8rem 1.55rem;
    margin: 0.1rem 0 1.3rem; box-shadow: 0 14px 34px rgba(11, 31, 51, 0.20);
    position: relative; overflow: hidden;
}
.ep-hero::after {
    content: ""; position: absolute; right: -90px; top: -110px;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(13, 148, 136, 0.35), transparent 68%);
    z-index: 0;
}
.ep-hero > * { position: relative; z-index: 1; }
.ep-hero-eyebrow {
    text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.7rem;
    font-weight: 750; color: #7dd3c7; margin-bottom: 0.35rem;
}
.ep-hero h1 {
    color: #ffffff; font-size: 2.15rem; line-height: 1.1; margin: 0;
    font-weight: 800;
}
.ep-hero-tagline {
    color: #cbd8e6; font-size: 1rem; line-height: 1.6; max-width: 60rem;
    margin-top: 0.55rem;
}
.ep-hero-alert {
    display: inline-block; margin-top: 0.95rem; padding: 0.36rem 0.8rem;
    border-radius: 999px; background: rgba(245, 158, 11, 0.16);
    border: 1px solid rgba(245, 158, 11, 0.45); color: #fde3b8;
    font-size: 0.78rem; font-weight: 650;
}
.ep-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.9rem; }
.ep-chip {
    background: rgba(255, 255, 255, 0.09); border: 1px solid rgba(255, 255, 255, 0.18);
    color: #e7eef6; border-radius: 999px; padding: 0.24rem 0.7rem;
    font-size: 0.75rem; font-weight: 600;
}
.ep-chip b { color: #ffffff; }

/* Metric cards. */
.ep-cards {
    display: flex; flex-wrap: wrap; gap: 0.85rem; margin: 0.55rem 0 1.4rem;
    align-items: stretch;
}
.ep-card {
    position: relative; flex: 1 1 0; min-width: 150px; background: #ffffff;
    border: 1px solid #e3e8ef; border-radius: 14px;
    padding: 0.9rem 1.05rem 1rem 1.15rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
.ep-card::before {
    content: ""; position: absolute; left: 0; top: 0.7rem; bottom: 0.7rem;
    width: 3px; border-radius: 0 3px 3px 0;
    background: linear-gradient(180deg, #1f6feb, #0d9488);
}
.ep-card-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #5b6b7f; font-weight: 750;
}
.ep-card-value {
    font-size: 1.55rem; font-weight: 750; color: #0f172a; line-height: 1.2;
    margin-top: 0.3rem; word-break: break-word; font-variant-numeric: tabular-nums;
}
.ep-card-note { font-size: 0.77rem; color: #5b6b7f; margin-top: 0.25rem; line-height: 1.45; }

/* Computed key-finding cards. */
.ep-findings {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 0.85rem; margin: 0.45rem 0 1.2rem;
}
.ep-finding {
    background: #ffffff; border: 1px solid #e3e8ef; border-top: 3px solid #0d9488;
    border-radius: 14px; padding: 0.9rem 1rem 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
.ep-finding-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #1d4ed8; font-weight: 750;
}
.ep-finding-value {
    font-size: 1.12rem; font-weight: 750; color: #0f172a; margin-top: 0.3rem;
    line-height: 1.3;
}
.ep-finding-detail { font-size: 0.85rem; color: #475569; margin-top: 0.35rem; line-height: 1.5; }

/* Notices. */
.ep-note {
    border-left: 3px solid #1f6feb; background: #eff6ff; padding: 0.75rem 1rem;
    border-radius: 0 10px 10px 0; font-size: 0.9rem; color: #1e3a8a;
    margin: 0.6rem 0 1.1rem; line-height: 1.55;
}
.ep-note p, .ep-note li { color: #1e3a8a; }
.ep-warn { border-left-color: #d97706; background: #fffbeb; color: #92400e; }
.ep-warn p, .ep-warn li { color: #92400e; }
.ep-ok { border-left-color: #16a34a; background: #f0fdf4; color: #166534; }
.ep-ok p, .ep-ok li { color: #166534; }
.ep-danger { border-left-color: #dc2626; background: #fef2f2; color: #991b1b; }
.ep-danger p, .ep-danger li { color: #991b1b; }

.ep-tag {
    display: inline-block; background: #eef2f7; color: #334155;
    border-radius: 999px; padding: 0.14rem 0.6rem; font-size: 0.74rem;
    font-weight: 650; margin-right: 0.3rem;
}
.ep-footer {
    color: #5b6b7f; font-size: 0.8rem; text-align: center; margin-top: 3.2rem;
    padding-top: 1.2rem; border-top: 1px solid #e3e8ef;
}

/* ---- Sidebar --------------------------------------------------------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1f33 0%, #0c2438 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
[data-testid="stSidebar"] > div { background: transparent; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #dbe6f1;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 { color: #ffffff; }
[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.12); margin: 0.9rem 0;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #9fb3c8; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p { color: #c9d7e5; }

/* Navigation pills (radio group) */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.12rem; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    width: 100%; padding: 0.44rem 0.6rem; border-radius: 9px;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.07);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: rgba(13, 148, 136, 0.26);
}
[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #dbe6f1; font-size: 0.92rem; font-weight: 500;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #ffffff; font-weight: 650;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
    border-color: #7f96ad;
}

/* Widgets keep light surfaces, so their text must stay dark. */
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="input"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] div { color: #0f172a; }
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff; border-radius: 9px;
}
[data-testid="stSidebar"] [data-baseweb="popover"] * { color: #0f172a; }
[data-testid="stSidebar"] button[kind="header"] { color: #c9d7e5; }

.ep-brand { display: flex; align-items: center; gap: 0.7rem; padding: 0.1rem 0 0.8rem; }
.ep-brand-mark {
    width: 40px; height: 40px; border-radius: 12px; flex: 0 0 40px;
    background: linear-gradient(135deg, #0d9488, #1f6feb);
    display: flex; align-items: center; justify-content: center;
    color: #ffffff; font-weight: 800; font-size: 0.95rem;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.35);
}
.ep-brand-name { color: #ffffff; font-size: 1.12rem; font-weight: 750; line-height: 1.15; }
.ep-brand-sub {
    color: #9fb3c8; font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 700; margin-top: 0.1rem;
}
.ep-side-label {
    font-size: 0.67rem; letter-spacing: 0.13em; text-transform: uppercase;
    color: #8fa6bd; font-weight: 750; margin: 0.2rem 0 0.35rem;
}
.ep-side-foot {
    background: rgba(255, 255, 255, 0.055); border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px; padding: 0.7rem 0.8rem; font-size: 0.78rem;
    line-height: 1.55; color: #c3d2e0;
}
.ep-side-foot strong { color: #ffffff; }

/* ---- Native Streamlit controls --------------------------------------- */
[data-testid="stCaptionContainer"] p { color: #5b6b7f; }
[data-testid="stExpander"] {
    border: 1px solid #e3e8ef; border-radius: 12px; background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 0.25rem 0.4rem; }
[data-testid="stExpander"] summary p { font-weight: 650; color: #0f172a; }
[data-testid="stExpander"] summary:hover p { color: #0b6b63; }
[data-testid="stDataFrame"] {
    border: 1px solid #e3e8ef; border-radius: 12px; overflow: hidden;
    background: #ffffff;
}
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stDownloadButton"] button, [data-testid="stButton"] button {
    border-radius: 9px; border: 1px solid #cbd5e1; background: #ffffff;
    color: #0f172a; font-weight: 600;
}
[data-testid="stDownloadButton"] button:hover, [data-testid="stButton"] button:hover {
    border-color: #0d9488; color: #0b6b63;
}
[data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p {
    color: #24303f; font-size: 0.9rem;
}
</style>
"""

DISCLAIMER = (
    "**Historical data only.** EcoPulse analyses the UCI Air Quality dataset "
    "(a fixed record from March 2004 to April 2005). There is no live feed, "
    "no real-time monitoring and no current air-quality information in this "
    "application. No AQI index is computed or displayed, because the dataset "
    "does not support one."
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def inject_css() -> None:
    """Apply the EcoPulse stylesheet to the running app."""
    st.markdown(STYLESHEET, unsafe_allow_html=True)


def sidebar_brand(name: str = "EcoPulse", subtitle: str = "Air Quality Analysis") -> None:
    """The product mark at the top of the sidebar."""
    st.markdown(
        '<div class="ep-brand">'
        '<div class="ep-brand-mark">EP</div>'
        f'<div><div class="ep-brand-name">{html.escape(name)}</div>'
        f'<div class="ep-brand-sub">{html.escape(subtitle)}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def sidebar_label(text: str) -> None:
    """A small uppercase label used to group sidebar controls."""
    st.markdown(f'<div class="ep-side-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def sidebar_footer(text: str) -> None:
    """A small information card at the bottom of the sidebar."""
    st.markdown(f'<div class="ep-side-foot">{text}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "EcoPulse") -> None:
    """The standard header at the top of every page."""
    st.markdown(f'<div class="ep-eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="ep-lead">{subtitle}</div>', unsafe_allow_html=True)


def hero(
    title: str,
    tagline: str,
    eyebrow: str = "Environmental analytics",
    chips: Sequence[str] | None = None,
    alert: str | None = None,
) -> None:
    """A dark product banner used at the top of the Overview page.

    ``chips`` are short facts (they may contain ``<b>`` markup) and ``alert`` is
    the prominent historical-data notice.
    """
    parts = ['<div class="ep-hero">']
    if eyebrow:
        parts.append(f'<div class="ep-hero-eyebrow">{html.escape(eyebrow)}</div>')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f'<div class="ep-hero-tagline">{tagline}</div>')
    if chips:
        parts.append('<div class="ep-chips">')
        for chip in chips:
            parts.append(f'<span class="ep-chip">{chip}</span>')
        parts.append("</div>")
    if alert:
        parts.append(f'<div class="ep-hero-alert">{alert}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, help_text: str | None = None) -> None:
    """A section heading with an optional explanation."""
    st.markdown(f"## {title}")
    if help_text:
        st.caption(help_text)


def notice(text: str, kind: str = "info") -> None:
    """A coloured explanatory box.

    ``kind`` is one of ``info``, ``warn``, ``ok`` or ``danger``.
    """
    css_class = {
        "info": "ep-note",
        "warn": "ep-note ep-warn",
        "ok": "ep-note ep-ok",
        "danger": "ep-note ep-danger",
    }.get(kind, "ep-note")
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def disclaimer() -> None:
    """The historical-data notice that must stay visible on every page."""
    notice(DISCLAIMER, kind="info")


def metric_cards(items: Sequence[dict]) -> None:
    """Render a row of metric cards.

    Each item is a dict with the keys ``label``, ``value`` and optional
    ``note`` and ``help``. ``help`` becomes the hover tooltip.
    """
    if not items:
        return
    parts = ['<div class="ep-cards">']
    for item in items:
        label = item.get("label", "")
        value = item.get("value", "-")
        note = item.get("note")
        tip = item.get("help")
        title_attr = f' title="{html.escape(str(tip), quote=True)}"' if tip else ""
        note_html = f'<div class="ep-card-note">{note}</div>' if note else ""
        parts.append(
            f'<div class="ep-card"{title_attr}>'
            f'<div class="ep-card-label">{label}</div>'
            f'<div class="ep-card-value">{value}</div>'
            f"{note_html}</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def findings(items: Sequence[dict]) -> None:
    """Render computed key-finding cards.

    Each item is a dict with ``label``, ``value`` and optional ``detail``.
    Use this only for values computed from the data on the current request.
    """
    if not items:
        return
    parts = ['<div class="ep-findings">']
    for item in items:
        label = item.get("label", "")
        value = item.get("value", "-")
        detail = item.get("detail")
        detail_html = f'<div class="ep-finding-detail">{detail}</div>' if detail else ""
        parts.append(
            '<div class="ep-finding">'
            f'<div class="ep-finding-label">{label}</div>'
            f'<div class="ep-finding-value">{value}</div>'
            f"{detail_html}</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def empty_state(message: str) -> None:
    """Shown when the current filters leave no rows to display."""
    st.info(f"**Nothing to show for these filters.** {message}")


def footer() -> None:
    """The footer shown at the bottom of every page."""
    st.markdown(
        '<div class="ep-footer">EcoPulse &middot; UCI Air Quality dataset '
        "(Vito, 2008) &middot; historical record only &middot; "
        "no live data</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt_number(value: Any, decimals: int = 1) -> str:
    """Format a number for a metric card, or ``n/a`` when it is not usable."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:,.{decimals}f}"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    """Format a fraction as a percentage."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number * 100:.{decimals}f}%"


def column_label(column: str) -> str:
    """A human-readable name for a dataset column."""
    meta = COLUMN_METADATA.get(column)
    if meta:
        return meta["label"]
    return column


def column_unit(column: str) -> str:
    """The unit of a dataset column, or an empty string."""
    meta = COLUMN_METADATA.get(column)
    return meta["unit"] if meta else ""


def column_role(column: str) -> str:
    """The role of a dataset column: reference, sensor or weather."""
    meta = COLUMN_METADATA.get(column)
    return meta["role"] if meta else ""


def with_unit(column: str, value: Any, decimals: int = 1) -> str:
    """Format a value with its unit, e.g. ``"139.9 ug/m3"``."""
    unit = column_unit(column)
    text = fmt_number(value, decimals)
    return f"{text} {unit}".strip()


def download_button(
    frame: pd.DataFrame,
    filename: str,
    label: str = "Download as CSV",
    mime: str = "text/csv",
) -> None:
    """Offer a table for download, but only when there is something to offer."""
    if frame is None or len(frame) == 0:
        return
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime=mime,
    )


def threshold_input(
    column: str,
    values: "pd.Series",
    default: float,
    help_text: str,
) -> float:
    """A threshold slider that cannot exceed the data it describes.

    The default value (for example the 200 ug/m3 NO2 limit) may sit outside a
    narrow date range, so it is clamped into the observed span before being
    handed to the widget. Streamlit rejects a value above ``max_value``, so
    without this clamp a short, clean date range would crash the page.
    """
    valid = values.dropna()
    lower = float(np.floor(valid.min()))
    upper = float(np.ceil(valid.max()))
    if upper <= lower:
        upper = lower + 1.0
    value = float(min(max(default, lower), upper))
    step = max(round((upper - lower) / 200, 2), 0.01)
    return float(
        st.number_input(
            f"Threshold ({column_unit(column) or 'value'})",
            min_value=lower,
            max_value=upper,
            value=value,
            step=step,
            help=help_text,
        )
    )


# ---------------------------------------------------------------------------
# Plotly defaults
# ---------------------------------------------------------------------------


def style_figure(
    figure,
    height: int | None = None,
    title: str | None = None,
    showlegend: bool = True,
) -> None:
    """Apply the EcoPulse look to a Plotly figure."""
    layout = dict(
        template="plotly_white",
        colorway=SEQUENCE,
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=13, color=INK),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        margin=dict(l=8, r=14, t=38 if title else 18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            font=dict(size=12, color=INK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11.5, color="#24303f"),
            title=dict(text=""),
        ),
        showlegend=showlegend,
        separators=".,",
    )
    if height is not None:
        layout["height"] = height
    if title:
        layout["title"]["text"] = title
    figure.update_layout(**layout)
    figure.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor="#cbd5e1",
        automargin=True,
        tickfont=dict(size=11.5, color="#334155"),
        title=dict(font=dict(size=12, color="#475569")),
    )
    figure.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor="#cbd5e1",
        automargin=True,
        tickfont=dict(size=11.5, color="#334155"),
        title=dict(font=dict(size=12, color="#475569")),
    )


def severity_color(value: float, good_above: float = 0.7) -> str:
    """Green / amber / red for a score, used for R2 and correlation cells."""
    if value is None or not np.isfinite(value):
        return MUTED
    if value >= good_above:
        return SECONDARY
    if value >= 0.4:
        return ACCENT
    return DANGER


def chunked(items: Iterable, size: int) -> list[list]:
    """Split an iterable into lists of at most ``size`` items."""
    items = list(items)
    return [items[i : i + size] for i in range(0, len(items), size)]


__all__ = [
    "PRIMARY",
    "SECONDARY",
    "ACCENT",
    "DANGER",
    "VIOLET",
    "MUTED",
    "INK",
    "SUBTLE",
    "GRID",
    "NAVY",
    "NAVY_SOFT",
    "SEQUENCE",
    "SEASON_COLORS",
    "DISCLAIMER",
    "inject_css",
    "sidebar_brand",
    "sidebar_label",
    "sidebar_footer",
    "page_header",
    "hero",
    "section",
    "notice",
    "disclaimer",
    "metric_cards",
    "findings",
    "empty_state",
    "footer",
    "fmt_number",
    "fmt_pct",
    "column_label",
    "column_unit",
    "column_role",
    "with_unit",
    "download_button",
    "style_figure",
    "severity_color",
    "chunked",
]
