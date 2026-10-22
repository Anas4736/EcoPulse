"""Page 5 - Anomaly Detection.

Contiguous runs of hours above a chosen threshold, grouped into episodes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ecopulse.analysis import episode_summary, find_episodes
from ecopulse.config import DEFAULT_NO2_THRESHOLD_UG_M3

from app import theme


def _default_threshold(column: str, values: pd.Series) -> float:
    if column == "NO2(GT)":
        return DEFAULT_NO2_THRESHOLD_UG_M3
    clean = values.dropna()
    if clean.empty:
        return 0.0
    return float(round(clean.quantile(0.95), 1))


def render(context) -> None:
    column = context.pollutant
    frame = context.frame

    theme.page_header(
        "Anomaly Detection",
        "An air-quality **episode** is a run of consecutive hours in which a pollutant "
        "stays above a threshold. This page finds those runs, measures how long they "
        "last and how high they peak, and shows where they sit in the record.",
    )
    theme.disclaimer()
    theme.notice(
        "**An exceedance is a statistical label, not a medical diagnosis or a health "
        "advisory.** It means a reading passed a threshold you chose. It says nothing "
        "about individual exposure, symptoms or risk, and the underlying record is "
        "historical rather than current.",
        kind="warn",
    )

    if column not in frame.columns:
        theme.empty_state(f"The column '{column}' is not present in this dataset.")
        return
    if frame.empty:
        theme.empty_state(
            "The selected date range contains no hours. Widen it in the sidebar."
        )
        return

    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        theme.empty_state(
            f"**{column} has no readings** in the selected range, so no episodes can "
            "be found. Try a different pollutant or a wider date range."
        )
        return

    unit = theme.column_unit(column)
    theme.section(
        "Episode settings",
        "Both controls apply immediately. The threshold is clamped to the range of "
        "the readings currently selected.",
    )
    with st.container(border=True):
        threshold = theme.threshold_input(
            column,
            valid,
            default=_default_threshold(column, valid),
            help_text=(
                "An hour counts as an exceedance when the reading is strictly above this "
                "value. For NO2 the default of 200 ug/m3 is the European hourly limit "
                "value (Directive 2008/50/EC), clamped to the range of the data shown; "
                "for the other pollutants it is the 95th percentile of the selected data."
            ),
        )
        min_duration = st.slider(
            "Minimum episode length (hours)",
            min_value=1,
            max_value=24,
            value=1,
            help="Shorten the list by ignoring single spikes. An episode is still counted in the totals only if it is at least this long.",
        )

    summary = episode_summary(frame, column, threshold, min_duration=min_duration)
    episodes = find_episodes(frame, column, threshold, min_duration=min_duration)

    if summary["n_valid"] == 0:
        theme.empty_state("There are no valid readings in this range.")
        return

    # -- summary cards ------------------------------------------------------
    theme.metric_cards(
        [
            {
                "label": "Hours above threshold",
                "value": theme.fmt_number(summary["n_exceedance_hours"], 0),
                "note": f"{summary['exceedance_pct_of_valid']:.2f}% of valid readings",
                "help": f"Hours where {theme.column_label(column)} was above {threshold:g} {unit}.",
            },
            {
                "label": "Episodes",
                "value": theme.fmt_number(summary["n_episodes"], 0),
                "note": f"at least {min_duration} h long",
                "help": "Runs of consecutive hours above the threshold.",
            },
            {
                "label": "Longest episode",
                "value": f"{theme.fmt_number(summary['max_episode_hours'], 0)} h",
                "note": "consecutive hours",
                "help": "The longest unbroken run above the threshold.",
            },
            {
                "label": "Median episode",
                "value": f"{theme.fmt_number(summary['median_episode_hours'], 1)} h",
                "note": "typical length",
                "help": "The middle episode length: half are shorter, half are longer.",
            },
            {
                "label": "Highest reading",
                "value": theme.fmt_number(summary["peak_value"], 1),
                "note": unit,
                "help": "The single highest value at any moment above the threshold in this range.",
            },
            {
                "label": "Hours by year",
                "value": " / ".join(
                    f"{year}: {hours:,}"
                    for year, hours in sorted(summary["exceedance_hours_by_year"].items())
                ),
                "note": "exceedance hours",
                "help": "How the exceedance hours split across the two calendar years in the record.",
            },
        ]
    )

    if summary["n_episodes"] == 0:
        theme.notice(
            f"**No episode reached {threshold:g} {unit}.** The highest reading in this "
            f"range was {valid.max():.1f} {unit}. Lower the threshold to see shorter, "
            "less severe events.",
            kind="ok",
        )
        return

    # -- time series -------------------------------------------------------
    theme.section(
        "Where the episodes sit",
        "The full series with the threshold line. Red dots mark the individual hours "
        "that exceed it; the table below joins those hours into episodes.",
    )

    exceed_mask = values > threshold
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["datetime"],
            y=values,
            mode="lines",
            name="Hourly reading",
            line=dict(color=theme.MUTED, width=1),
            opacity=0.7,
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.1f} " + unit + "<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.loc[exceed_mask, "datetime"],
            y=values[exceed_mask],
            mode="markers",
            name="Above threshold",
            marker=dict(color=theme.DANGER, size=4.5),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.1f} " + unit + "<extra></extra>",
        )
    )
    figure.add_hline(
        y=threshold,
        line=dict(color=theme.DANGER, width=1.6, dash="dash"),
        annotation_text=f"Threshold {threshold:g}",
        annotation_position="top left",
        annotation_font=dict(color=theme.DANGER, size=11),
    )
    figure.update_yaxes(title=f"{theme.column_label(column)} ({unit or 'value'})")
    theme.style_figure(figure, height=400)
    st.plotly_chart(figure, width="stretch")

    # -- episode table -----------------------------------------------------
    theme.section(
        "The episodes",
        "One row per episode, longest first. Each row is a run of consecutive hours "
        "in which the reading stayed above the threshold.",
    )

    display = episodes.copy()
    display["Start"] = pd.to_datetime(display["start"]).dt.strftime("%d %b %Y %H:%M")
    display["End"] = pd.to_datetime(display["end"]).dt.strftime("%d %b %Y %H:%M")
    display["Peak time"] = pd.to_datetime(display["peak_time"]).dt.strftime("%d %b %Y %H:%M")
    display = display.rename(
        columns={
            "duration_hours": "Hours",
            "peak_value": "Peak",
            "mean_value": "Episode mean",
        }
    )
    display = display[
        ["Start", "End", "Hours", "Peak", "Episode mean", "Peak time"]
    ].sort_values("Hours", ascending=False)

    st.dataframe(
        display.style.format({"Peak": "{:.1f}", "Episode mean": "{:.1f}", "Hours": "{:.0f}"}),
        width="stretch",
        hide_index=True,
    )
    theme.download_button(display, filename=f"ecopulse_episodes_{column.replace('(', '_').replace(')', '')}.csv")

    # -- duration distribution ---------------------------------------------
    theme.section(
        "How long do episodes last?",
        "Most episodes are short. The long tail is what matters for health "
        "exposure, because a long episode keeps people above the threshold for hours.",
    )
    durations = episodes["duration_hours"].astype(int)
    bins = np.arange(0.5, max(2, durations.max() + 1.5), 1.0)
    counts, edges = np.histogram(durations, bins=bins)

    figure = go.Figure(
        go.Bar(
            x=(edges[:-1] + edges[1:]) / 2,
            y=counts,
            width=np.diff(edges),
            marker_color=theme.ACCENT,
            hovertemplate="%{x:.0f} hours long<br>Episodes: %{y:,}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Episode length (hours)")
    figure.update_yaxes(title="Number of episodes")
    theme.style_figure(figure, height=300, showlegend=False)
    st.plotly_chart(figure, width="stretch")
