"""Page 2 - Pollution Explorer.

One pollutant, explored four ways: as a time series, as a daily profile, as a
distribution and as a monthly trend. The date range and the pollutant are
chosen in the sidebar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ecopulse.analysis import (
    daily_profile,
    hourly_profile,
    monthly_aggregate,
    pollutant_distribution,
)
from ecopulse.config import DEFAULT_NO2_THRESHOLD_UG_M3
from ecopulse.features import add_rolling_features, rolling_name

from app import theme


def _default_threshold(column: str, values: pd.Series) -> float:
    """A sensible starting threshold: the 95th percentile of what is there."""
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
        "Pollution Explorer",
        f"How **{theme.column_label(column)}** behaves across the selected period. "
        "All four charts below are computed from the hours inside the date range in "
        "the sidebar.",
    )
    theme.disclaimer()

    st.caption(
        f"Showing **{theme.column_label(column)}** for "
        f"{frame['datetime'].min():%d %b %Y} – {frame['datetime'].max():%d %b %Y} "
        f"({len(frame):,} hours). Change the range or the pollutant in the sidebar."
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
            f"**{column} has no readings at all** in the selected range, so nothing "
            "can be charted. Try a different pollutant or a wider date range."
        )
        return

    theme.section(
        "Threshold",
        "Every chart marks this level, and the card below counts the hours above it.",
    )
    with st.container(border=True):
        threshold = theme.threshold_input(
            column,
            valid,
            default=_default_threshold(column, valid),
            help_text=(
                "The level above which an hour counts as an exceedance. "
                "For NO2 the default is 200 ug/m3, the European hourly limit value "
                "(Directive 2008/50/EC), clamped to the range of the data shown. For "
                "the other pollutants it is the 95th percentile of the selected data."
            ),
        )

    above = valid[valid > threshold]
    unit = theme.column_unit(column)

    # -- summary numbers ---------------------------------------------------
    theme.section(
        "The selected period",
        f"Summary statistics for {theme.column_label(column)} over the hours inside "
        "the date range. The threshold is the one set above.",
    )
    theme.metric_cards(
        [
            {
                "label": "Hours with a reading",
                "value": theme.fmt_number(valid.size, 0),
                "note": f"of {theme.fmt_number(len(frame), 0)} hours in range",
                "help": "Hours where this pollutant was actually measured.",
            },
            {
                "label": "Mean",
                "value": theme.fmt_number(valid.mean(), 1),
                "note": unit,
                "help": "The arithmetic mean of every reading in the selected period.",
            },
            {
                "label": "Median",
                "value": theme.fmt_number(valid.median(), 1),
                "note": unit,
                "help": "The middle reading. When the mean is much higher, the distribution has a long upper tail.",
            },
            {
                "label": "Maximum",
                "value": theme.fmt_number(valid.max(), 1),
                "note": unit,
                "help": "The highest single hourly reading in the selected period.",
            },
            {
                "label": "95th percentile",
                "value": theme.fmt_number(valid.quantile(0.95), 1),
                "note": unit,
                "help": "The level that 95% of readings fall below.",
            },
            {
                "label": "Hours above threshold",
                "value": theme.fmt_number(above.size, 0),
                "note": f"{theme.fmt_pct(above.size / valid.size, 2)} of valid readings",
                "help": f"Hours where {theme.column_label(column)} was above {threshold:g} {unit}.",
            },
        ]
    )

    # -- time series -------------------------------------------------------
    theme.section(
        "The time series",
        "The raw hourly readings with a 24-hour moving average on top, so the "
        "daily rhythm stays visible inside the overall trend.",
    )

    prepared = add_rolling_features(
        frame, columns=[column], window=24, stats=("mean",), shift=0
    )
    rolling_column = rolling_name(column, 24, "mean")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prepared["datetime"],
            y=prepared[column],
            mode="lines",
            name="Hourly reading",
            line=dict(color=theme.MUTED, width=1),
            opacity=0.65,
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.2f} " + unit + "<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=prepared["datetime"],
            y=prepared[rolling_column],
            mode="lines",
            name="24-hour average",
            line=dict(color=theme.PRIMARY, width=2.4),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>24h mean: %{y:.2f} " + unit + "<extra></extra>",
        )
    )
    figure.add_hline(
        y=threshold,
        line=dict(color=theme.DANGER, width=1.6, dash="dash"),
        annotation_text=f"Threshold {threshold:g}",
        annotation_position="top left",
        annotation_font=dict(color=theme.DANGER, size=11),
    )
    figure.update_xaxes(title="")
    figure.update_yaxes(title=f"{theme.column_label(column)} ({unit or 'value'})")
    theme.style_figure(figure, height=380)
    st.plotly_chart(figure, width="stretch")

    # -- hourly and seasonal profile ---------------------------------------
    theme.section(
        "The daily rhythm",
        "The same readings grouped by hour of the day, with the option to split "
        "them by season.",
    )
    split_by_season = st.checkbox(
        "Show one line per season",
        value=False,
        help="Each season has its own diurnal profile, because traffic and heating change through the year.",
    )
    profile = hourly_profile(frame, column, by_season=split_by_season)

    if split_by_season and "season" in profile.columns:
        figure = px.line(
            profile,
            x="hour",
            y="mean",
            color="season",
            color_discrete_map=theme.SEASON_COLORS,
            labels={"hour": "Hour of day", "mean": f"Mean {column} ({unit or 'value'})"},
            custom_data=["n"],
            height=340,
        )
        figure.update_traces(
            hovertemplate=(
                "<b>%{fullData.name}</b> at %{x}:00<br>Mean: %{y:.1f}<br>"
                "Hours: %{customdata[0]:,}<extra></extra>"
            )
        )
    else:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=profile["hour"],
                y=profile["mean"],
                marker_color=theme.PRIMARY,
                opacity=0.85,
                name=f"Mean {column}",
                customdata=np.stack([profile["n"], profile["min"], profile["max"]], axis=-1),
                hovertemplate=(
                    "<b>%{x}:00</b><br>Mean: %{y:.1f}<br>Min: %{customdata[1]:.1f}<br>"
                    "Max: %{customdata[2]:.1f}<br>Hours: %{customdata[0]:,}<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=profile["hour"],
                y=profile["median"],
                mode="lines+markers",
                name="Median",
                line=dict(color=theme.ACCENT, width=2.4),
                hovertemplate="<b>%{x}:00</b><br>Median: %{y:.1f}<extra></extra>",
            )
        )
        figure.update_xaxes(
            title="Hour of day",
            tickmode="array",
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
        )
        figure.update_yaxes(title=f"Mean {column} ({unit or 'value'})")

    theme.style_figure(figure, height=340)
    st.plotly_chart(figure, width="stretch")

    # -- distribution ------------------------------------------------------
    left, right = st.columns(2)

    with left:
        theme.section("The distribution")
        distribution = pollutant_distribution(frame, column)
        edges = distribution["edges"]
        counts = distribution["counts"]
        centres = (edges[:-1] + edges[1:]) / 2 if len(edges) > 1 else []
        widths = np.diff(edges) if len(edges) > 1 else []

        figure = go.Figure(
            go.Bar(
                x=centres,
                y=counts,
                width=widths,
                marker_color=theme.PRIMARY,
                opacity=0.85,
                name="Hours",
                customdata=np.stack([centres, widths], axis=-1),
                hovertemplate=(
                    "Around %{customdata[0]:.1f}<br>Hours: %{y:,}<br>"
                    "Bin width: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )
        figure.add_vline(
            x=float(valid.mean()),
            line=dict(color=theme.ACCENT, width=2),
            annotation_text=f"Mean {valid.mean():.1f}",
            annotation_position="top",
            annotation_font=dict(color=theme.ACCENT, size=11),
        )
        figure.add_vline(
            x=float(valid.median()),
            line=dict(color=theme.SECONDARY, width=2, dash="dot"),
            annotation_text=f"Median {valid.median():.1f}",
            annotation_position="bottom right",
            annotation_font=dict(color=theme.SECONDARY, size=11),
        )
        figure.update_xaxes(title=f"{theme.column_label(column)} ({unit or 'value'})")
        figure.update_yaxes(title="Hours")
        theme.style_figure(figure, height=330, showlegend=False)
        st.plotly_chart(figure, width="stretch")

    with right:
        theme.section("The monthly trend")
        monthly = monthly_aggregate(frame, column)
        monthly["date"] = pd.to_datetime(monthly["month"], format="%Y-%m")
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=monthly["date"],
                y=monthly["value_mean"],
                marker_color=theme.SECONDARY,
                opacity=0.85,
                name="Monthly mean",
                customdata=monthly["n"],
                hovertemplate=(
                    "%{x|%b %Y}<br>Mean: %{y:.1f}<br>Hours: %{customdata:,}<extra></extra>"
                ),
            )
        )
        figure.update_xaxes(title="")
        figure.update_yaxes(title=f"Monthly mean ({unit or 'value'})")
        theme.style_figure(figure, height=330, showlegend=False)
        st.plotly_chart(figure, width="stretch")

    # -- daily profile -----------------------------------------------------
    theme.section(
        "The daily profile",
        "Each day collapsed to its own mean, minimum and maximum. Flat stretches are "
        "outages or calm weather; sharp peaks are episodes.",
    )
    daily = daily_profile(frame, column)
    if daily.empty:
        theme.empty_state("No complete days fall inside the selected range.")
    else:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["daily_max"],
                mode="lines",
                name="Daily maximum",
                line=dict(color=theme.MUTED, width=1),
                opacity=0.6,
                hovertemplate="%{x|%d %b %Y}<br>Max: %{y:.1f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["daily_mean"],
                mode="lines",
                name="Daily mean",
                line=dict(color=theme.PRIMARY, width=2.2),
                hovertemplate="%{x|%d %b %Y}<br>Mean: %{y:.1f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["daily_min"],
                mode="lines",
                name="Daily minimum",
                line=dict(color=theme.MUTED, width=1),
                opacity=0.6,
                hovertemplate="%{x|%d %b %Y}<br>Min: %{y:.1f}<extra></extra>",
            )
        )
        figure.update_xaxes(title="")
        figure.update_yaxes(title=f"{theme.column_label(column)} ({unit or 'value'})")
        theme.style_figure(figure, height=340)
        st.plotly_chart(figure, width="stretch")

        theme.download_button(
            daily.rename(
                columns={
                    "date": "Date",
                    "daily_mean": "Daily mean",
                    "daily_min": "Daily minimum",
                    "daily_max": "Daily maximum",
                    "n": "Hours measured",
                }
            ),
            filename=f"ecopulse_daily_{column.replace('(', '_').replace(')', '')}.csv",
        )
