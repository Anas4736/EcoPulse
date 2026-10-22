"""Page 1 - Overview.

What the dataset is, how complete it is, and the headline patterns that the
rest of the dashboard unpacks. Every figure here comes from
:mod:`ecopulse.analysis` run on the cleaned dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ecopulse.analysis import (
    correlation_matrix,
    describe_variables,
    episode_summary,
    hourly_profile,
    seasonal_aggregate,
)
from ecopulse.config import (
    DEFAULT_NO2_THRESHOLD_UG_M3,
    MEASURED_COLUMNS,
    REFERENCE_COLUMNS,
    SENSOR_COLUMNS,
    WEATHER_COLUMNS,
)

from app import theme


def render(context) -> None:
    frame = context.all_frame

    if frame.empty:
        theme.page_header(
            "Overview",
            "EcoPulse analyses a single urban air-quality monitoring station.",
        )
        theme.disclaimer()
        theme.empty_state("The dataset could not be read.")
        return

    timestamps = frame["datetime"]
    description = describe_variables(frame)
    valid = int(description["Valid"].sum())
    cells = len(frame) * len(MEASURED_COLUMNS)
    unique_hours = int(timestamps.nunique())
    span_hours = (
        int((timestamps.max() - timestamps.min()) / pd.Timedelta(hours=1)) + 1
    )
    gaps = max(span_hours - unique_hours, 0)
    duplicates = max(len(frame) - unique_hours, 0)

    # -- product banner ----------------------------------------------------
    theme.hero(
        "EcoPulse",
        "An air-quality analytics dashboard for one monitoring station in northern "
        "Italy. Every chart, correlation and model score is computed from the "
        "cleaned UCI record when a page loads — nothing is precomputed.",
        eyebrow="Environmental data analysis",
        chips=[
            f"<b>{len(frame):,}</b> hourly readings",
            f"<b>{timestamps.min():%b %Y} – {timestamps.max():%b %Y}</b>",
            f"<b>{len(MEASURED_COLUMNS)}</b> measured variables",
        ],
        alert=(
            "Historical record only — UCI data from 2004–2005. "
            "Not live monitoring."
        ),
    )

    # -- headline numbers -------------------------------------------------
    theme.metric_cards(
        [
            {
                "label": "Valid hours",
                "value": theme.fmt_number(len(frame), 0),
                "note": f"{timestamps.min():%d %b %Y} to {timestamps.max():%d %b %Y}",
                "help": "One row per hour. The timeline is continuous: no gaps and no duplicated timestamps.",
            },
            {
                "label": "Measured variables",
                "value": len(MEASURED_COLUMNS),
                "note": f"{len(REFERENCE_COLUMNS)} analysers · {len(SENSOR_COLUMNS)} sensors · {len(WEATHER_COLUMNS)} weather",
                "help": "Five certified reference analysers, five metal-oxide sensor responses and three weather measurements.",
            },
            {
                "label": "Reading completeness",
                "value": theme.fmt_pct(valid / cells, 1),
                "note": f"{theme.fmt_number(valid, 0)} of {theme.fmt_number(cells, 0)} cells",
                "help": "Share of all measured cells that hold a real reading rather than a missing marker.",
            },
            {
                "label": "Timeline gaps",
                "value": theme.fmt_number(gaps, 0),
                "note": f"{theme.fmt_number(duplicates, 0)} duplicate timestamps",
                "help": "Missing hours in the timeline itself. The record is unbroken, so every hour between the first and last reading is present exactly once.",
            },
        ]
    )

    # -- computed key findings --------------------------------------------
    profile = hourly_profile(frame, "NO2(GT)")
    peak = profile.loc[profile["mean"].idxmax()]
    trough = profile.loc[profile["mean"].idxmin()]

    seasons = seasonal_aggregate(frame, "NO2(GT)").set_index("season")
    winter = float(seasons.loc["Winter", "mean"]) if "Winter" in seasons.index else np.nan
    summer = float(seasons.loc["Summer", "mean"]) if "Summer" in seasons.index else np.nan

    correlations = correlation_matrix(frame, columns=list(REFERENCE_COLUMNS))
    values = correlations.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, np.nan)
    upper = np.triu(np.ones_like(values, dtype=bool), k=1)
    masked = np.where(upper, values, np.nan)
    if np.isfinite(masked).any():
        flat = np.nanargmax(np.abs(masked))
        row, column_index = np.unravel_index(flat, masked.shape)
        pair_a = correlations.index[row]
        pair_b = correlations.columns[column_index]
        strongest_r = float(masked[row, column_index])
    else:
        pair_a = pair_b = ""
        strongest_r = np.nan

    episodes = episode_summary(
        frame, "NO2(GT)", DEFAULT_NO2_THRESHOLD_UG_M3, min_duration=1
    )

    theme.section(
        "Key findings",
        "Each figure below is computed from the cleaned record on this page load.",
    )
    theme.findings(
        [
            {
                "label": "Daily cycle",
                "value": f"{theme.fmt_number(peak['mean'])} µg/m³ at {int(peak['hour']):02d}:00",
                "detail": (
                    f"NO2 peaks in the evening and falls to "
                    f"{theme.fmt_number(trough['mean'])} µg/m³ around "
                    f"{int(trough['hour']):02d}:00."
                ),
            },
            {
                "label": "Seasonal swing",
                "value": f"Winter {theme.fmt_number(winter)} vs summer {theme.fmt_number(summer)}",
                "detail": "Mean NO2 in µg/m³. Cold weather raises heating emissions and traps them near the ground.",
            },
            {
                "label": "Shared source",
                "value": f"r = {theme.fmt_number(strongest_r, 3)}",
                "detail": (
                    f"The strongest pairwise correlation between reference analysers "
                    f"is {theme.column_label(pair_a)} against "
                    f"{theme.column_label(pair_b)}."
                ),
            },
            {
                "label": "Limit exceedances",
                "value": f"{episodes['n_exceedance_hours']:,} h in {episodes['n_episodes']:,} episodes",
                "detail": (
                    f"Hours above {DEFAULT_NO2_THRESHOLD_UG_M3:g} µg/m³ NO2, the "
                    "European hourly limit value used as the default threshold."
                ),
            },
        ]
    )

    # -- the daily cycle ---------------------------------------------------
    theme.section(
        "The daily cycle of nitrogen dioxide",
        "Hourly averages of NO2 across the whole record. The bar is the mean and "
        "the line is the median, so a long upper tail cannot hide in one number.",
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=profile["hour"],
            y=profile["mean"],
            marker_color=theme.PRIMARY,
            opacity=0.85,
            name="Mean NO2",
            customdata=np.stack([profile["n"], profile["min"], profile["max"]], axis=-1),
            hovertemplate=(
                "<b>%{x}:00</b><br>Mean: %{y:.1f} µg/m³<br>"
                "Min: %{customdata[1]:.1f}<br>Max: %{customdata[2]:.1f}<br>"
                "Hours measured: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=profile["hour"],
            y=profile["median"],
            mode="lines+markers",
            name="Median NO2",
            line=dict(color=theme.ACCENT, width=2.5),
            hovertemplate="<b>%{x}:00</b><br>Median: %{y:.1f} µg/m³<extra></extra>",
        )
    )
    figure.update_xaxes(
        title="Hour of day",
        tickmode="array",
        tickvals=list(range(0, 24, 2)),
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
    )
    figure.update_yaxes(title="NO2 concentration (µg/m³)")
    theme.style_figure(figure, height=340)
    st.plotly_chart(figure, width="stretch")

    # -- the seasonal cycle ------------------------------------------------
    theme.section(
        "The seasonal cycle",
        "The same kind of average, grouped by season instead of by hour. The error "
        "bar is one standard deviation.",
    )
    seasons = seasonal_aggregate(frame, "NO2(GT)")
    order = ["Winter", "Spring", "Summer", "Autumn"]
    seasons = seasons.set_index("season").reindex(order).reset_index()

    figure = px.bar(
        seasons,
        x="season",
        y="mean",
        color="season",
        color_discrete_map=theme.SEASON_COLORS,
        error_y="std",
        labels={"mean": "Mean NO2 (µg/m³)", "season": ""},
        custom_data=["n", "min", "max", "median"],
        height=300,
    )
    figure.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>Mean: %{y:.1f} µg/m³<br>Median: %{customdata[3]:.1f}<br>"
            "Min: %{customdata[1]:.1f}<br>Max: %{customdata[2]:.1f}<br>"
            "Hours: %{customdata[0]:,}<extra></extra>"
        )
    )
    theme.style_figure(figure, showlegend=False)
    st.plotly_chart(figure, width="stretch")

    st.markdown(
        f"NO2 is highest in **winter** ({theme.fmt_number(winter)} µg/m³) and lowest "
        f"in **summer** ({theme.fmt_number(summer)} µg/m³). This is also why the "
        "sensor-drift analysis compares the same calendar month in different years: "
        "a naive early-versus-late comparison would see the season instead of the "
        "sensor."
    )

    # -- supporting detail, kept out of the way -----------------------------
    with st.expander("What the dataset contains"):
        st.markdown(
            "The UCI documentation describes one monitoring site with five reference "
            "analysers, five metal-oxide sensors and three weather instruments. The "
            "five **reference analysers** measure a real concentration: CO, "
            "non-methane hydrocarbons, benzene, NOx and NO2. The five **sensor "
            "responses** (the `PT08.*` columns) are raw readings from cheap "
            "metal-oxide devices, given in device units rather than concentrations. "
            "The three **weather** columns are temperature, relative humidity and a "
            "scaled humidity measure."
        )
        composition = pd.DataFrame(
            [
                {
                    "Group": "Reference analysers",
                    "Count": len(REFERENCE_COLUMNS),
                    "Members": ", ".join(REFERENCE_COLUMNS),
                },
                {
                    "Group": "Sensor responses",
                    "Count": len(SENSOR_COLUMNS),
                    "Members": ", ".join(SENSOR_COLUMNS),
                },
                {
                    "Group": "Weather",
                    "Count": len(WEATHER_COLUMNS),
                    "Members": ", ".join(WEATHER_COLUMNS),
                },
            ]
        )
        st.dataframe(
            composition,
            width="stretch",
            hide_index=True,
            column_config={
                "Group": st.column_config.TextColumn(width="medium"),
                "Count": st.column_config.NumberColumn(width="small"),
                "Members": st.column_config.TextColumn(width="large"),
            },
        )

    with st.expander("Completeness by variable"):
        completeness = description.copy()
        completeness["Label"] = completeness["Column"].map(theme.column_label)
        completeness["Role"] = completeness["Column"].map(theme.column_role)
        completeness["Valid %"] = 100.0 - completeness["Missing %"]

        figure = px.bar(
            completeness.sort_values("Valid %"),
            x="Valid %",
            y="Label",
            orientation="h",
            color="Role",
            color_discrete_map={
                "reference": theme.PRIMARY,
                "sensor": theme.SECONDARY,
                "weather": theme.ACCENT,
            },
            custom_data=["Column", "Valid", "Missing"],
            labels={"Valid %": "Valid readings (%)", "Label": ""},
            height=320,
        )
        figure.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>Column: %{customdata[0]}<br>"
                "Valid readings: %{customdata[1]:,}<br>Missing: %{customdata[2]:,}<br>"
                "Complete: %{x:.1f}%<extra></extra>"
            )
        )
        theme.style_figure(figure, showlegend=True)
        st.plotly_chart(figure, width="stretch")

        worst = completeness.nsmallest(1, "Valid %").iloc[0]
        theme.notice(
            f"**`{worst['Column']}` is the weak spot.** Only "
            f"{theme.fmt_pct(1 - worst['Missing %'] / 100, 1)} of its readings are "
            "present, because roughly 90% of the hours have no non-methane "
            "hydrocarbon reference value at all. EcoPulse describes that variable but "
            "keeps it out of the machine-learning models.",
            kind="warn",
        )

    with st.expander("How the reference analysers move together"):
        correlations = correlation_matrix(frame, columns=list(REFERENCE_COLUMNS))
        labels = [theme.column_label(column) for column in correlations.columns]

        figure = go.Figure(
            data=go.Heatmap(
                z=correlations.to_numpy(),
                x=labels,
                y=labels,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(correlations.to_numpy(), 2),
                texttemplate="%{text}",
                textfont=dict(size=12),
                hovertemplate=(
                    "<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
                ),
                colorbar=dict(title="r", thickness=12, len=0.7),
            )
        )
        theme.style_figure(figure, height=380, showlegend=False)
        st.plotly_chart(figure, width="stretch")

        benzene = correlations.loc["C6H6(GT)", "NO2(GT)"]
        theme.notice(
            f"Benzene and NO2 correlate at **r = {benzene:.2f}** over the hours where "
            "both are present. They share a common cause — road traffic — rather than "
            "causing each other; the Pollutant Relationships page unpacks that in "
            "detail.",
            kind="info",
        )

    with st.expander("Where to go next"):
        st.markdown(
            "| Page | What it answers |\n"
            "|---|---|\n"
            "| **Pollution Explorer** | How one pollutant behaves hour by hour, day by day and season by season. |\n"
            "| **Pollutant Relationships** | Which variables move together, and how well each sensor tracks its reference analyser. |\n"
            "| **Data Quality** | What was missing in the original file, how the gaps group together, and what is still missing. |\n"
            "| **Anomaly Detection** | The hours that exceed a threshold, grouped into episodes. |\n"
            "| **Benzene Regression** | How well benzene can be estimated from other signals, and what a leakage-free evaluation looks like. |\n"
            "| **Methodology & About** | How every number was produced, and the limits of what the data can support. |"
        )
