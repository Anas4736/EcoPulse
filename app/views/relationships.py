"""Page 3 - Pollutant Relationships.

How the variables move together, and how well each metal-oxide sensor
tracks the reference analyser it is supposed to represent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ecopulse.analysis import correlation_matrix, pairwise_counts, sensor_reference_summary
from ecopulse.config import (
    MEASURED_COLUMNS,
    REFERENCE_COLUMNS,
    SENSOR_REFERENCE_PAIRS,
)

from app import theme


def render(context) -> None:
    frame = context.frame
    focus = context.pollutant

    theme.page_header(
        "Pollutant Relationships",
        "Correlation between the measured variables, and the behaviour of the five "
        "metal-oxide sensors against the certified analysers they are paired with. "
        "The date range in the sidebar applies to this page.",
    )
    theme.disclaimer()

    if frame.empty:
        theme.empty_state(
            "The selected date range contains no hours. Widen it in the sidebar."
        )
        return

    # -- correlation heatmap -----------------------------------------------
    theme.section(
        "The correlation matrix",
        "Pearson or Spearman correlation, computed only on the hours where both "
        "variables have a reading. The number of overlapping hours is listed "
        "separately, because a high correlation on 900 hours means something quite "
        "different from the same correlation on 8,900.",
    )

    method = st.radio(
        "Correlation method",
        ["pearson", "spearman"],
        horizontal=True,
        help=(
            "Pearson measures linear association. Spearman measures the same "
            "thing on the ranks, so it is not thrown off by a few extreme values."
        ),
    )
    variables = st.multiselect(
        "Variables to include",
        options=list(MEASURED_COLUMNS),
        default=list(REFERENCE_COLUMNS) + ["T", "RH"],
        format_func=theme.column_label,
        help="Only the reference analysers and the weather variables are shown by default.",
    )

    if len(variables) < 2:
        theme.empty_state("Choose at least two variables to build a correlation matrix.")
        return

    correlations = correlation_matrix(frame, columns=variables, method=method)
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
                "<b>%{y}</b> vs <b>%{x}</b><br>"
                f"{method.capitalize()} correlation: %{{z:.3f}}<extra></extra>"
            ),
            colorbar=dict(title="r", thickness=12, len=0.75),
        )
    )
    theme.style_figure(figure, height=max(380, 46 * len(variables) + 120), showlegend=False)
    st.plotly_chart(figure, width="stretch")

    # -- overlapping sample sizes ------------------------------------------
    with st.expander("How many hours overlap behind each correlation?"):
        st.caption(
            "A correlation is only as trustworthy as the number of hours behind it. "
            "Cells are the number of hours in which both the row and the column "
            "variable have a reading."
        )
        overlaps = pairwise_counts(frame, columns=variables)
        overlaps.index.name = "Variable"
        st.dataframe(
            overlaps.style.format("{:.0f}", na_rep="0"),
            width="stretch",
        )

    # -- scatter ------------------------------------------------------------
    theme.section(
        "Two variables against each other",
        "The relationship between the selected pollutant and any other variable, "
        "with the least-squares line fitted to the paired readings.",
    )

    others = [column for column in MEASURED_COLUMNS if column != focus]
    other = st.selectbox(
        "Compare against",
        options=others,
        index=others.index("CO(GT)") if "CO(GT)" in others else 0,
        format_func=theme.column_label,
    )

    paired = frame[[focus, other]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(paired) < 3:
        theme.empty_state(
            f"There are only {len(paired)} hours where both {focus} and {other} were "
            "measured, which is too few to fit a line."
        )
    else:
        fit = stats.linregress(paired[other], paired[focus])
        r_value = paired[focus].corr(paired[other])
        rho = paired[focus].corr(paired[other], method="spearman")

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=paired[other],
                y=paired[focus],
                mode="markers",
                name="Paired hours",
                marker=dict(
                    size=4,
                    color=theme.PRIMARY,
                    opacity=0.28,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"{theme.column_label(other)}: %{{x:.2f}}<br>"
                    f"{theme.column_label(focus)}: %{{y:.2f}}<extra></extra>"
                ),
            )
        )
        x_values = np.linspace(paired[other].min(), paired[other].max(), 60)
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=fit.intercept + fit.slope * x_values,
                mode="lines",
                name=f"Fitted line (slope {fit.slope:.3f})",
                line=dict(color=theme.DANGER, width=2.4),
                hovertemplate="Fitted: %{y:.2f}<extra></extra>",
            )
        )
        figure.update_xaxes(title=theme.column_label(other))
        figure.update_yaxes(title=theme.column_label(focus))
        theme.style_figure(figure, height=400)
        st.plotly_chart(figure, width="stretch")

        theme.metric_cards(
            [
                {
                    "label": "Pearson r",
                    "value": theme.fmt_number(r_value, 3),
                    "note": "linear association",
                    "help": "Pearson correlation on the paired hours.",
                },
                {
                    "label": "Spearman rho",
                    "value": theme.fmt_number(rho, 3),
                    "note": "rank association",
                    "help": "Spearman rank correlation on the paired hours.",
                },
                {
                    "label": "Paired hours",
                    "value": theme.fmt_number(len(paired), 0),
                    "note": "both variables present",
                    "help": "The number of hours the correlation is based on.",
                },
                {
                    "label": "Regression slope",
                    "value": theme.fmt_number(fit.slope, 3),
                    "note": f"p = {fit.pvalue:.2e}",
                    "help": "How much the focus variable changes per unit of the other variable.",
                },
            ]
        )
        st.caption(
            f"Based on {len(paired):,} hours in which both variables were measured."
        )

    # -- sensors against their reference analysers -------------------------
    theme.section(
        "Each sensor against its reference analyser",
        "The metal-oxide sensors report raw device responses, not concentrations. "
        "These statistics describe how faithfully each sensor follows the certified "
        "analyser it is paired with, over the hours where both were recorded.",
    )

    rows = []
    for sensor, reference in SENSOR_REFERENCE_PAIRS:
        if sensor not in frame.columns or reference not in frame.columns:
            continue
        summary = sensor_reference_summary(frame, sensor, reference)
        rows.append(
            {
                "Sensor": sensor,
                "Sensor reads": theme.column_label(sensor),
                "Reference": reference,
                "Reference reads": theme.column_label(reference),
                "Paired hours": summary["n_pairs"],
                "Pearson r": summary["pearson_r"],
                "Spearman rho": summary["spearman_rho"],
                "Slope": summary["slope"],
                "R2": summary["r_squared"],
                "RMSE": summary["rmse"],
                "Bias": summary["bias"],
            }
        )

    if rows:
        table = pd.DataFrame(rows)
        st.dataframe(
            table.style.format(
                {
                    "Pearson r": "{:.3f}",
                    "Spearman rho": "{:.3f}",
                    "Slope": "{:.5f}",
                    "R2": "{:.3f}",
                    "RMSE": "{:.3f}",
                    "Bias": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        strongest = table.loc[table["Pearson r"].idxmax()]
        weakest = table.loc[table["Pearson r"].idxmin()]
        theme.notice(
            f"**The sensors are not equally good.** "
            f"`{strongest['Sensor']}` tracks `{strongest['Reference']}` closely "
            f"(r = {strongest['Pearson r']:.3f} over {strongest['Paired hours']:,} hours), "
            f"while `{weakest['Sensor']}` against `{weakest['Reference']}` barely "
            f"follows it at all (r = {weakest['Pearson r']:.3f}). A device that reads "
            "badly is not a device whose raw output can be used as a concentration.",
            kind="info",
        )

        # Scatter for one chosen pair.
        chosen = st.selectbox(
            "Show one pair in detail",
            options=[f"{sensor} vs {reference}" for sensor, reference in SENSOR_REFERENCE_PAIRS],
            format_func=lambda text: text.replace("PT08.S", "Sensor S"),
        )
        sensor_name, reference_name = chosen.split(" vs ")
        paired = frame[[sensor_name, reference_name]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(paired) >= 3:
            fit = stats.linregress(paired[sensor_name], paired[reference_name])
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=paired[sensor_name],
                    y=paired[reference_name],
                    mode="markers",
                    name="Paired hours",
                    marker=dict(size=4, color=theme.SECONDARY, opacity=0.3),
                    hovertemplate=(
                        f"{theme.column_label(sensor_name)}: %{{x:.1f}}<br>"
                        f"{theme.column_label(reference_name)}: %{{y:.2f}}<extra></extra>"
                    ),
                )
            )
            x_values = np.linspace(paired[sensor_name].min(), paired[sensor_name].max(), 60)
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=fit.intercept + fit.slope * x_values,
                    mode="lines",
                    name="Calibration line",
                    line=dict(color=theme.DANGER, width=2.4),
                    hovertemplate="Calibrated: %{y:.2f}<extra></extra>",
                )
            )
            figure.update_xaxes(title=f"{theme.column_label(sensor_name)} (device units)")
            figure.update_yaxes(title=f"{theme.column_label(reference_name)}")
            theme.style_figure(figure, height=380)
            st.plotly_chart(figure, width="stretch")

    theme.notice(
        "**A high correlation is not a calibration.** It says two instruments rise "
        "and fall together; it does not say the sensor's raw number can be read as "
        "a concentration. The slope and the bias in the table above are what turn "
        "one into the other, and the Benzene Regression page is careful about the "
        "difference.",
        kind="warn",
    )
