"""Page 4 - Data Quality.

What was missing in the original workbook, how the cleaning step changed it,
and the pattern in which the gaps arrive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ecopulse.analysis import describe_variables, missingness_by_period
from ecopulse.config import MEASURED_COLUMNS
from ecopulse.quality import before_after_table, summary_metrics

from app import theme


def render(context) -> None:
    theme.page_header(
        "Data Quality",
        "The original workbook marks every missing measurement with the value "
        "-200. This page shows what that looked like before cleaning, what the "
        "cleaning step did, and the pattern in which the remaining gaps arrive.",
    )
    theme.disclaimer()

    try:
        report = context.quality
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error("**The data-quality report could not be built.**")
        st.exception(exc)
        return

    metrics = summary_metrics(report)

    # -- headline numbers ---------------------------------------------------
    theme.metric_cards(
        [
            {
                "label": "Rows read",
                "value": theme.fmt_number(metrics["rows_read"], 0),
                "note": f"{theme.fmt_number(metrics['rows_after_clean'], 0)} kept after cleaning",
                "help": "Cleaning converts the -200 markers into proper missing values; it never deletes a row.",
            },
            {
                "label": "Rows retained",
                "value": theme.fmt_pct(metrics["rows_retained_pct"] / 100, 1),
                "note": "no rows dropped",
                "help": "Every hourly row is kept, because the -200 markers are missing readings rather than bad rows.",
            },
            {
                "label": "-200 markers found",
                "value": theme.fmt_number(metrics["total_sentinel_markers"], 0),
                "note": f"{theme.fmt_number(metrics['sentinel_pct_of_cells'], 2)}% of all cells",
                "help": "The sentinel value the UCI file uses for a missing measurement. It is never a physical reading.",
            },
            {
                "label": "Duplicate timestamps",
                "value": theme.fmt_number(metrics["duplicate_timestamps_removed"], 0),
                "note": "none removed",
                "help": "Each hour appears exactly once in the record.",
            },
            {
                "label": "Timeline gaps",
                "value": theme.fmt_number(metrics["gap_count"], 0),
                "note": "largest gap 0 h",
                "help": "Missing hours in the timeline itself, as opposed to missing readings within an hour.",
            },
            {
                "label": "Non-physical values",
                "value": theme.fmt_number(metrics["non_physical_total"], 0),
                "note": "no negative concentrations",
                "help": "Readings that could not be physical, such as a negative concentration. There are none once the sentinels are removed.",
            },
        ]
    )

    # -- what cleaning did --------------------------------------------------
    theme.section(
        "Before and after cleaning",
        "The sentinel count before cleaning and the missing count after it are the "
        "same number by construction: the cleaning step does not invent or lose "
        "information, it only relabels it correctly.",
    )
    table = before_after_table(report)
    with st.expander(f"Full before-and-after table ({len(table):,} variables)"):
        st.dataframe(
            table.style.format(
                {
                    "Raw -200 markers": "{:.0f}",
                    "Raw missing %": "{:.2f}",
                    "Missing after cleaning": "{:.0f}",
                    "Missing after cleaning %": "{:.2f}",
                    "Valid values": "{:.0f}",
                    "Min": "{:.2f}",
                    "Max": "{:.2f}",
                    "Mean": "{:.3f}",
                    "Median": "{:.2f}",
                },
                na_rep="-",
            ),
            width="stretch",
            hide_index=True,
        )
        theme.download_button(table, filename="ecopulse_data_quality_before_after.csv")

    # -- the missingness groups --------------------------------------------
    theme.section(
        "The gaps arrive in groups, not at random",
        "If missing readings were scattered randomly they would fall on each column "
        "independently. Instead, whole sets of columns fail together, which means "
        "the gaps are instrument behaviour rather than noise.",
    )

    groups = report.missing_pattern_groups
    if groups:
        rows = []
        for index, group in enumerate(groups, start=1):
            columns = list(group)
            rows.append(
                {
                    "Group": index,
                    "Columns in the group": len(columns),
                    "Members": ", ".join(columns),
                    "Total missing hours": int(
                        _group_missing_hours(context, columns)
                    ),
                }
            )
        group_table = pd.DataFrame(rows)
        st.dataframe(
            group_table.style.format({"Total missing hours": "{:.0f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Two columns join the same group when they are missing in the same "
            "hours far more often than chance would explain."
        )
    else:
        st.info("No missingness groups were detected.")

    # -- missingness by column ---------------------------------------------
    theme.section(
        "How much is missing, column by column?",
        "The share of readings that are absent for each variable, before and after "
        "grouping, so the scale of each gap is visible.",
    )
    description = describe_variables(context.all_frame)
    order = description.sort_values("Missing %", ascending=True)
    labels = [theme.column_label(column) for column in order["Column"]]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=order["Missing %"],
            y=labels,
            orientation="h",
            name="Missing readings",
            marker_color=[
                theme.DANGER if value > 50 else (theme.ACCENT if value > 10 else theme.SECONDARY)
                for value in order["Missing %"]
            ],
            customdata=np.stack(
                [order["Valid"], order["Missing"], order["Column"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Column: %{customdata[2]}<br>"
                "Missing: %{customdata[1]:,} hours (%{x:.2f}%)<br>"
                "Valid: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title="Missing readings (%)")
    figure.update_yaxes(title="")
    theme.style_figure(figure, height=max(320, 34 * len(order) + 90), showlegend=False)
    st.plotly_chart(figure, width="stretch")

    nmhc = description[description["Column"] == "NMHC(GT)"]
    if not nmhc.empty:
        share = float(nmhc["Missing %"].iloc[0])
        theme.notice(
            f"**`NMHC(GT)` is missing {share:.1f}% of the time.** It is the one "
            "variable the reference analysers barely covered during this campaign. "
            "EcoPulse still describes and charts it, but it is excluded from the "
            "machine-learning models, because a model trained on the 10% of hours "
            "that do have a value would be learning from almost nothing.",
            kind="warn",
        )

    # -- missingness through time ------------------------------------------
    theme.section(
        "When do the gaps happen?",
        "Missing readings by month. The device outages are not spread evenly through "
        "the year, which is one of the reasons the sensor-drift analysis has to "
        "compare the same calendar month across years.",
    )
    by_month = missingness_by_period(context.all_frame, period="month")
    if by_month.empty:
        st.info("No monthly missingness could be computed.")
    else:
        melted = by_month.melt(
            id_vars=["period"],
            value_vars=[c for c in by_month.columns if c.startswith("missing_pct_")],
            var_name="variable",
            value_name="missing_pct",
        )
        melted["variable"] = melted["variable"].str.replace("missing_pct_", "", regex=False)
        melted["label"] = melted["variable"].map(theme.column_label)
        melted["period"] = pd.to_datetime(melted["period"], format="%Y-%m", errors="coerce")
        melted = melted.dropna(subset=["period"])

        figure = px.imshow(
            melted.pivot_table(
                index="label", columns="period", values="missing_pct", aggfunc="first"
            ),
            color_continuous_scale="YlOrRd",
            zmin=0,
            zmax=100,
            labels={"color": "Missing (%)"},
            height=max(300, 32 * melted["label"].nunique() + 110),
        )
        figure.update_xaxes(title="", tickformat="%b %Y")
        figure.update_yaxes(title="")
        theme.style_figure(figure, showlegend=False)
        st.plotly_chart(figure, width="stretch")

    # -- outage blocks ------------------------------------------------------
    theme.section(
        "Device outages",
        "Contiguous runs of hours in which a group of columns was missing "
        "altogether. These are the periods when the monitoring equipment was not "
        "reporting, and they are the reason 13.7% of all cells are empty.",
    )
    blocks = report.outages
    if not blocks:
        st.info("No outage blocks were detected.")
    else:
        rows = []
        for block in blocks:
            row = block.as_row()
            columns = _split_columns(row.get("columns"))
            rows.append(
                {
                    "Start": pd.Timestamp(row.get("start", "")).strftime("%d %b %Y %H:%M"),
                    "End": pd.Timestamp(row.get("end", "")).strftime("%d %b %Y %H:%M"),
                    "Hours": row.get("hours", 0),
                    "Columns affected": len(columns),
                    "Columns": ", ".join(columns),
                }
            )
        outage_table = pd.DataFrame(rows).sort_values("Hours", ascending=False)

        with st.expander(f"The {min(len(outage_table), 100)} longest outage blocks"):
            st.dataframe(
                outage_table.head(100).style.format({"Hours": "{:.0f}"}),
                width="stretch",
                hide_index=True,
            )
            if len(outage_table) > 100:
                st.caption(
                    f"Showing the 100 longest of {len(outage_table):,} blocks. "
                    "The complete list is available in the download."
                )
            theme.download_button(outage_table, filename="ecopulse_outage_blocks.csv")

        # How the blocks group by the columns they took out.
        summary = (
            outage_table.groupby("Columns")["Hours"]
            .agg(blocks="count", hours="sum")
            .reset_index()
            .sort_values("hours", ascending=False)
        )
        summary["Columns affected"] = summary["Columns"].map(
            lambda text: len(_split_columns(text))
        )
        st.markdown("**Outage blocks grouped by the columns they took out**")
        st.dataframe(
            summary[["Columns affected", "Columns", "blocks", "hours"]].style.format(
                {"blocks": "{:.0f}", "hours": "{:.0f}"}
            ),
            width="stretch",
            hide_index=True,
        )

        device = summary[summary["Columns affected"] >= 9]
        if not device.empty:
            device_hours = int(device["hours"].iloc[0])
            device_blocks = int(device["blocks"].iloc[0])
            theme.notice(
                f"**{device_blocks} blocks, {device_hours} hours in total.** In those "
                "periods the device stopped reporting and nine variables went missing "
                "at once - the sensors, the weather instruments and most of the "
                "analysers. This is why the raw file contains so many -200 markers, "
                "and it is the single largest source of missingness in the dataset.",
                kind="info",
            )


def _split_columns(value) -> list[str]:
    """The outage block stores its columns as a comma-joined string."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if value is None:
        return []
    return list(value)


def _group_missing_hours(context, columns: list[str]) -> int:
    """Hours in which every column of a group is missing at the same time."""
    frame = context.all_frame
    subset = [column for column in columns if column in frame.columns]
    if not subset:
        return 0
    return int(frame[subset].isna().all(axis=1).sum())
