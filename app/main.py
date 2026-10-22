"""EcoPulse - Air Quality Monitoring and Data Analysis System.

The Streamlit entry point. It sets up the page, draws the sidebar (navigation
plus the global filters) and hands each view a context object holding the
filtered data and the cached analysis results.

Run it with::

    python -m streamlit run app/main.py
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Make "src" and the project root importable when the app is started from
# the project root without the package being installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app import data as data_access  # noqa: E402
from app import theme  # noqa: E402

#: The reference analysers a visitor can explore.
POLLUTANT_CHOICES = ["NO2(GT)", "CO(GT)", "NOx(GT)", "C6H6(GT)", "NMHC(GT)"]


@dataclass
class PageFilters:
    """Which global filters a page actually uses."""

    date: bool = False
    pollutant: bool = False


@dataclass
class PageSpec:
    """One entry in the navigation."""

    title: str
    module: str
    filters: PageFilters = field(default_factory=PageFilters)
    blurb: str = ""


#: The navigation, in the order it appears in the sidebar.
PAGES: dict[str, PageSpec] = {
    "Overview": PageSpec(
        "Overview",
        "app.views.overview",
        PageFilters(date=False, pollutant=False),
        "What the dataset contains and what the headline findings are.",
    ),
    "Pollution Explorer": PageSpec(
        "Pollution Explorer",
        "app.views.pollution_explorer",
        PageFilters(date=True, pollutant=True),
        "Hour by hour, day by day, for any pollutant in the record.",
    ),
    "Pollutant Relationships": PageSpec(
        "Pollutant Relationships",
        "app.views.relationships",
        PageFilters(date=True, pollutant=True),
        "How the pollutants and the sensors move together.",
    ),
    "Data Quality": PageSpec(
        "Data Quality",
        "app.views.data_quality",
        PageFilters(date=False, pollutant=False),
        "What was missing before cleaning, and what is still missing.",
    ),
    "Anomaly Detection": PageSpec(
        "Anomaly Detection",
        "app.views.anomaly_detection",
        PageFilters(date=True, pollutant=True),
        "Pollution episodes above a threshold, hour by hour.",
    ),
    "Benzene Regression": PageSpec(
        "Benzene Regression",
        "app.views.benzene_regression",
        PageFilters(date=False, pollutant=False),
        "How well benzene can be estimated, and on what evidence.",
    ),
    "Methodology & About": PageSpec(
        "Methodology & About",
        "app.views.methodology",
        PageFilters(date=False, pollutant=False),
        "How every number was produced, and what the limits are.",
    ),
}


@dataclass
class AppContext:
    """Everything a view needs, with the expensive results loaded on demand."""

    frame: pd.DataFrame
    all_frame: pd.DataFrame
    pollutant: str
    start: date
    end: date
    filtered: bool

    @property
    def quality(self):
        return data_access.load_quality_report()

    @property
    def benchmark(self):
        return data_access.load_benchmark()

    @property
    def leakage(self):
        return data_access.load_leakage_report()

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


def _render_filters(spec: PageSpec, frame: pd.DataFrame) -> dict:
    """Draw the filter widgets a page needs and return their values."""
    values: dict = {}

    if spec.filters.date:
        theme.sidebar_label("Filters")
        timestamps = frame["datetime"]
        first = timestamps.min().date()
        last = timestamps.max().date()
        chosen = st.sidebar.date_input(
            "Date range",
            value=(first, last),
            min_value=first,
            max_value=last,
            format="DD/MM/YYYY",
            help="Every chart on this page uses only the hours inside this range.",
        )
        if isinstance(chosen, (tuple, list)):
            if len(chosen) == 2:
                start, end = chosen
            else:
                start, end = chosen[0], chosen[0]
        else:
            start = end = chosen
        values["start"] = start
        values["end"] = end

    if spec.filters.pollutant:
        if not spec.filters.date:
            theme.sidebar_label("Filters")
        values["pollutant"] = st.sidebar.selectbox(
            "Pollutant",
            POLLUTANT_CHOICES,
            format_func=theme.column_label,
            help=(
                "Concentration measured by the certified reference analyser. "
                "Values are averaged over each hour."
            ),
        )

    return values


def _apply_filters(frame: pd.DataFrame, values: dict) -> tuple[pd.DataFrame, bool]:
    """Cut the frame down to the selected dates."""
    if "start" not in values:
        return frame, False

    stamps = frame["datetime"]
    start = pd.Timestamp(values["start"])
    end = pd.Timestamp(values["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if start > end:
        start, end = end, start
    mask = (stamps >= start) & (stamps <= end)
    return frame.loc[mask].reset_index(drop=True), bool(values["start"] != values["end"])


def main() -> None:
    st.set_page_config(
        page_title="EcoPulse - Air Quality Analysis",
        page_icon=str(PROJECT_ROOT / "app" / "static" / "favicon.svg"),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject_css()

    with st.sidebar:
        theme.sidebar_brand()
        st.caption(
            "UCI Air Quality dataset — a fixed historical record from "
            "March 2004 to April 2005."
        )
        theme.sidebar_label("Navigation")
        page_name = st.radio(
            "Navigate",
            list(PAGES),
            label_visibility="collapsed",
        )
        st.divider()

    spec = PAGES[page_name]

    try:
        view = importlib.import_module(spec.module)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(
            f"**The '{spec.title}' page could not be loaded.** "
            "This is a bug in the dashboard code, not in the data."
        )
        st.exception(exc)
        return

    try:
        all_frame = data_access.load_frame()
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(
            "**The dataset could not be loaded.** EcoPulse reads the original "
            "workbook from `Data/AirQualityUCI.xlsx` and cleans it before "
            "anything is shown. The error below explains what went wrong."
        )
        st.exception(exc)
        st.info(
            "Try running `python -m ecopulse.cleaning` to rebuild "
            "`output/airquality_clean.csv`, then restart the app."
        )
        return

    with st.sidebar:
        values = _render_filters(spec, all_frame)
        st.divider()
        theme.sidebar_footer(
            "<strong>Historical data only.</strong> A fixed 2004–2005 record. "
            "No live feed and no current air-quality readings."
        )

    frame, filtered = _apply_filters(all_frame, values)
    pollutant = values.get("pollutant", POLLUTANT_CHOICES[0])

    context = AppContext(
        frame=frame,
        all_frame=all_frame,
        pollutant=pollutant,
        start=values.get("start", all_frame["datetime"].min().date()),
        end=values.get("end", all_frame["datetime"].max().date()),
        filtered=filtered,
    )

    try:
        view.render(context)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(
            f"**The '{spec.title}' page could not be drawn.** "
            "The error below is shown in full so it can be reported."
        )
        st.exception(exc)
    theme.footer()


main()
