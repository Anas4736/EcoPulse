"""Cached data access for the EcoPulse dashboard.

Every figure and table on every page is derived from the original workbook
through the tested modules in :mod:`ecopulse`. This module only caches those
results so the dashboard stays responsive.

The expensive pieces are:

* reading and cleaning the workbook (a second or two), and
* fitting the benzene models (roughly twenty seconds).

Both are wrapped in :func:`streamlit.cache_data`, so they are computed once
and reused for every visitor and every page.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ecopulse.cleaning import load_clean_dataset
from ecopulse.config import SOURCE_XLSX
from ecopulse.loader import load_raw
from ecopulse.ml import (
    REFERENCE_ASSISTED_FEATURE_SET,
    run_benzene_benchmark,
    run_leakage_diagnostic,
)
from ecopulse.quality import build_quality_report


@st.cache_data(show_spinner="Reading and cleaning the original workbook ...")
def load_frame() -> pd.DataFrame:
    """The cleaned dataset."""
    return load_clean_dataset().frame


@st.cache_data(show_spinner="Building the data-quality report ...")
def load_quality_report():
    """The before/after data-quality report, built from the raw workbook."""
    raw = load_raw()
    clean = load_frame()
    return build_quality_report(raw, clean, str(SOURCE_XLSX))


@st.cache_data(
    show_spinner="Fitting the benzene models on the real data (done once, then cached) ..."
)
def load_benchmark():
    """The benzene regression benchmark.

    All three feature tiers are fitted so the Benzene Regression page can
    show how much each group of predictors contributes.  The headline tier is
    the sensor-only one; the reference-assisted tier is fitted as a labelled
    comparison.
    """
    frame = load_frame()
    return run_benzene_benchmark(
        frame,
        feature_sets=("weather", "sensors", REFERENCE_ASSISTED_FEATURE_SET),
        cv_splits=5,
    )


@st.cache_data(
    show_spinner="Running the leakage diagnostic on the real data (done once, then cached) ..."
)
def load_leakage_report():
    """What the excluded proxy column would have done to the headline score."""
    return run_leakage_diagnostic(load_frame(), cv_splits=5)


__all__ = [
    "load_frame",
    "load_quality_report",
    "load_benchmark",
    "load_leakage_report",
]
