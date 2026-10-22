"""Page 7 - Methodology & About.

Everything a reader needs to check the numbers on the other six pages: where
the data came from, how it was cleaned, how each statistic was computed, and
what the analysis cannot support.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import pandas as pd
import streamlit as st

from ecopulse.config import (
    COLUMN_METADATA,
    PRIMARY_TARGET,
    RAW_DIR,
    SENTINEL_VALUE,
    SOURCE_HASHES,
)
from ecopulse.ml import (
    FEATURE_SET_DESCRIPTIONS,
    FEATURE_SET_LABELS,
    PRIMARY_FEATURE_SET,
    REFERENCE_ASSISTED_FEATURE_SET,
)

from app import theme


def _feature_set_text(name: str) -> str:
    """A short sentence describing one feature tier."""
    return FEATURE_SET_DESCRIPTIONS.get(name, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def render(context) -> None:
    theme.page_header(
        "Methodology & About",
        "Where the data comes from, how it was handled, and exactly how each "
        "figure in this dashboard was produced - so that nothing here has to be "
        "taken on trust.",
    )
    theme.disclaimer()

    # -- provenance --------------------------------------------------------
    theme.section(
        "Provenance",
        "The dashboard reads the original UCI files, checks that they are "
        "byte-for-byte the ones the project was built against, and derives every "
        "number live. Nothing is precomputed and pasted in.",
    )

    st.markdown(
        "- **Dataset:** UCI Machine Learning Repository, *Air Quality* (dataset ID "
        "360), contributed by S. De Vito, E. Massera, M. Piga, L. Martinotto and "
        "G. Di Francia.\n"
        "- **Collection:** a gas multi-sensor device deployed at a road site in an "
        "Italian city, co-located with a certified reference analyser.\n"
        "- **Period:** 10 March 2004 to 4 April 2005, hourly, 9,357 rows.\n"
        "- **Licence and citation:** see the repository entry for ID 360."
    )

    hash_rows = []
    for name, expected in SOURCE_HASHES.items():
        path = RAW_DIR / name
        actual = _sha256(path) if path.exists() else "(file missing)"
        hash_rows.append(
            {
                "File": name,
                "Bytes": path.stat().st_size if path.exists() else 0,
                "SHA-256": actual,
                "Matches expected": actual == expected,
            }
        )
    hash_table = pd.DataFrame(hash_rows)
    verified = int(hash_table["Matches expected"].sum())
    frame = context.all_frame
    timestamps = frame["datetime"] if "datetime" in frame.columns else None
    theme.metric_cards(
        [
            {
                "label": "Rows in the record",
                "value": theme.fmt_number(len(frame), 0),
                "note": (
                    f"{timestamps.min():%b %Y} – {timestamps.max():%b %Y}"
                    if timestamps is not None
                    else "cleaned dataset"
                ),
                "help": "Hourly rows in the cleaned dataset used by every page.",
            },
            {
                "label": "Variables described",
                "value": len(COLUMN_METADATA),
                "note": "columns with a unit and role",
                "help": "Every measured column is documented in the dictionary below.",
            },
            {
                "label": "Source files verified",
                "value": f"{verified} / {len(hash_table)}",
                "note": "SHA-256 digests",
                "help": "The original files are hashed at page load and compared with the expected digests.",
            },
            {
                "label": "Missing-value marker",
                "value": f"{SENTINEL_VALUE:g}",
                "note": "replaced with a true null",
                "help": "The raw file's sentinel for a missing reading. It is never treated as a measurement.",
            },
        ]
    )
    st.dataframe(hash_table, width="stretch", hide_index=True)
    if bool(hash_table["Matches expected"].all()):
        theme.notice(
            "**Both source files match their expected SHA-256 digests**, so the "
            "figures shown here were computed from the exact files the project was "
            "verified against.",
            kind="ok",
        )
    else:
        theme.notice(
            "**At least one source file does not match its expected digest.** The "
            "figures may not be reproducible. Re-check the files in `Data/`.",
            kind="danger",
        )

    # -- pipeline ----------------------------------------------------------
    theme.section("The pipeline", "Five stages, each independently tested.")
    st.markdown(
        "1. **Load** - read the Excel sheet, assemble the date and time columns into "
        "a single hourly timestamp, and check the result is unique and continuous.\n"
        "2. **Clean** - replace the `-200` sentinel with a true missing value, drop "
        "exact duplicates, and confirm no hours are lost. This produces the CSV in "
        "`output/`.\n"
        "3. **Assess quality** - measure missingness per column, group columns that "
        "go missing together, and extract every unbroken outage block.\n"
        "4. **Analyse** - distributions, diurnal and seasonal profiles, correlations, "
        "sensor-versus-reference calibration, pollution episodes and sensor drift.\n"
        "5. **Model** - the benzene regression, with target leakage excluded and a "
        "chronological train/test split."
    )

    # -- column dictionary -------------------------------------------------
    with st.expander("The variables: every column, its unit and its role"):
        st.caption(
            "Every column in the cleaned dataset, what it measures and how it is used."
        )
        dictionary = pd.DataFrame(
            [
                {
                    "Column": column,
                    "Meaning": meta.get("label", column),
                    "Unit": meta.get("unit", ""),
                    "Role": meta.get("role", ""),
                    "Notes": meta.get("description", ""),
                }
                for column, meta in COLUMN_METADATA.items()
            ]
        )
        st.dataframe(dictionary, width="stretch", hide_index=True)

    # -- missing data ------------------------------------------------------
    theme.section(
        "How missing data is handled",
        f"The raw sheet marks a missing reading with `{SENTINEL_VALUE}`. Treating "
        "that as a real value would corrupt every average, so it is converted to a "
        "true missing value at load time.",
    )
    st.markdown(
        "- **Averages and medians** are computed over the readings that exist. No "
        "value is invented to fill a gap before a statistic is calculated.\n"
        "- **Charts** draw a line break where data is missing rather than joining "
        "across the gap.\n"
        "- **Correlations** use pairwise-complete observations, and the number of "
        "shared hours behind each pair is shown, so a correlation built on few "
        "points cannot masquerade as a strong one.\n"
        f"- **The benzene model** imputes missing predictors with the median of the "
        "training data and adds a flag for every imputed value, all *inside* the "
        "fitted pipeline so the imputation is learned from training rows only. Rows "
        f"without a `{PRIMARY_TARGET}` reading are dropped, because a missing target "
        "cannot be learned from or scored against."
    )

    # -- drift method ------------------------------------------------------
    theme.section(
        "How sensor drift is tested",
        "Drift is the slow loss of a sensor's response over months. It is easy to "
        "claim and hard to prove, so the dashboard holds every candidate to four "
        "checks at once.",
    )
    st.markdown(
        "1. **A per-month trend.** The sensor-to-reference relationship is re-fitted "
        "month by month and the slope is regressed against time. A trustworthy "
        "change shows a consistent direction.\n"
        "2. **Matched same-calendar-month comparison.** Early and late periods are "
        "compared only when they fall in the same month of the year, so seasonal "
        "effects cannot be mistaken for drift.\n"
        "3. **A confidence interval from resampling.** Paired hours are bootstrapped "
        "so the uncertainty of the slope is estimated from the data rather than "
        "assumed.\n"
        "4. **A sensitivity interpretation.** A change is only called a *loss* when "
        "the sensor's response to the reference gas actually weakened."
    )
    st.markdown(
        "Only a variable that clears all four checks is reported as *evidence "
        "consistent with a temporal change*. Everything else is reported as *no "
        "clear drift signal*. The dashboard never claims general sensor drift; it "
        "reports what this one record shows."
    )

    # -- model method ------------------------------------------------------
    theme.section("How the benzene model is evaluated")
    st.markdown(
        "- **Target:** `C6H6(GT)`, the reference benzene concentration.\n"
        "- **Excluded:** the target itself; `NMHC(GT)`, which the brief excludes and "
        "which is about 90% missing; and `PT08.S2(NMHC)`, which correlates at "
        "r = 0.982 with the target and would let the model read the answer from a "
        "second instrument.\n"
        "- **Split:** chronological. The earlier 80% of rows are used for training and "
        "the later 20% for testing, so the score reflects performance on a future "
        "period rather than on interpolated neighbours.\n"
        "- **Cross-validation:** five expanding-window folds on the training data "
        "only, never touching the test period.\n"
        "- **Models:** a mean baseline, an ordinary linear regression and a Random "
        "Forest with 300 trees. All three see identical rows.\n"
        "- **Scoring:** R2, MAE and RMSE on the held-out test period, with RMSE "
        "computed from the squared errors directly.\n"
        "- **Leakage check:** the same model is re-fitted with the proxy added back "
        "in, and the gain is reported so the cost of excluding it is measurable."
    )

    theme.section(
        "Sensor-only and reference-assisted models",
        "Two different questions, two different predictor sets, and two different "
        "labels - the distinction is the point, not a detail.",
    )
    st.markdown(
        "The dashboard reports a benzene model twice, and the two results answer "
        "different questions. Conflating them would overstate what cheap sensors "
        "can do."
    )
    sensor_only_text = _feature_set_text(PRIMARY_FEATURE_SET)
    reference_assisted_text = _feature_set_text(REFERENCE_ASSISTED_FEATURE_SET)
    st.markdown(
        f"- **Sensor-only (headline, `{PRIMARY_FEATURE_SET}`):** {sensor_only_text} "
        "Every predictor is a cheap weather measurement or a metal-oxide sensor "
        "response, so this is the result for the low-cost monitoring scenario. It is "
        "the number quoted as the headline.\n"
        f"- **Reference-assisted (comparison, `{REFERENCE_ASSISTED_FEATURE_SET}`):** "
        f"{reference_assisted_text} It also reads `CO(GT)`, `NOx(GT)` and `NO2(GT)` "
        "from the co-located certified reference analysers, so it describes a "
        "station that already owns the expensive instruments. It is reported as a "
        "labelled comparison and is never described as a low-cost sensor model."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature set": name,
                    "Label": FEATURE_SET_LABELS.get(name, name),
                    "What it may use": FEATURE_SET_DESCRIPTIONS.get(name, name),
                }
                for name in ("weather", PRIMARY_FEATURE_SET, REFERENCE_ASSISTED_FEATURE_SET)
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    theme.notice(
        "**Why the reference-assisted score is not the headline.** A model that may "
        "read the certified analysers is not estimating benzene from cheap hardware; "
        "it is being handed the analysers' own answers for CO, NOx and NO2 and "
        "reconstructing benzene from them. That is a legitimate upper bound for a "
        "well-equipped site, and it is a different claim from 'benzene can be "
        "estimated from low-cost sensors'. EcoPulse reports the sensor-only model as "
        "the headline and the reference-assisted model beside it, clearly labelled, "
        "so neither number can be mistaken for the other.",
        kind="info",
    )

    # -- reproducibility ---------------------------------------------------
    theme.section("Reproducing this")
    st.markdown(
        "From the project root, on Windows:"
    )
    st.code(
        "python -m pip install -e .\n"
        "python -m ecopulse.cleaning       # rebuild output/airquality_clean.csv\n"
        "python -m streamlit run app/main.py\n"
        "python -m pytest -q               # the full test suite",
        language="powershell",
    )
    with st.expander("Environment details"):
        versions = pd.DataFrame(
            [
                {"Component": "Python", "Version": platform.python_version()},
                {"Component": "Platform", "Version": platform.platform()},
                {"Component": "pandas", "Version": _version("pandas")},
                {"Component": "numpy", "Version": _version("numpy")},
                {"Component": "scikit-learn", "Version": _version("scikit-learn")},
                {"Component": "Streamlit", "Version": _version("streamlit")},
                {"Component": "Plotly", "Version": _version("plotly")},
            ]
        )
        st.dataframe(versions, width="stretch", hide_index=True)
        st.caption(
            "Figures can shift in the last digit across library versions. The models "
            "use a fixed random seed, so repeated runs on the same versions reproduce "
            "the same scores."
        )

    # -- limitations -------------------------------------------------------
    theme.section("What this analysis cannot tell you")
    theme.notice(
        "**One site, fourteen months.** The record describes a single monitoring "
        "location in one city. Nothing here generalises to another street, city or "
        "year without re-running the analysis on that data.\n\n"
        "**Historical data, not a live service.** These are measurements from 2004 "
        "and 2005. The dashboard does not read current air quality, does not produce "
        "an air-quality index, and must not be used for health or safety decisions.\n\n"
        "**Sensors, not certified instruments.** The metal-oxide sensors drift and "
        "respond to more than one gas. They are useful for pattern and relative "
        "change, not for regulatory compliance.\n\n"
        "**A retrospective estimate, not a forecast.** The benzene model uses "
        "same-hour readings from other instruments. It is not predicting the future "
        "and its error would be larger if it tried.\n\n"
        "**Missingness is not random.** The device failed in long blocks, and those "
        "blocked hours may differ systematically from the hours that were recorded. "
        "Treat any statistic as describing the observed hours, not the whole period.",
        kind="warn",
    )

    # -- credits -----------------------------------------------------------
    theme.section("About this project")
    st.markdown(
        "**EcoPulse** is a data-analysis and visualisation project built on the UCI "
        "Air Quality dataset. Every statistic on every page is computed from the "
        "cleaned data when the page loads; the project contains no hard-coded "
        "results, and the test suite re-checks the key claims against the source "
        "files."
    )
    credits = pd.DataFrame(
        [
            {"Name": "Mohd Anas", "Roll number": "2026010138"},
            {"Name": "Sumayah Lateef", "Roll number": "2026010125"},
        ]
    )
    st.dataframe(credits, width="stretch", hide_index=True)


def _version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # pragma: no cover - defensive
        return "(not installed)"
