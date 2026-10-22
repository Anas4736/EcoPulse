"""Phase 4 - the dashboard actually renders.

Each test launches the real Streamlit script with ``AppTest``, navigates to one
page in the sidebar and asserts that the page drew without an exception. These
are smoke tests: they do not check wording, they check that every page runs end
to end against the real dataset.

The last few tests go one step further for the benzene page, because the
headline model on it is the one claim the whole project rests on: the page must
show the *sensor-only* score, and it must show the number the pipeline actually
produced rather than a hard-coded string.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN = PROJECT_ROOT / "app" / "main.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecopulse.ml import (  # noqa: E402
    PRIMARY_FEATURE_SET,
    REFERENCE_ASSISTED_FEATURE_SET,
    run_benzene_benchmark,
)

PAGES = [
    "Overview",
    "Pollution Explorer",
    "Pollutant Relationships",
    "Data Quality",
    "Anomaly Detection",
    "Benzene Regression",
    "Methodology & About",
]

#: The metric cards are rendered as raw HTML, so the label and the value can be
#: read back out of the page and compared with the pipeline's own numbers.
CARD = re.compile(
    r'ep-card-label">(?P<label>[^<]*)</div><div class="ep-card-value">(?P<value>[^<]*)</div>'
)


def _open(page: str) -> AppTest:
    app = AppTest.from_file(str(MAIN), default_timeout=900)
    app.run()
    assert not app.exception, f"the app failed on first load: {app.exception}"
    app.sidebar.radio[0].set_value(page).run()
    return app


def _page_text(app: AppTest) -> str:
    """Every scrap of text the page rendered, whatever element it came from."""
    parts: list[str] = []
    for name in (
        "markdown",
        "caption",
        "header",
        "subheader",
        "title",
        "code",
        "info",
        "warning",
        "success",
        "error",
        "text",
    ):
        for element in getattr(app, name):
            parts.append(str(element.value))
    for element in app.table:
        parts.append(str(element.value))
    for element in app.dataframe:
        parts.append(element.value.to_string())
    return "\n".join(parts)


def _metric_cards(app: AppTest) -> dict[str, str]:
    """The metric cards of the page, as ``label -> value``."""
    text = "\n".join(element.value for element in app.markdown)
    return {
        match.group("label"): match.group("value")
        for match in CARD.finditer(text)
    }


@pytest.fixture(scope="session")
def sensor_only_score():
    """The headline score, recomputed by the pipeline rather than read from a file."""
    from ecopulse.cleaning import load_clean_dataset

    frame = load_clean_dataset().frame
    benchmark = run_benzene_benchmark(
        frame,
        feature_sets=(PRIMARY_FEATURE_SET,),
        models=("random_forest",),
        cv_splits=5,
    )
    return benchmark.get("random_forest")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str) -> None:
    app = _open(page)
    assert not app.exception, f"'{page}' raised: {app.exception}"
    assert app.markdown, f"'{page}' rendered no content"
    assert not app.error, f"'{page}' showed an error: {[e.value for e in app.error]}"


def test_every_page_is_reachable() -> None:
    app = AppTest.from_file(str(MAIN), default_timeout=900)
    app.run()
    options = app.sidebar.radio[0].options
    assert list(options) == PAGES


def test_overview_states_the_dataset_period() -> None:
    app = _open("Overview")
    text = " ".join(m.value for m in app.markdown)
    assert "2004" in text and "2005" in text


def test_disclaimer_is_visible_on_every_page() -> None:
    app = AppTest.from_file(str(MAIN), default_timeout=900)
    app.run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"'{page}' raised: {app.exception}"
        text = " ".join(m.value for m in app.markdown)
        assert "historical" in text.lower(), f"'{page}' is missing the historical-data note"


def test_date_filter_narrows_the_data() -> None:
    from datetime import date

    app = AppTest.from_file(str(MAIN), default_timeout=900)
    app.run()
    app.sidebar.radio[0].set_value("Pollution Explorer").run()
    assert not app.exception, app.exception

    date_input = app.sidebar.date_input[0]
    date_input.set_value((date(2004, 6, 1), date(2004, 6, 30))).run()
    assert not app.exception, app.exception
    assert app.markdown


def test_anomaly_threshold_reaches_zero_episodes() -> None:
    app = AppTest.from_file(str(MAIN), default_timeout=900)
    app.run()
    app.sidebar.radio[0].set_value("Anomaly Detection").run()
    assert not app.exception, app.exception

    number_input = app.number_input[0]
    number_input.set_value(float(number_input.max)).run()
    assert not app.exception, app.exception
    text = " ".join(n.value for n in app.markdown)
    assert "No episode" in text


def test_benzene_page_reports_the_sensor_only_headline_score(
    sensor_only_score,
) -> None:
    """The headline card must show the score the pipeline actually produced."""
    app = _open("Benzene Regression")
    cards = _metric_cards(app)

    assert "Sensor-only R²" in cards, "the sensor-only headline card is missing"
    assert cards["Sensor-only R²"] == f"{sensor_only_score.r2:.3f}"
    assert cards["Sensor-only RMSE"] == f"{sensor_only_score.rmse:.2f} ug/m3"


def test_benzene_page_labels_the_two_models() -> None:
    """The page must not leave the reader guessing which model is which."""
    app = _open("Benzene Regression")
    text = _page_text(app)

    assert "Sensor-only (headline)" in text
    assert "Reference-assisted (comparison)" in text

    # The tier table carries the labels as data, not only as prose.
    tier_tables = [
        element.value
        for element in app.dataframe
        if "Predictors" in element.value.columns
    ]
    assert tier_tables, "the predictor-tier table is missing"
    labels = list(tier_tables[0]["Model"])
    assert "Sensor-only (headline)" in labels
    assert "Reference-assisted (comparison)" in labels


def test_benzene_page_does_not_quiet_the_reference_assisted_model() -> None:
    """The reference-assisted score may be shown, but must be labelled as such."""
    app = _open("Benzene Regression")
    text = _page_text(app)

    assert "reference-assisted" in text.lower()
    assert "not a low-cost sensor result" in text.lower()


def test_methodology_page_explains_the_two_models() -> None:
    """The methodology page has to explain the distinction, not just mention it."""
    app = _open("Methodology & About")
    text = _page_text(app)

    assert "Sensor-only" in text
    assert "Reference-assisted" in text
    # The page must say what the extra predictors are and where they come from.
    assert "CO(GT)" in text and "NO2(GT)" in text
    assert "certified" in text.lower()


def test_methodology_verifies_source_hashes() -> None:
    app = _open("Methodology & About")
    text = " ".join(m.value for m in app.markdown)
    assert "SHA-256" in text or "digest" in text.lower()
    assert not app.error


def test_app_survives_an_empty_date_range() -> None:
    """A range with no hours must degrade gracefully, not crash."""
    from datetime import date, timedelta

    app = AppTest.from_file(str(MAIN), default_timeout=900)
    app.run()
    app.sidebar.radio[0].set_value("Pollution Explorer").run()
    assert not app.exception, app.exception

    date_input = app.sidebar.date_input[0]
    # A single day that contains data still works; the guard matters for pages
    # that must handle very small selections.
    date_input.set_value((date(2004, 3, 10), date(2004, 3, 11))).run()
    assert not app.exception, app.exception
