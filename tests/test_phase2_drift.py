"""Phase 2 tests: the sensor-drift investigation.

The drift question is the easiest place in this project to fool yourself, so
these tests do two different things:

* they check the machinery on **synthetic data with a known answer** - a
  sensor whose calibration is constant, and a sensor whose calibration
  changes between years - and
* they pin down the verdict for the **real** dataset, so a change in the
  analysis cannot quietly change the headline claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecopulse.drift import (
    DEFAULT_MIN_PAIRS,
    RANDOM_SEED,
    _bootstrap_trend_ci,
    _trend,
    assess_all_sensors,
    assess_drift,
    matched_period_comparison,
    monthly_calibration,
    year_over_year_calibration,
)

#: Every verdict the module is allowed to produce.
ALLOWED_VERDICTS = {
    "insufficient data",
    "no clear drift signal",
    "evidence consistent with a temporal change",
    "temporal trend detected, seasonal control unavailable",
}

#: The sensor/reference pairs used by the dashboard.
SENSOR_PAIRS = [
    ("PT08.S1(CO)", "CO(GT)"),
    ("PT08.S2(NMHC)", "C6H6(GT)"),
    ("PT08.S3(NOx)", "NOx(GT)"),
    ("PT08.S4(NO2)", "NO2(GT)"),
    ("PT08.S5(O3)", "NO2(GT)"),
]


# ---------------------------------------------------------------------------
# Synthetic data with a known answer
# ---------------------------------------------------------------------------
def make_synthetic(
    days: int = 730,
    drift: bool = False,
    second_half_change: bool = False,
    seed: int = 7,
) -> pd.DataFrame:
    """Two years of hourly data with a known sensor calibration.

    The reference concentration follows a smooth seasonal cycle plus noise.
    The sensor is a *noisy* measurement of the reference, so regressing the
    reference on the sensor recovers the calibration ``scale``:

        reference ~ scale * sensor

    With ``drift=True`` the scale changes from 2.0 in 2004 to 1.5 in 2005,
    which is a real temporal change. With ``drift=False`` the scale never
    changes, so any apparent trend would be a false positive.
    """
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2004-01-01", periods=days * 24, freq="h")
    years = stamps.year.to_numpy()
    day_of_year = stamps.dayofyear.to_numpy()

    season = 30.0 * np.sin(2 * np.pi * day_of_year / 365.25)
    reference = 100.0 + season + rng.normal(0, 3.0, len(stamps))

    if drift:
        scale = np.where(years == 2004, 2.0, 1.5)
    elif second_half_change:
        halfway = days * 24 // 2
        scale = np.where(np.arange(len(stamps)) < halfway, 2.0, 1.6)
    else:
        scale = np.full(len(stamps), 2.0)

    sensor = reference / scale + rng.normal(0, 0.3, len(stamps))

    return pd.DataFrame(
        {
            "datetime": stamps,
            "SENSOR": sensor,
            "REFERENCE": reference,
        }
    )


# ---------------------------------------------------------------------------
# Monthly calibration
# ---------------------------------------------------------------------------
class TestMonthlyCalibration:
    def test_one_row_per_month(self):
        frame = make_synthetic()
        table = monthly_calibration(frame, "SENSOR", "REFERENCE")
        assert len(table) == 24
        assert list(table["month"]) == sorted(table["month"])

    def test_columns_and_dtypes(self):
        frame = make_synthetic()
        table = monthly_calibration(frame, "SENSOR", "REFERENCE")
        for column in [
            "month", "year", "month_number", "n_pairs", "slope", "intercept",
            "pearson_r", "r_squared", "rmse", "mean_sensor", "mean_reference",
        ]:
            assert column in table.columns
        assert table["n_pairs"].dtype.kind == "i"

    def test_recovers_the_true_calibration(self):
        """The fitted slope comes back close to the true value of 2.0.

        It comes back slightly *under* it, because regressing a clean
        quantity on a noisy measurement always understates the slope (the
        classical regression-dilution effect). What matters for drift
        detection is that the estimate is consistent from month to month.
        """
        frame = make_synthetic()
        table = monthly_calibration(frame, "SENSOR", "REFERENCE")
        assert table["slope"].mean() < 2.0  # attenuated by the sensor noise
        assert table["slope"].mean() == pytest.approx(2.0, abs=0.1)
        assert table["slope"].std() < 0.05  # and stable across the months

    def test_min_pairs_is_respected(self):
        frame = make_synthetic()
        strict = monthly_calibration(frame, "SENSOR", "REFERENCE", min_pairs=1000)
        assert len(strict) == 0
        assert len(monthly_calibration(frame, "SENSOR", "REFERENCE", min_pairs=30)) == 24

    def test_missing_values_are_excluded_from_the_fit(self):
        frame = make_synthetic(days=60)
        frame.loc[0:50, "REFERENCE"] = np.nan
        table = monthly_calibration(frame, "SENSOR", "REFERENCE", min_pairs=30)
        # January holds 744 hours, of which the first 51 have no reference.
        assert table.iloc[0]["n_pairs"] == 744 - 51
        assert table.iloc[0]["n_pairs"] < 744

    def test_a_column_with_no_usable_data_gives_an_empty_table(self):
        frame = make_synthetic(days=60)
        frame["REFERENCE"] = np.nan
        table = monthly_calibration(frame, "SENSOR", "REFERENCE")
        assert len(table) == 0
        assert "slope" in table.columns


# ---------------------------------------------------------------------------
# Matched period comparison (the seasonal control)
# ---------------------------------------------------------------------------
class TestMatchedPeriodComparison:
    def test_one_row_per_calendar_month_shared_by_two_years(self):
        frame = make_synthetic()
        table = matched_period_comparison(frame, "SENSOR", "REFERENCE")
        assert len(table) == 12  # every calendar month appears in both years
        assert sorted(table["month_number"]) == list(range(1, 13))

    def test_both_years_are_trimmed_to_the_same_days(self):
        """The comparison must be fair in coverage, not just in season."""
        frame = make_synthetic(days=400)  # ends mid-February of the second year
        table = matched_period_comparison(frame, "SENSOR", "REFERENCE")
        assert len(table) == 2  # only January and February are in both years
        for _, row in table.iterrows():
            # The later year is cut short, so the shared day is small.
            assert row["shared_day"] <= 31
            assert row["shared_day"] == row["shared_day"]  # sanity

    def test_the_real_dataset_compares_march_and_april(self, cleaned_result):
        table = matched_period_comparison(
            cleaned_result.frame, "PT08.S1(CO)", "CO(GT)"
        )
        assert sorted(table["month_number"]) == [3, 4]
        # April 2005 only covers the first four days, so the shared day is 4.
        april = table[table["month_number"] == 4].iloc[0]
        assert april["shared_day"] == 4

    def test_a_year_on_its_own_gives_no_comparison(self):
        frame = make_synthetic(days=365)
        table = matched_period_comparison(frame, "SENSOR", "REFERENCE")
        assert len(table) == 0

    def test_the_slope_change_is_reported(self):
        frame = make_synthetic(drift=True)
        table = matched_period_comparison(frame, "SENSOR", "REFERENCE")
        # The calibration really did change from 2.0 to 1.5.
        assert table["slope_change"].mean() == pytest.approx(-0.5, abs=0.05)


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------
class TestBootstrapTrendCI:
    def test_a_perfect_trend_is_recovered(self):
        values = np.arange(14, dtype=float)
        low, high = _bootstrap_trend_ci(values)
        assert low <= 1.0 <= high
        assert (high - low) < 0.05

    def test_the_interval_contains_the_point_estimate(self):
        """A percentile interval that does not contain the estimate is a bug.

        This is the invariant the earlier, value-only resampling broke: it
        shuffled the series into a random order, whose trend is zero by
        construction, so the interval was centred on zero instead of on the
        estimate.
        """
        values = np.array(
            [-0.26, -0.25, -0.25, -0.29, -0.41, -0.24, -0.61, -0.95,
             -0.67, -0.38, -0.74, -0.77, -0.68, -0.56]
        )
        estimate = _trend(values)
        low, high = _bootstrap_trend_ci(values)
        assert low <= estimate <= high

    def test_a_flat_series_includes_zero(self):
        rng = np.random.default_rng(3)
        values = rng.normal(0, 1, 14)
        low, high = _bootstrap_trend_ci(values)
        assert low < 0 < high

    def test_too_few_points_gives_nan(self):
        low, high = _bootstrap_trend_ci(np.array([1.0, 2.0]))
        assert np.isnan(low) and np.isnan(high)

    def test_the_result_is_reproducible(self):
        values = np.arange(14, dtype=float) + np.array([0.1, -0.2, 0.3] * 4 + [0.1, 0.1])
        assert _bootstrap_trend_ci(values) == _bootstrap_trend_ci(values)

    def test_different_seeds_agree_closely(self):
        values = np.arange(14, dtype=float)
        first = _bootstrap_trend_ci(values, seed=1)
        second = _bootstrap_trend_ci(values, seed=2)
        assert first[0] == pytest.approx(second[0], abs=0.05)
        assert first[1] == pytest.approx(second[1], abs=0.05)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
class TestAssessDrift:
    def test_no_drift_is_reported_as_no_signal(self):
        """A constant calibration with a strong seasonal cycle must not be
        mistaken for drift - that is exactly the mistake a naive
        first-versus-last comparison would make."""
        frame = make_synthetic(drift=False)
        assessment = assess_drift(frame, "SENSOR", "REFERENCE")
        assert assessment.verdict == "no clear drift signal"
        assert assessment.trend_ci_excludes_zero is False

    def test_a_real_drift_is_detected(self):
        frame = make_synthetic(drift=True)
        assessment = assess_drift(frame, "SENSOR", "REFERENCE")
        assert assessment.verdict == "evidence consistent with a temporal change"
        assert assessment.trend_ci_excludes_zero is True
        assert assessment.slope_trend_per_month < 0

    def test_the_seasonal_control_supports_the_drift(self):
        frame = make_synthetic(drift=True)
        assessment = assess_drift(frame, "SENSOR", "REFERENCE")
        assert assessment.matched_months == 12
        assert assessment.matched_agrees_with_trend == 12

    def test_a_drift_without_a_seasonal_control_says_so(self):
        """One year of data cannot separate a trend from the season."""
        frame = make_synthetic(days=365, second_half_change=True)
        assessment = assess_drift(frame, "SENSOR", "REFERENCE")
        assert assessment.verdict == "temporal trend detected, seasonal control unavailable"
        assert assessment.matched_months == 0

    def test_too_little_data_is_reported_as_insufficient(self):
        frame = make_synthetic(days=90)
        assessment = assess_drift(frame, "SENSOR", "REFERENCE")
        assert assessment.verdict == "insufficient data"
        assert "not enough" in assessment.verdict_reasons[0]

    def test_the_verdict_is_always_one_of_the_allowed_strings(self):
        frame = make_synthetic(drift=True)
        for sensor, reference in [("SENSOR", "REFERENCE")]:
            assessment = assess_drift(frame, sensor, reference)
            assert assessment.verdict in ALLOWED_VERDICTS

    def test_the_verdict_is_reproducible(self):
        frame = make_synthetic(drift=True)
        first = assess_drift(frame, "SENSOR", "REFERENCE")
        second = assess_drift(frame, "SENSOR", "REFERENCE")
        assert first.verdict == second.verdict
        assert first.slope_trend_ci == second.slope_trend_ci

    def test_coverage_and_missingness_notes_are_present(self):
        frame = make_synthetic(drift=True)
        assessment = assess_drift(frame, "SENSOR", "REFERENCE")
        assert "Record spans" in assessment.coverage_note
        assert "%" in assessment.missingness_note

    def test_the_reasons_explain_the_verdict(self):
        frame = make_synthetic(drift=True)
        assessment = assess_drift(frame, "SENSOR", "REFERENCE")
        assert len(assessment.verdict_reasons) >= 3
        assert any("bootstrap interval" in reason for reason in assessment.verdict_reasons)
        assert any("matched same-season" in reason for reason in assessment.verdict_reasons)


# ---------------------------------------------------------------------------
# The real dataset
# ---------------------------------------------------------------------------
class TestRealDatasetDrift:
    def test_every_sensor_pair_produces_a_valid_verdict(self, cleaned_result):
        assessments = assess_all_sensors(cleaned_result.frame, SENSOR_PAIRS)
        assert len(assessments) == 5
        for assessment in assessments:
            assert assessment.verdict in ALLOWED_VERDICTS
            assert assessment.months_analysed == 14

    def test_the_verdicts_for_the_real_data(self, cleaned_result):
        """These are the findings the Data Quality page reports."""
        expected = {
            "PT08.S1(CO)": "no clear drift signal",
            "PT08.S2(NMHC)": "no clear drift signal",
            "PT08.S3(NOx)": "evidence consistent with a temporal change",
            "PT08.S4(NO2)": "no clear drift signal",
            "PT08.S5(O3)": "no clear drift signal",
        }
        assessments = assess_all_sensors(cleaned_result.frame, SENSOR_PAIRS)
        for assessment in assessments:
            assert assessment.verdict == expected[assessment.sensor], assessment.sensor

    def test_only_the_nox_sensor_shows_a_temporal_change(self, cleaned_result):
        """One sensor drifting out of five is the finding, so it is checked
        that no other sensor is quietly claiming the same thing."""
        assessments = assess_all_sensors(cleaned_result.frame, SENSOR_PAIRS)
        flagged = [a.sensor for a in assessments if a.trend_ci_excludes_zero]
        assert flagged == ["PT08.S3(NOx)"]

    def test_the_nox_sensor_drift_is_a_sensitivity_loss(self, cleaned_result):
        """The NOx sensor's response weakens: the slope becomes more
        negative, so the same sensor reading means a larger concentration."""
        assessment = assess_drift(cleaned_result.frame, "PT08.S3(NOx)", "NOx(GT)")
        assert assessment.slope_trend_per_month < 0
        assert assessment.monthly["slope"].iloc[0] > assessment.monthly["slope"].iloc[-1]

    def test_the_seasonal_control_is_applied_to_the_real_data(self, cleaned_result):
        assessments = assess_all_sensors(cleaned_result.frame, SENSOR_PAIRS)
        for assessment in assessments:
            assert assessment.matched_months == 2  # March and April

    def test_the_confidence_intervals_are_ordered_and_finite(self, cleaned_result):
        assessments = assess_all_sensors(cleaned_result.frame, SENSOR_PAIRS)
        for assessment in assessments:
            low, high = assessment.slope_trend_ci
            assert np.isfinite(low) and np.isfinite(high)
            assert low <= high


# ---------------------------------------------------------------------------
# The simple year-over-year view
# ---------------------------------------------------------------------------
class TestYearOverYearCalibration:
    def test_empty_input_gives_an_empty_frame(self):
        table = year_over_year_calibration(monthly_calibration(
            make_synthetic(days=60), "SENSOR", "REFERENCE", min_pairs=1000
        ))
        assert len(table) == 0
        assert "slope_change" in table.columns

    def test_it_reports_the_two_shared_months(self, cleaned_result):
        monthly = monthly_calibration(cleaned_result.frame, "PT08.S1(CO)", "CO(GT)")
        table = year_over_year_calibration(monthly)
        assert sorted(table["month_number"]) == [3, 4]
        assert table.loc[0, "years"] == "2004 vs 2005"
