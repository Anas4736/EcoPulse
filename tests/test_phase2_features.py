"""Phase 2 tests: feature engineering.

The promises checked here are the ones the modelling page depends on:

* a lag feature really is the value from the past, looked up by **calendar
  time** so that a gap in the timeline cannot silently shift it,
* a rolling window really covers ``window`` hours of calendar time and does
  not quietly shrink when hours are missing,
* the shift argument keeps a window from containing the value being
  predicted, and
* missing values stay missing instead of being silently filled in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecopulse.config import DATETIME_COLUMN, MEASURED_COLUMNS
from ecopulse.features import (
    DEFAULT_LAGS,
    add_lag_features,
    add_missing_indicators,
    add_rolling_features,
    add_time_features,
    build_feature_frame,
    indicator_name,
    lag_name,
    rolling_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_frame(values, start="2004-01-01", freq="h", column="NO2(GT)"):
    """A tiny frame with a datetime column and one measured column."""
    stamps = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame(
        {
            "datetime": stamps,
            column: np.asarray(values, dtype=float),
        }
    )


@pytest.fixture
def simple_frame():
    """72 hours of a clean, strictly increasing signal."""
    return make_frame(np.arange(100.0, 172.0))


@pytest.fixture
def frame_with_gap():
    """A 72-hour signal with hours 30, 31 and 32 removed from the timeline."""
    stamps = pd.date_range("2004-01-01", periods=72, freq="h")
    keep = [i for i in range(72) if i not in (30, 31, 32)]
    return pd.DataFrame(
        {
            "datetime": stamps[keep],
            "NO2(GT)": np.arange(72, dtype=float)[keep],
        }
    )


@pytest.fixture
def frame_with_missing():
    """A 48-hour signal where four readings are missing."""
    values = np.arange(100.0, 148.0)
    values[[5, 6, 20, 40]] = np.nan
    return make_frame(values)


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------
class TestColumnNaming:
    def test_indicator_name_is_identifier_safe(self):
        assert indicator_name("CO(GT)") == "missing_CO_GT"
        assert indicator_name("PT08.S1(CO)") == "missing_PT08_S1_CO"

    def test_indicator_names_are_unique(self):
        names = [indicator_name(column) for column in MEASURED_COLUMNS]
        assert len(names) == len(set(names))

    def test_lag_and_rolling_names(self):
        assert lag_name("CO(GT)", 24) == "CO_GT_lag24"
        assert rolling_name("CO(GT)", 24, "mean") == "CO_GT_roll24_mean"
        assert rolling_name("CO(GT)", 24, "mean", shift=1) == "CO_GT_roll24_mean_shift1"


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------
class TestTimeFeatures:
    def test_adds_calendar_columns(self, simple_frame):
        result = add_time_features(simple_frame)
        for column in [
            "year", "month", "month_name", "day", "hour", "dayofweek",
            "day_name", "quarter", "dayofyear", "is_weekend", "season",
            "date", "is_daytime",
        ]:
            assert column in result.columns

    def test_row_count_and_original_columns_preserved(self, simple_frame):
        result = add_time_features(simple_frame)
        assert len(result) == len(simple_frame)
        assert "NO2(GT)" in result.columns

    def test_does_not_modify_the_input(self, simple_frame):
        before = simple_frame.copy()
        add_time_features(simple_frame)
        pd.testing.assert_frame_equal(simple_frame, before)

    def test_hour_and_dayofweek_are_correct(self):
        # 2004-01-05 is a Monday (dayofweek == 0).
        frame = make_frame([1.0], start="2004-01-05 13:00")
        result = add_time_features(frame)
        assert result.loc[0, "hour"] == 13
        assert result.loc[0, "dayofweek"] == 0
        assert result.loc[0, "is_weekend"] is np.False_ or not result.loc[0, "is_weekend"]

    def test_weekend_is_detected(self):
        frame = make_frame([1.0], start="2004-01-10")  # Saturday
        result = add_time_features(frame)
        assert bool(result.loc[0, "is_weekend"])

    def test_seasons_follow_the_northern_hemisphere(self):
        stamps = pd.to_datetime(
            ["2004-01-15", "2004-02-15", "2004-03-15", "2004-04-15"]
        )
        frame = pd.DataFrame({"datetime": stamps, "NO2(GT)": [1.0] * 4})
        result = add_time_features(frame)
        # January, February, March, April -> Winter, Winter, Spring, Spring
        assert list(result["season"]) == ["Winter", "Winter", "Spring", "Spring"]

    def test_is_daytime_window(self):
        stamps = pd.to_datetime(
            ["2004-01-01 05:00", "2004-01-01 06:00", "2004-01-01 21:00", "2004-01-01 22:00"]
        )
        frame = pd.DataFrame({"datetime": stamps, "NO2(GT)": [1.0] * 4})
        result = add_time_features(frame)
        assert list(result["is_daytime"]) == [False, True, True, False]

    def test_cyclical_hour_encoding_round_trips(self, simple_frame):
        result = add_time_features(simple_frame)
        angle = 2 * np.pi * result["hour"] / 24.0
        assert np.allclose(result["hour_sin"], np.sin(angle))
        assert np.allclose(result["hour_cos"], np.cos(angle))

    def test_cyclical_encoding_makes_midnight_and_23_close(self):
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2004-01-01 23:00", "2004-01-02 00:00"]),
                "NO2(GT)": [1.0, 1.0],
            }
        )
        result = add_time_features(frame)
        distance = np.hypot(
            result.loc[0, "hour_sin"] - result.loc[1, "hour_sin"],
            result.loc[0, "hour_cos"] - result.loc[1, "hour_cos"],
        )
        assert distance < 0.3  # one hour apart on a unit circle

    def test_cyclical_can_be_switched_off(self, simple_frame):
        result = add_time_features(simple_frame, include_cyclical=False)
        assert "hour_sin" not in result.columns
        assert "doy_sin" not in result.columns

    def test_missing_datetime_column_raises(self, simple_frame):
        with pytest.raises(ValueError, match="datetime"):
            add_time_features(simple_frame.drop(columns=[DATETIME_COLUMN]))


# ---------------------------------------------------------------------------
# Missing-value indicators
# ---------------------------------------------------------------------------
class TestMissingIndicators:
    def test_flags_match_the_underlying_missing_values(self, frame_with_missing):
        result = add_missing_indicators(frame_with_missing, columns=["NO2(GT)"])
        expected = frame_with_missing["NO2(GT)"].isna().astype("int8")
        assert np.array_equal(result[indicator_name("NO2(GT)")].to_numpy(), expected.to_numpy())

    def test_missing_count_is_correct(self, frame_with_missing):
        result = add_missing_indicators(frame_with_missing, columns=["NO2(GT)"])
        expected = [1 if index in (5, 6, 20, 40) else 0 for index in range(48)]
        assert list(result["missing_count"]) == expected

    def test_is_complete_is_the_inverse_of_missing_count(self, frame_with_missing):
        result = add_missing_indicators(frame_with_missing, columns=["NO2(GT)"])
        assert np.array_equal(
            result["is_complete"].to_numpy(),
            (result["missing_count"] == 0).astype("int8").to_numpy(),
        )

    def test_all_thirteen_measured_columns_get_a_flag(self, simple_frame):
        for column in MEASURED_COLUMNS:
            simple_frame[column] = 1.0
        result = add_missing_indicators(simple_frame)
        for column in MEASURED_COLUMNS:
            assert indicator_name(column) in result.columns

    def test_only_the_requested_columns_are_flagged(self, frame_with_missing):
        result = add_missing_indicators(frame_with_missing, columns=["NO2(GT)"])
        assert indicator_name("NO2(GT)") in result.columns
        assert indicator_name("CO(GT)") not in result.columns

    def test_an_absent_column_raises_instead_of_being_skipped(self, frame_with_missing):
        """A typo must not quietly produce a frame without that indicator."""
        with pytest.raises(ValueError, match="CO\\(GT\\)"):
            add_missing_indicators(frame_with_missing, columns=["NO2(GT)", "CO(GT)"])


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------
class TestLagFeatures:
    def test_lag_one_is_the_previous_hour(self, simple_frame):
        result = add_lag_features(simple_frame, ["NO2(GT)"], lags=(1,))
        # Row 0 has no previous hour, so it must be missing.
        assert np.isnan(result.loc[0, lag_name("NO2(GT)", 1)])
        assert result.loc[1, lag_name("NO2(GT)", 1)] == 100.0
        assert result.loc[5, lag_name("NO2(GT)", 1)] == 104.0

    def test_lag_24_is_the_same_hour_previous_day(self, simple_frame):
        result = add_lag_features(simple_frame, ["NO2(GT)"], lags=(24,))
        assert np.isnan(result.loc[0, lag_name("NO2(GT)", 24)])
        assert result.loc[24, lag_name("NO2(GT)", 24)] == 100.0
        assert result.loc[30, lag_name("NO2(GT)", 24)] == 106.0

    def test_lag_168_is_the_same_hour_previous_week(self, simple_frame):
        result = add_lag_features(simple_frame, ["NO2(GT)"], lags=(168,))
        # The fixture only holds 72 hours, so every weekly lag is missing.
        assert result[lag_name("NO2(GT)", 168)].isna().all()

    def test_lag_never_contains_the_current_value(self, simple_frame):
        """A lag must look strictly backwards."""
        result = add_lag_features(simple_frame, ["NO2(GT)"], lags=DEFAULT_LAGS)
        for lag in DEFAULT_LAGS:
            name = lag_name("NO2(GT)", lag)
            valid = result[name].notna()
            assert (result.loc[valid, name] < result.loc[valid, "NO2(GT)"]).all()

    def test_gap_makes_the_lag_missing_rather_than_shifted(self, frame_with_gap):
        """A removed hour must not be silently replaced by the wrong hour.

        The gap covers 2004-01-02 06:00-08:00. The first row after the gap
        is 2004-01-02 09:00, whose one-hour lag is 08:00 - an hour that is
        not in the data. The calendar-time lookup therefore returns nothing.
        A naive ``shift(1)`` on the rows would have returned the value of
        05:00, i.e. the wrong hour, which is exactly the bug the timestamp
        lookup prevents.
        """
        result = add_lag_features(frame_with_gap, ["NO2(GT)"], lags=(1,))
        name = lag_name("NO2(GT)", 1)

        row = result[result["datetime"] == pd.Timestamp("2004-01-02 09:00")]
        assert np.isnan(row.loc[row.index[0], name])

        naive = frame_with_gap["NO2(GT)"].shift(1)
        assert naive.loc[row.index[0]] == 29.0  # the wrong hour

    def test_gap_does_not_shift_later_lags(self, frame_with_gap):
        """A value further back in time is still found by timestamp."""
        result = add_lag_features(frame_with_gap, ["NO2(GT)"], lags=(24,))
        name = lag_name("NO2(GT)", 24)
        # 2004-01-03 09:00 minus 24 hours is 2004-01-02 09:00, which is in
        # the data and holds the value 33.
        row = result[result["datetime"] == pd.Timestamp("2004-01-03 09:00")]
        assert row.loc[row.index[0], name] == 33.0

    def test_calendar_lag_differs_from_a_positional_shift(self, frame_with_gap):
        """The whole point of looking lags up by time, not by row."""
        result = add_lag_features(frame_with_gap, ["NO2(GT)"], lags=(24,))
        name = lag_name("NO2(GT)", 24)

        # Direct check of the calendar-time semantics for every row: the lag
        # must be the value recorded exactly 24 hours earlier.
        lookup = frame_with_gap.set_index("datetime")["NO2(GT)"]
        for index, row in frame_with_gap.iterrows():
            wanted = row["datetime"] - pd.Timedelta(hours=24)
            expected = lookup.reindex([wanted]).iloc[0]
            actual = result.loc[index, name]
            if pd.isna(expected):
                assert pd.isna(actual)
            else:
                assert actual == expected

        # And a positional shift is genuinely a different thing.
        naive = frame_with_gap["NO2(GT)"].shift(24)
        assert not np.allclose(result[name].fillna(-1.0), naive.fillna(-1.0))

    def test_missing_source_value_gives_a_missing_lag(self, frame_with_missing):
        result = add_lag_features(frame_with_missing, ["NO2(GT)"], lags=(1,))
        name = lag_name("NO2(GT)", 1)
        # Row 6's lag is row 5, which is missing.
        assert np.isnan(result.loc[6, name])
        # Row 7's lag is row 6, which is missing too.
        assert np.isnan(result.loc[7, name])
        # Row 5's lag is row 4, which is present.
        assert result.loc[5, name] == 104.0

    def test_original_column_is_untouched(self, simple_frame):
        result = add_lag_features(simple_frame, ["NO2(GT)"], lags=(1,))
        assert np.array_equal(
            result["NO2(GT)"].to_numpy(), simple_frame["NO2(GT)"].to_numpy()
        )


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------
class TestRollingFeatures:
    def test_rolling_mean_covers_the_window(self, simple_frame):
        result = add_rolling_features(simple_frame, ["NO2(GT)"], window=24)
        name = rolling_name("NO2(GT)", 24, "mean")
        # The first full window ends at row 23 and covers values 100..123.
        expected = np.arange(100.0, 124.0).mean()
        assert result.loc[23, name] == pytest.approx(expected)

    def test_rolling_mean_uses_only_past_values(self, simple_frame):
        result = add_rolling_features(simple_frame, ["NO2(GT)"], window=24)
        name = rolling_name("NO2(GT)", 24, "mean")
        for index in (24, 40, 71):
            expected = np.arange(100.0 + index - 23, 100.0 + index + 1).mean()
            assert result.loc[index, name] == pytest.approx(expected)

    def test_shift_keeps_the_window_out_of_the_target(self, simple_frame):
        """With shift=1 the window must not contain the row's own value."""
        result = add_rolling_features(simple_frame, ["NO2(GT)"], window=24, shift=1)
        name = rolling_name("NO2(GT)", 24, "mean", shift=1)
        # Without the shift the window at row 24 covers 101..124 and so
        # includes the row's own value of 124.0. The shifted window at row
        # 24 covers 100..123, so it must be lower.
        assert result.loc[24, name] == pytest.approx(np.arange(100.0, 124.0).mean())
        assert result.loc[24, name] < 124.0

    def test_window_does_not_shrink_across_a_gap(self, frame_with_gap):
        """A missing hour inside the window must not be silently dropped."""
        result = add_rolling_features(frame_with_gap, ["NO2(GT)"], window=24)
        name = rolling_name("NO2(GT)", 24, "mean")
        # The row at 2004-01-02 09:00 is the first one after the gap. Its
        # 24-hour window reaches back to 2004-01-01 10:00, and the three
        # removed hours (06:00-08:00 on the 2nd) fall inside that window,
        # so the mean is over 21 observations, not 24.
        row = result[result["datetime"] == pd.Timestamp("2004-01-02 09:00")]
        expected = (sum(range(10, 30)) + 33.0) / 21
        assert row.loc[row.index[0], name] == pytest.approx(expected)

        # The naive "average of the previous 24 rows" would have been
        # something else entirely, which is the bug the re-indexing prevents.
        naive = frame_with_gap["NO2(GT)"].iloc[:30].mean()
        assert not np.isclose(row.loc[row.index[0], name], naive)

    def test_rolling_window_spans_a_gap_in_calendar_time(self, frame_with_gap):
        """The window is 24 hours of *time*, so it covers 24 hours even when
        the frame has fewer than 24 rows left after a gap."""
        result = add_rolling_features(frame_with_gap, ["NO2(GT)"], window=24)
        name = rolling_name("NO2(GT)", 24, "mean")
        # 2004-01-03 09:00 -> the window starts at 2004-01-02 10:00, which
        # is past the gap, so all 24 hours are present (values 34..57).
        row = result[result["datetime"] == pd.Timestamp("2004-01-03 09:00")]
        assert row.loc[row.index[0], name] == pytest.approx(np.arange(34.0, 58.0).mean())

    def test_missing_values_are_ignored_not_filled(self, frame_with_missing):
        result = add_rolling_features(frame_with_missing, ["NO2(GT)"], window=6)
        name = rolling_name("NO2(GT)", 6, "mean")
        # Row 10's window covers rows 5..10, of which 5 and 6 are missing,
        # so the mean is over the four valid values 107, 108, 109, 110.
        assert result.loc[10, name] == pytest.approx(np.mean([107.0, 108.0, 109.0, 110.0]))

    def test_supported_statistics(self, simple_frame):
        result = add_rolling_features(
            simple_frame, ["NO2(GT)"], window=24,
            stats=("mean", "median", "std", "min", "max"),
        )
        for stat in ("mean", "median", "std", "min", "max"):
            assert rolling_name("NO2(GT)", 24, stat) in result.columns

    def test_unsupported_statistic_raises(self, simple_frame):
        with pytest.raises(ValueError, match="Unsupported rolling statistic"):
            add_rolling_features(simple_frame, ["NO2(GT)"], stats=("variance",))

    def test_rolling_mean_of_a_constant_series_is_that_constant(self):
        frame = make_frame([7.0] * 50)
        result = add_rolling_features(frame, ["NO2(GT)"], window=24)
        name = rolling_name("NO2(GT)", 24, "mean")
        assert result.loc[30, name] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# The full feature frame
# ---------------------------------------------------------------------------
class TestBuildFeatureFrame:
    def test_contains_every_feature_group(self, cleaned_result):
        result = build_feature_frame(cleaned_result.frame)
        assert "hour_sin" in result.columns
        assert "missing_count" in result.columns
        assert lag_name("CO(GT)", 24) in result.columns
        assert rolling_name("CO(GT)", 24, "mean") in result.columns

    def test_row_count_is_unchanged(self, simple_frame):
        result = build_feature_frame(simple_frame)
        assert len(result) == len(simple_frame)

    def test_defaults_skip_columns_the_frame_does_not_have(self, simple_frame):
        """A small frame still builds, using only the columns it has."""
        result = build_feature_frame(simple_frame)
        # NO2(GT) is one of the default columns and the frame has it.
        assert lag_name("NO2(GT)", 1) in result.columns
        # CO(GT) is a default column the frame does not have, so it is skipped
        # rather than raising.
        assert lag_name("CO(GT)", 1) not in result.columns
        assert "missing_count" in result.columns

    def test_works_on_the_real_dataset(self, cleaned_result):
        frame = cleaned_result.frame
        result = build_feature_frame(frame)
        assert len(result) == len(frame)
        assert result["missing_count"].max() <= len(MEASURED_COLUMNS)

    def test_does_not_modify_the_input(self, cleaned_result):
        frame = cleaned_result.frame
        before = frame.copy()
        build_feature_frame(frame)
        pd.testing.assert_frame_equal(frame, before)
