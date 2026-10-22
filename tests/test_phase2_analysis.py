"""Phase 2 tests: the analysis layer.

These tests check the promise that every number in the dashboard is computed
at runtime from the data, with the sample size reported next to it, and that
missing values and gaps in the timeline are handled honestly rather than
being quietly ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ecopulse.analysis import (
    absolute_humidity_check,
    correlation_matrix,
    daily_profile,
    describe_variables,
    episode_summary,
    find_episodes,
    hourly_profile,
    missingness_by_period,
    monthly_aggregate,
    pairwise_counts,
    pollutant_distribution,
    seasonal_aggregate,
    sensor_reference_summary,
)
from ecopulse.config import DATETIME_COLUMN, DEFAULT_NO2_THRESHOLD_UG_M3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_frame(values, start="2004-01-01", freq="h"):
    """A tiny frame with a datetime column and a NO2(GT) column."""
    stamps = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame(
        {
            "datetime": stamps,
            "NO2(GT)": np.asarray(values, dtype=float),
        }
    )


@pytest.fixture
def clean_frame():
    """Five days of a clean hourly NO2 signal."""
    return make_frame(np.arange(50.0, 170.0))


@pytest.fixture
def frame_with_missing():
    """48 hours with four missing readings."""
    values = np.arange(100.0, 148.0)
    values[[3, 4, 25, 44]] = np.nan
    return make_frame(values)


@pytest.fixture
def frame_with_gap():
    """72 hours of a signal with three hours removed from the timeline."""
    stamps = pd.date_range("2004-01-01", periods=72, freq="h")
    keep = [i for i in range(72) if i not in (30, 31, 32)]
    return pd.DataFrame(
        {
            "datetime": stamps[keep],
            "NO2(GT)": np.arange(72, dtype=float)[keep],
        }
    )


# ---------------------------------------------------------------------------
# describe_variables
# ---------------------------------------------------------------------------
class TestDescribeVariables:
    def test_counts_match_the_data(self, frame_with_missing):
        table = describe_variables(frame_with_missing, ["NO2(GT)"])
        row = table.iloc[0]
        assert row["Valid"] == 44
        assert row["Missing"] == 4
        assert row["Missing %"] == pytest.approx(100 * 4 / 48, abs=0.01)

    def test_statistics_use_valid_values_only(self, frame_with_missing):
        table = describe_variables(frame_with_missing, ["NO2(GT)"])
        expected = frame_with_missing["NO2(GT)"].dropna()
        row = table.iloc[0]
        assert row["Mean"] == pytest.approx(expected.mean())
        assert row["Median"] == pytest.approx(expected.median())
        assert row["Min"] == pytest.approx(expected.min())
        assert row["Max"] == pytest.approx(expected.max())
        assert row["Q1"] == pytest.approx(expected.quantile(0.25))
        assert row["IQR"] == pytest.approx(expected.quantile(0.75) - expected.quantile(0.25))

    def test_every_measured_column_is_described(self, cleaned_result):
        from ecopulse.config import MEASURED_COLUMNS

        table = describe_variables(cleaned_result.frame)
        assert list(table["Column"]) == MEASURED_COLUMNS
        assert table["Valid"].sum() > 0

    def test_a_column_with_no_valid_values_does_not_crash(self):
        frame = make_frame([np.nan, np.nan, np.nan])
        table = describe_variables(frame, ["NO2(GT)"])
        row = table.iloc[0]
        assert row["Valid"] == 0
        assert row["Missing"] == 3
        assert np.isnan(row["Mean"])

    def test_nmhc_missingness_is_reported_honestly(self, cleaned_result):
        table = describe_variables(cleaned_result.frame, ["NMHC(GT)"])
        row = table.iloc[0]
        assert row["Missing %"] > 85
        assert row["Valid"] < 1000


# ---------------------------------------------------------------------------
# pollutant_distribution
# ---------------------------------------------------------------------------
class TestPollutantDistribution:
    def test_histogram_counts_cover_every_valid_value(self, clean_frame):
        result = pollutant_distribution(clean_frame, "NO2(GT)", bins=10)
        assert int(result["counts"].sum()) == result["n_valid"] == 120

    def test_edges_bracket_the_data(self, clean_frame):
        result = pollutant_distribution(clean_frame, "NO2(GT)", bins=10)
        assert result["edges"][0] <= 50.0
        assert result["edges"][-1] >= 169.0

    def test_summary_statistics_match_the_data(self, clean_frame):
        result = pollutant_distribution(clean_frame, "NO2(GT)")
        assert result["stats"]["mean"] == pytest.approx(clean_frame["NO2(GT)"].mean())
        assert result["stats"]["min"] == pytest.approx(50.0)
        assert result["stats"]["max"] == pytest.approx(169.0)

    def test_missing_values_are_excluded(self, frame_with_missing):
        result = pollutant_distribution(frame_with_missing, "NO2(GT)")
        assert result["n_valid"] == 44
        assert result["stats"]["min"] == pytest.approx(100.0)

    def test_an_all_missing_column_gives_an_empty_result(self):
        frame = make_frame([np.nan, np.nan])
        result = pollutant_distribution(frame, "NO2(GT)")
        assert result["n_valid"] == 0
        assert result["counts"].size == 0


# ---------------------------------------------------------------------------
# Patterns in time
# ---------------------------------------------------------------------------
class TestHourlyProfile:
    def test_one_row_per_hour_of_day(self, clean_frame):
        profile = hourly_profile(clean_frame, "NO2(GT)")
        assert len(profile) == 24
        assert list(profile["hour"]) == list(range(24))

    def test_observation_counts_add_up(self, frame_with_missing):
        profile = hourly_profile(frame_with_missing, "NO2(GT)")
        assert int(profile["n"].sum()) == 44

    def test_means_are_correct(self, clean_frame):
        profile = hourly_profile(clean_frame, "NO2(GT)")
        # The fixture covers five days, so each hour of the day appears five
        # times with values 24 hours apart.
        assert profile.loc[profile["hour"] == 0, "mean"].iloc[0] == pytest.approx(
            np.mean([50.0, 74.0, 98.0, 122.0, 146.0])
        )
        assert profile.loc[profile["hour"] == 5, "mean"].iloc[0] == pytest.approx(
            np.mean([55.0, 79.0, 103.0, 127.0, 151.0])
        )

    def test_seasonal_split_has_24_rows_per_season(self, cleaned_result):
        profile = hourly_profile(cleaned_result.frame, "NO2(GT)", by_season=True)
        assert set(profile["season"]) == {"Winter", "Spring", "Summer", "Autumn"}
        assert len(profile) == 96

    def test_the_real_data_has_a_real_diurnal_signal(self, cleaned_result):
        """The evening peak must be visible - this is a headline finding."""
        profile = hourly_profile(cleaned_result.frame, "NO2(GT)")
        quietest = profile.loc[profile["mean"].idxmin()]
        busiest = profile.loc[profile["mean"].idxmax()]
        assert busiest["mean"] > quietest["mean"] * 1.5
        assert 17 <= busiest["hour"] <= 21


class TestDailyProfile:
    def test_one_row_per_day(self, clean_frame):
        profile = daily_profile(clean_frame, "NO2(GT)")
        assert len(profile) == 5

    def test_daily_statistics_are_correct(self, clean_frame):
        profile = daily_profile(clean_frame, "NO2(GT)")
        first = profile.iloc[0]
        assert first["daily_mean"] == pytest.approx(np.mean([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]))
        assert first["n"] == 24
        assert first["daily_min"] == pytest.approx(50.0)
        assert first["daily_max"] == pytest.approx(73.0)

    def test_a_gap_in_the_timeline_shortens_that_day(self, frame_with_gap):
        profile = daily_profile(frame_with_gap, "NO2(GT)")
        # The gap removes three hours from 2004-01-02.
        second = profile[profile["date"] == pd.Timestamp("2004-01-02")].iloc[0]
        assert second["n"] == 21

    def test_missing_values_reduce_the_count(self, frame_with_missing):
        profile = daily_profile(frame_with_missing, "NO2(GT)")
        assert int(profile["n"].sum()) == 44


class TestMonthlyAggregate:
    def test_one_row_per_month(self, cleaned_result):
        table = monthly_aggregate(cleaned_result.frame, "NO2(GT)")
        assert len(table) == 14  # March 2004 to April 2005

    def test_aggregation_name_is_in_the_column(self, clean_frame):
        table = monthly_aggregate(clean_frame, "NO2(GT)", "mean")
        assert "value_mean" in table.columns
        assert "n" in table.columns

    def test_count_is_reported_next_to_the_statistic(self, clean_frame):
        table = monthly_aggregate(clean_frame, "NO2(GT)", "mean")
        assert int(table.loc[0, "n"]) == 120

    def test_max_and_min_aggregations_work(self, clean_frame):
        maximum = monthly_aggregate(clean_frame, "NO2(GT)", "max")
        minimum = monthly_aggregate(clean_frame, "NO2(GT)", "min")
        assert maximum.loc[0, "value_max"] == pytest.approx(169.0)
        assert minimum.loc[0, "value_min"] == pytest.approx(50.0)

    def test_median_aggregation_works(self, clean_frame):
        table = monthly_aggregate(clean_frame, "NO2(GT)", "median")
        assert table.loc[0, "value_median"] == pytest.approx(np.median(np.arange(50.0, 170.0)))

    def test_missing_values_are_excluded_from_the_statistic(self, frame_with_missing):
        table = monthly_aggregate(frame_with_missing, "NO2(GT)", "mean")
        assert int(table.loc[0, "n"]) == 44

    def test_a_gap_reduces_the_month_count(self, frame_with_gap):
        table = monthly_aggregate(frame_with_gap, "NO2(GT)", "mean")
        assert int(table.loc[0, "n"]) == 69  # 72 hours minus the three removed


class TestSeasonalAggregate:
    def test_all_four_seasons_are_present_in_order(self, cleaned_result):
        table = seasonal_aggregate(cleaned_result.frame, "NO2(GT)")
        assert list(table["season"]) == ["Winter", "Spring", "Summer", "Autumn"]

    def test_winter_is_the_most_polluted_season(self, cleaned_result):
        """A headline finding that must be recomputed, not hard-coded."""
        table = seasonal_aggregate(cleaned_result.frame, "NO2(GT)").set_index("season")
        assert table.loc["Winter", "mean"] > table.loc["Summer", "mean"]

    def test_counts_add_up_to_the_valid_observations(self, cleaned_result):
        table = seasonal_aggregate(cleaned_result.frame, "NO2(GT)")
        assert int(table["n"].sum()) == int(cleaned_result.frame["NO2(GT)"].notna().sum())


class TestMissingnessByPeriod:
    def test_one_row_per_month_with_percentages(self, cleaned_result):
        table = missingness_by_period(cleaned_result.frame, ["NO2(GT)"], period="month")
        assert len(table) == 14
        assert "missing_pct_NO2(GT)" in table.columns
        assert "rows" in table.columns

    def test_percentages_are_between_zero_and_one_hundred(self, cleaned_result):
        table = missingness_by_period(cleaned_result.frame, ["CO(GT)", "NO2(GT)"])
        for column in ("missing_pct_CO(GT)", "missing_pct_NO2(GT)"):
            assert table[column].between(0, 100).all()

    def test_a_fully_present_column_reports_zero_percent(self, clean_frame):
        table = missingness_by_period(clean_frame, ["NO2(GT)"])
        assert table.loc[0, "missing_pct_NO2(GT)"] == 0.0

    def test_missing_rows_are_counted_per_period(self, frame_with_missing):
        table = missingness_by_period(frame_with_missing, ["NO2(GT)"], period="month")
        assert int(table.loc[0, "rows"]) == 48
        assert table.loc[0, "missing_pct_NO2(GT)"] == pytest.approx(100 * 4 / 48, abs=0.01)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------
class TestCorrelationMatrix:
    def test_diagonal_is_one(self, cleaned_result):
        matrix = correlation_matrix(cleaned_result.frame, ["CO(GT)", "C6H6(GT)", "NO2(GT)"])
        assert np.allclose(np.diag(matrix.to_numpy()), 1.0)

    def test_matrix_is_symmetric(self, cleaned_result):
        matrix = correlation_matrix(cleaned_result.frame, ["CO(GT)", "C6H6(GT)", "NO2(GT)"])
        assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)

    def test_matches_scipy(self, frame_with_missing):
        frame = frame_with_missing.copy()
        frame["CO(GT)"] = frame["NO2(GT)"] * 2 + 1
        matrix = correlation_matrix(frame, ["NO2(GT)", "CO(GT)"])
        paired = frame[["NO2(GT)", "CO(GT)"]].dropna()
        expected = stats.pearsonr(paired["NO2(GT)"], paired["CO(GT)"]).statistic
        assert matrix.loc["NO2(GT)", "CO(GT)"] == pytest.approx(expected)

    def test_missing_values_are_dropped_pairwise(self, frame_with_missing):
        frame = frame_with_missing.copy()
        frame["CO(GT)"] = frame["NO2(GT)"] * 2 + 1
        matrix = correlation_matrix(frame, ["NO2(GT)", "CO(GT)"])
        assert matrix.loc["NO2(GT)", "CO(GT)"] == pytest.approx(1.0)

    def test_spearman_can_be_requested(self, cleaned_result):
        matrix = correlation_matrix(cleaned_result.frame, ["CO(GT)", "C6H6(GT)"], method="spearman")
        assert matrix.loc["CO(GT)", "C6H6(GT)"] == pytest.approx(0.93, abs=0.05)


class TestPairwiseCounts:
    def test_counts_are_symmetric(self, cleaned_result):
        counts = pairwise_counts(cleaned_result.frame, ["CO(GT)", "C6H6(GT)", "NO2(GT)"])
        assert counts.loc["CO(GT)", "C6H6(GT)"] == counts.loc["C6H6(GT)", "CO(GT)"]

    def test_counts_match_a_direct_computation(self, frame_with_missing):
        frame = frame_with_missing.copy()
        frame["CO(GT)"] = frame["NO2(GT)"]
        counts = pairwise_counts(frame, ["NO2(GT)", "CO(GT)"])
        assert counts.loc["NO2(GT)", "CO(GT)"] == 44

    def test_diagonal_is_the_valid_count(self, cleaned_result):
        counts = pairwise_counts(cleaned_result.frame, ["NO2(GT)"])
        assert counts.loc["NO2(GT)", "NO2(GT)"] == int(cleaned_result.frame["NO2(GT)"].notna().sum())


class TestSensorReferenceSummary:
    def test_pair_count_is_correct(self, cleaned_result):
        frame = cleaned_result.frame
        summary = sensor_reference_summary(frame, "PT08.S2(NMHC)", "C6H6(GT)")
        expected = int(
            (frame["PT08.S2(NMHC)"].notna() & frame["C6H6(GT)"].notna()).sum()
        )
        assert summary["n_pairs"] == expected

    def test_correlation_matches_scipy(self, cleaned_result):
        frame = cleaned_result.frame
        summary = sensor_reference_summary(frame, "PT08.S1(CO)", "CO(GT)")
        paired = frame[["PT08.S1(CO)", "CO(GT)"]].dropna()
        expected = stats.pearsonr(paired["PT08.S1(CO)"], paired["CO(GT)"]).statistic
        assert summary["pearson_r"] == pytest.approx(expected)

    def test_fitted_line_recovers_the_data(self, cleaned_result):
        frame = cleaned_result.frame
        summary = sensor_reference_summary(frame, "PT08.S1(CO)", "CO(GT)")
        paired = frame[["PT08.S1(CO)", "CO(GT)"]].dropna()
        predicted = summary["slope"] * paired["PT08.S1(CO)"] + summary["intercept"]
        assert summary["rmse"] == pytest.approx(
            float(np.sqrt(np.mean((predicted - paired["CO(GT)"]) ** 2)))
        )

    def test_bias_is_reported(self, cleaned_result):
        summary = sensor_reference_summary(cleaned_result.frame, "PT08.S1(CO)", "CO(GT)")
        assert isinstance(summary["bias"], float)
        assert abs(summary["bias"]) < 1e-6  # a least-squares fit has no bias

    def test_too_few_points_gives_nan_rather_than_a_crash(self):
        frame = make_frame([1.0, 2.0, 3.0])
        frame["SENSOR"] = [1.0, 2.0, np.nan]
        summary = sensor_reference_summary(frame, "SENSOR", "NO2(GT)")
        assert summary["n_pairs"] == 2
        assert np.isnan(summary["pearson_r"])

    def test_the_benzene_sensor_tracks_benzene_very_closely(self, cleaned_result):
        """A headline finding: the S2 sensor ranks benzene almost perfectly."""
        summary = sensor_reference_summary(cleaned_result.frame, "PT08.S2(NMHC)", "C6H6(GT)")
        assert summary["pearson_r"] > 0.95
        assert summary["spearman_rho"] > 0.99


class TestAbsoluteHumidityCheck:
    def test_ah_is_a_scaled_function_of_temperature_and_humidity(self, cleaned_result):
        """A verified finding: AH carries almost no extra information."""
        result = absolute_humidity_check(cleaned_result.frame)
        assert result["n_pairs"] == 8991
        assert result["pearson_r"] > 0.999
        assert result["scale_cv_pct"] < 2.0

    def test_the_scale_factor_is_stable(self, cleaned_result):
        result = absolute_humidity_check(cleaned_result.frame)
        assert result["scale_factor"] == pytest.approx(0.099, abs=0.005)

    def test_ah_tracks_temperature_more_than_raw_humidity(self, cleaned_result):
        """Saturation vapour pressure grows exponentially with temperature,
        so AH follows T more closely than it follows RH on its own."""
        result = absolute_humidity_check(cleaned_result.frame)
        assert abs(result["correlation_with_temperature"]) > abs(
            result["correlation_with_humidity"]
        )

    def test_missing_rows_are_excluded(self):
        frame = make_frame([10.0, 20.0, 30.0])
        frame["T"] = [15.0, 16.0, np.nan]
        frame["RH"] = [50.0, 60.0, 70.0]
        frame["AH"] = [1.0, 1.2, 1.4]
        result = absolute_humidity_check(frame)
        assert result["n_pairs"] == 2


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
class TestFindEpisodes:
    def test_a_contiguous_run_is_one_episode(self):
        frame = make_frame([10.0, 250.0, 260.0, 270.0, 10.0])
        episodes = find_episodes(frame, "NO2(GT)", 200.0)
        assert len(episodes) == 1
        assert episodes.loc[0, "duration_hours"] == 3
        assert episodes.loc[0, "peak_value"] == 270.0

    def test_separate_runs_are_separate_episodes(self):
        frame = make_frame([250.0, 10.0, 10.0, 260.0])
        episodes = find_episodes(frame, "NO2(GT)", 200.0)
        assert len(episodes) == 2
        assert list(episodes["duration_hours"]) == [1, 1]

    def test_a_missing_value_ends_the_run(self):
        """An outage means we cannot claim the pollutant stayed high."""
        frame = make_frame([250.0, 260.0, np.nan, 270.0])
        episodes = find_episodes(frame, "NO2(GT)", 200.0)
        assert len(episodes) == 2
        assert list(episodes["duration_hours"]) == [2, 1]

    def test_min_duration_filters_short_runs(self):
        frame = make_frame([250.0, 10.0, 10.0, 260.0, 270.0])
        assert len(find_episodes(frame, "NO2(GT)", 200.0, min_duration=2)) == 1
        assert len(find_episodes(frame, "NO2(GT)", 200.0, min_duration=3)) == 0

    def test_the_threshold_is_respected_exactly(self):
        frame = make_frame([200.0, 200.01, 199.99])
        episodes = find_episodes(frame, "NO2(GT)", 200.0)
        assert len(episodes) == 1
        assert episodes.loc[0, "duration_hours"] == 1

    def test_no_episodes_gives_an_empty_frame_with_columns(self):
        frame = make_frame([1.0, 2.0, 3.0])
        episodes = find_episodes(frame, "NO2(GT)", 200.0)
        assert len(episodes) == 0
        assert "peak_value" in episodes.columns

    def test_start_and_end_timestamps_are_correct(self):
        frame = make_frame([10.0, 250.0, 260.0, 10.0])
        episodes = find_episodes(frame, "NO2(GT)", 200.0)
        assert episodes.loc[0, "start"] == pd.Timestamp("2004-01-01 01:00")
        assert episodes.loc[0, "end"] == pd.Timestamp("2004-01-01 02:00")

    def test_a_gap_in_the_timeline_ends_the_run(self):
        """Three missing hours break the episode even though the values are
        above the threshold on both sides."""
        stamps = pd.to_datetime(["2004-01-01 00:00", "2004-01-01 01:00",
                                  "2004-01-01 05:00", "2004-01-01 06:00"])
        frame = pd.DataFrame({"datetime": stamps, "NO2(GT)": [250.0, 260.0, 270.0, 280.0]})
        episodes = find_episodes(frame, "NO2(GT)", 200.0)
        assert len(episodes) == 2


class TestEpisodeSummary:
    def test_exceedance_hours_are_counted(self, cleaned_result):
        summary = episode_summary(cleaned_result.frame, "NO2(GT)", DEFAULT_NO2_THRESHOLD_UG_M3)
        expected = int(
            (cleaned_result.frame["NO2(GT)"] > DEFAULT_NO2_THRESHOLD_UG_M3).sum()
        )
        assert summary["n_exceedance_hours"] == expected
        assert summary["n_exceedance_hours"] > 0

    def test_percentage_is_of_the_valid_observations(self, cleaned_result):
        summary = episode_summary(cleaned_result.frame, "NO2(GT)", DEFAULT_NO2_THRESHOLD_UG_M3)
        assert summary["exceedance_pct_of_valid"] == pytest.approx(
            100 * summary["n_exceedance_hours"] / summary["n_valid"]
        )

    def test_episode_count_matches_the_episode_table(self, cleaned_result):
        summary = episode_summary(cleaned_result.frame, "NO2(GT)", DEFAULT_NO2_THRESHOLD_UG_M3)
        assert summary["n_episodes"] == len(summary["episodes"])

    def test_yearly_split_is_reported(self, cleaned_result):
        summary = episode_summary(cleaned_result.frame, "NO2(GT)", DEFAULT_NO2_THRESHOLD_UG_M3)
        total = sum(summary["exceedance_hours_by_year"].values())
        assert total == summary["n_exceedance_hours"]
        assert set(summary["exceedance_hours_by_year"]) == {"2004", "2005"}

    def test_peak_value_is_the_column_maximum(self, cleaned_result):
        summary = episode_summary(cleaned_result.frame, "NO2(GT)", DEFAULT_NO2_THRESHOLD_UG_M3)
        assert summary["peak_value"] == pytest.approx(
            cleaned_result.frame["NO2(GT)"].max()
        )

    def test_a_threshold_nothing_exceeds_gives_zero_episodes(self, cleaned_result):
        summary = episode_summary(cleaned_result.frame, "NO2(GT)", 10000.0)
        assert summary["n_episodes"] == 0
        assert summary["n_exceedance_hours"] == 0
        assert summary["max_episode_hours"] == 0
