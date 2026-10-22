"""Phase 1 tests: loading, cleaning, the quality report and the CSV export.

These tests check the promises made to the user:

* the original workbook is opened read-only and never modified,
* ``-200`` becomes a real missing value **only** in the measured columns,
* the date and time columns are parsed into a correct hourly timeline,
* no rows are silently lost, and
* the cleaned CSV that ships with the project matches the cleaned frame.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ecopulse.cleaning import clean_frame, export_clean_csv, load_clean_dataset
from ecopulse.config import (
    CLEANED_CSV,
    DATETIME_COLUMN,
    DATE_COLUMN,
    MEASURED_COLUMNS,
    SENTINEL_VALUE,
    TIME_COLUMN,
)
from ecopulse.loader import DataLoadError, load_raw
from ecopulse.quality import (
    before_after_table,
    group_columns_by_missing_pattern,
    missing_pattern_overlaps,
    summary_metrics,
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class TestLoader:
    def test_loads_expected_shape(self, loaded_frame):
        assert len(loaded_frame) == 9357
        assert len(loaded_frame.columns) == 16  # Date, Time, 13 measured, datetime

    def test_all_measured_columns_present(self, loaded_frame):
        for column in MEASURED_COLUMNS:
            assert column in loaded_frame.columns

    def test_datetime_column_is_parsed(self, loaded_frame):
        assert pd.api.types.is_datetime64_any_dtype(loaded_frame[DATETIME_COLUMN])
        assert loaded_frame[DATETIME_COLUMN].notna().all()

    def test_datetime_first_and_last_values(self, loaded_frame):
        times = loaded_frame[DATETIME_COLUMN]
        assert times.iloc[0] == pd.Timestamp("2004-03-10 18:00:00")
        assert times.iloc[-1] == pd.Timestamp("2005-04-04 14:00:00")

    def test_rows_are_sorted_by_time(self, loaded_frame):
        times = loaded_frame[DATETIME_COLUMN]
        assert times.is_monotonic_increasing

    def test_hourly_spacing_with_no_gaps(self, loaded_frame):
        diffs = loaded_frame[DATETIME_COLUMN].diff().dt.total_seconds() / 3600
        assert (diffs.dropna() == 1.0).all()

    def test_sentinels_survive_loading(self, loaded_frame):
        """The loader must NOT clean anything - that is the cleaner's job."""
        total = int((loaded_frame[MEASURED_COLUMNS] == SENTINEL_VALUE).sum().sum())
        assert total > 0

    def test_missing_file_raises_clear_error(self, tmp_path):
        with pytest.raises(DataLoadError):
            load_raw(tmp_path / "does_not_exist.xlsx")

    def test_time_parsing_accepts_text_format(self):
        """The UCI CSV sibling uses 'HH.MM.SS' strings - both must work."""
        from ecopulse.loader import _parse_times

        parsed = _parse_times(pd.Series(["00.00.00", "18.00.00", "23.59.59"]))
        assert list(parsed) == [0.0, 64800.0, 86399.0]

    def test_time_parsing_accepts_time_objects(self):
        from ecopulse.loader import _parse_times

        parsed = _parse_times(pd.Series([dt.time(6, 30), dt.time(0, 0)]))
        assert list(parsed) == [23400.0, 0.0]

    def test_date_parsing_handles_day_first_strings(self):
        from ecopulse.loader import _parse_dates

        parsed = _parse_dates(pd.Series(["10/03/2004", "01/04/2004"]))
        assert parsed.iloc[0] == pd.Timestamp("2004-03-10")
        assert parsed.iloc[1] == pd.Timestamp("2004-04-01")

    def test_loader_does_not_modify_the_source_file(self, source_path):
        """Re-reading must not change a single byte of the original file."""
        import hashlib

        def digest() -> str:
            return hashlib.sha256(source_path.read_bytes()).hexdigest().upper()

        before = digest()
        load_raw()
        assert digest() == before


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
class TestCleaning:
    def test_all_rows_retained(self, cleaned_result):
        assert cleaned_result.report.rows_read == 9357
        assert cleaned_result.report.rows_after_clean == 9357
        assert cleaned_result.report.rows_dropped == 0

    def test_no_duplicate_rows_or_timestamps(self, cleaned_result):
        frame = cleaned_result.frame
        assert not frame.duplicated().any()
        assert not frame.duplicated(subset=[DATETIME_COLUMN]).any()

    def test_sentinel_removed_from_every_measured_column(self, cleaned_result):
        frame = cleaned_result.frame
        for column in MEASURED_COLUMNS:
            remaining = int((frame[column] == SENTINEL_VALUE).sum())
            assert remaining == 0, f"{column} still contains {remaining} sentinels"

    def test_date_and_time_columns_untouched(self, cleaned_result):
        """The sentinel must never be applied to the timestamp columns."""
        frame = cleaned_result.frame
        assert frame[DATE_COLUMN].notna().all()
        assert frame[TIME_COLUMN].notna().all()
        assert not (frame[DATE_COLUMN] == SENTINEL_VALUE).any()

    def test_all_measured_columns_are_float(self, cleaned_result):
        frame = cleaned_result.frame
        for column in MEASURED_COLUMNS:
            assert frame[column].dtype == np.float64, f"{column} is not float64"

    def test_nmhc_kept_but_mostly_missing(self, cleaned_result):
        """NMHC(GT) must never be silently removed from the dataset."""
        frame = cleaned_result.frame
        assert "NMHC(GT)" in frame.columns
        valid = int(frame["NMHC(GT)"].notna().sum())
        assert valid == 914
        assert 100 * (1 - valid / len(frame)) > 85

    def test_expected_valid_counts_per_column(self, cleaned_result):
        """Counts computed during the dataset inspection, used as a regression guard."""
        expected = {
            "CO(GT)": 7674,
            "PT08.S1(CO)": 8991,
            "NMHC(GT)": 914,
            "C6H6(GT)": 8991,
            "PT08.S2(NMHC)": 8991,
            "NOx(GT)": 7718,
            "PT08.S3(NOx)": 8991,
            "NO2(GT)": 7715,
            "PT08.S4(NO2)": 8991,
            "PT08.S5(O3)": 8991,
            "T": 8991,
            "RH": 8991,
            "AH": 8991,
        }
        frame = cleaned_result.frame
        for column, count in expected.items():
            assert int(frame[column].notna().sum()) == count, column

    def test_missing_counts_match_sentinel_counts(self, cleaned_result):
        """Each -200 marker becomes exactly one missing value."""
        frame = cleaned_result.frame
        for entry in cleaned_result.report.columns:
            assert entry.missing_after_clean == entry.sentinel_count

    def test_no_non_physical_values_remain(self, cleaned_result):
        for entry in cleaned_result.report.columns:
            assert entry.non_physical_count == 0, f"{entry.column} holds implausible values"

    def test_timeline_is_continuous(self, cleaned_result):
        report = cleaned_result.report
        assert report.gap_count == 0
        assert report.missing_hours == 0
        assert report.expected_hours == 9357

    def test_cleaning_does_not_mutate_its_input(self, loaded_frame):
        before = loaded_frame["CO(GT)"].copy()
        clean_frame(loaded_frame)
        assert loaded_frame["CO(GT)"].equals(before)

    def test_cleaning_is_idempotent(self, loaded_frame):
        """Cleaning an already-clean frame must not change it any further."""
        once = clean_frame(loaded_frame).frame
        twice = clean_frame(once).frame
        pd.testing.assert_frame_equal(once, twice)

    def test_recleaning_reports_no_new_sentinels(self, loaded_frame):
        """A second cleaning pass finds no -200 markers left, and says so."""
        once = clean_frame(loaded_frame)
        twice = clean_frame(once.frame)
        assert twice.report.rows_after_clean == once.report.rows_after_clean
        assert sum(e.sentinel_count for e in twice.report.columns) == 0
        assert twice.report.missing_pattern_groups == []

    def test_valid_mask_works(self, cleaned_result):
        mask = cleaned_result.valid_mask
        assert mask.sum() == 827  # rows where all 13 variables are present
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------
class TestQualityReport:
    def test_report_row_counts(self, cleaned_result):
        report = cleaned_result.report
        assert report.rows_read == 9357
        assert report.rows_after_clean == 9357
        assert report.rows_retained_pct == 100.0
        assert report.duplicate_rows_removed == 0
        assert report.duplicate_timestamps_removed == 0

    def test_missing_mechanisms_are_identified(self, cleaned_result):
        """Columns that fail together must be grouped together.

        The grouping works on the RAW frame, because that is where the -200
        markers still exist.
        """
        groups = group_columns_by_missing_pattern(load_raw())

        device_group = next(group for group in groups if "T" in group)
        assert len(device_group) == 9
        assert "PT08.S1(CO)" in device_group and "AH" in device_group

        analyzer_group = next(
            group for group in groups if "NOx(GT)" in group and "NO2(GT)" in group
        )
        assert analyzer_group == ["NOx(GT)", "NO2(GT)"]

        assert len(groups) == 4

    def test_grouping_rejects_an_already_cleaned_frame(self, cleaned_result):
        """Calling the grouping with a cleaned frame is a mistake - it must fail loudly."""
        with pytest.raises(ValueError, match="RAW frame"):
            group_columns_by_missing_pattern(cleaned_result.frame)

    def test_outage_blocks_are_found(self, cleaned_result):
        report = cleaned_result.report
        assert report.outages, "expected at least one outage block"
        assert sum(block.hours for block in report.outages) > 0
        for block in report.outages:
            assert block.end >= block.start

    def test_before_after_table_shape(self, cleaned_result):
        table = before_after_table(cleaned_result.report)
        assert len(table) == 13
        assert "Raw -200 markers" in table.columns
        assert "Valid values" in table.columns
        assert table["Valid values"].sum() > 0

    def test_summary_metrics(self, cleaned_result):
        metrics = summary_metrics(cleaned_result.report)
        assert metrics["rows_read"] == 9357
        assert metrics["rows_after_clean"] == 9357
        assert metrics["gap_count"] == 0
        assert metrics["non_physical_total"] == 0
        assert metrics["total_sentinel_markers"] > 0

    def test_overlap_table_reports_nox_no2_similarity(self, cleaned_result):
        """NOx and NO2 share the same analyzer channel - the report must show it."""
        raw = load_raw()
        overlaps = missing_pattern_overlaps(raw)
        pair = overlaps[
            (overlaps["Column A"] == "NO2(GT)") & (overlaps["Column B"] == "NOx(GT)")
        ]
        assert not pair.empty
        assert float(pair.iloc[0]["Jaccard similarity"]) > 0.99


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
class TestCsvExport:
    def test_export_creates_the_file(self, tmp_path, cleaned_result):
        target = tmp_path / "airquality_clean.csv"
        written = export_clean_csv(cleaned_result.frame, target)
        assert written.exists()
        assert written.stat().st_size > 0

    def test_exported_csv_round_trips(self, tmp_path, cleaned_result):
        target = tmp_path / "airquality_clean.csv"
        export_clean_csv(cleaned_result.frame, target)
        reread = pd.read_csv(target)

        assert len(reread) == len(cleaned_result.frame)
        expected_columns = [DATE_COLUMN, TIME_COLUMN, DATETIME_COLUMN, *MEASURED_COLUMNS]
        assert list(reread.columns) == expected_columns

    def test_exported_csv_date_column_is_date_only(self, tmp_path, cleaned_result):
        """The Date column must not carry a misleading midnight timestamp."""
        target = tmp_path / "airquality_clean.csv"
        export_clean_csv(cleaned_result.frame, target)
        reread = pd.read_csv(target, dtype={DATE_COLUMN: str})
        assert reread[DATE_COLUMN].iloc[0] == "2004-03-10"
        assert reread[DATE_COLUMN].iloc[-1] == "2005-04-04"
        assert reread[DATE_COLUMN].str.match(r"^\d{4}-\d{2}-\d{2}$").all()

    def test_exported_csv_keeps_meaningful_precision(self, tmp_path, cleaned_result):
        """Rounding must not distort the sensor values."""
        target = tmp_path / "airquality_clean.csv"
        export_clean_csv(cleaned_result.frame, target)
        reread = pd.read_csv(target)
        original = cleaned_result.frame
        for column in ["C6H6(GT)", "RH", "AH", "CO(GT)"]:
            a = original[column].to_numpy(dtype=float)
            b = reread[column].to_numpy(dtype=float)
            assert np.nanmax(np.abs(a - b)) < 1e-6, column

    def test_exported_csv_keeps_missing_values_empty(self, tmp_path, cleaned_result):
        target = tmp_path / "airquality_clean.csv"
        export_clean_csv(cleaned_result.frame, target)
        reread = pd.read_csv(target)

        # NMHC(GT) is ~90% missing, so the column must load back with NaNs.
        assert int(reread["NMHC(GT)"].isna().sum()) == 8443
        assert reread["NMHC(GT)"].notna().sum() == 914

    def test_exported_csv_has_no_negative_200(self, tmp_path, cleaned_result):
        target = tmp_path / "airquality_clean.csv"
        export_clean_csv(cleaned_result.frame, target)
        reread = pd.read_csv(target)
        for column in MEASURED_COLUMNS:
            assert not (reread[column] == SENTINEL_VALUE).any(), column

    def test_exported_datetime_matches_the_frame(self, tmp_path, cleaned_result):
        target = tmp_path / "airquality_clean.csv"
        export_clean_csv(cleaned_result.frame, target)
        reread = pd.read_csv(target, parse_dates=[DATETIME_COLUMN])
        expected = cleaned_result.frame[DATETIME_COLUMN]
        assert reread[DATETIME_COLUMN].iloc[0] == expected.iloc[0]
        assert reread[DATETIME_COLUMN].iloc[-1] == expected.iloc[-1]

    def test_default_export_path_is_the_agreed_location(self):
        assert CLEANED_CSV.parent.name == "output"
        assert CLEANED_CSV.name == "airquality_clean.csv"
