"""Turn the raw workbook into a clean, analysis-ready dataset.

Cleaning rules used here (and nothing else):

* The ``-200`` marker is converted to a real missing value (``NaN``) **only in
  the 13 measured columns**. ``Date`` and ``Time`` are timestamps, so the
  marker can never apply to them.
* Every measured column is cast to ``float64`` so that columns such as
  ``NMHC(GT)`` (stored as integers) can hold missing values.
* Exact duplicate rows are dropped, and if two rows still share the same
  timestamp only the first is kept - the timeline must have one value per hour.
* No value is ever invented. Missing measurements stay missing; they are
  handled at aggregation time (skip missing) or inside a scikit-learn
  imputation pipeline at modelling time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CLEANED_CSV,
    DATETIME_COLUMN,
    DATE_COLUMN,
    MEASURED_COLUMNS,
    PROCESSED_DIR,
    SENTINEL_VALUE,
    SOURCE_XLSX,
    TIME_COLUMN,
)
from .loader import load_raw
from .quality import QualityReport, build_quality_report


@dataclass
class CleaningResult:
    """The cleaned frame together with the report describing what changed."""

    frame: pd.DataFrame
    report: QualityReport

    @property
    def valid_mask(self) -> pd.Series:
        """Boolean mask of rows where every measured variable is present."""
        return self.frame[MEASURED_COLUMNS].notna().all(axis=1)


def clean_frame(raw: pd.DataFrame, source_file: str = "") -> CleaningResult:
    """Apply the cleaning rules to a raw frame.

    The input frame is never modified - a copy is made first.
    """
    required = {DATE_COLUMN, TIME_COLUMN, DATETIME_COLUMN, *MEASURED_COLUMNS}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Cannot clean: expected columns are missing {sorted(missing)}")

    frame = raw.copy()

    # --- Step 1: -200 -> NaN, only in the measured columns ------------------
    for column in MEASURED_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        # .mask() replaces the sentinel positions with NaN and leaves the rest.
        values = values.mask(values == SENTINEL_VALUE)
        frame[column] = values.astype("float64")

    # --- Step 2: remove exact duplicate rows --------------------------------
    duplicate_rows_removed = int(frame.duplicated(keep="first").sum())
    if duplicate_rows_removed:
        frame = frame.drop_duplicates(keep="first")

    # --- Step 3: one row per timestamp ---------------------------------------
    frame = frame.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    duplicate_timestamps_removed = int(
        frame.duplicated(subset=[DATETIME_COLUMN], keep="first").sum()
    )
    if duplicate_timestamps_removed:
        frame = (
            frame.drop_duplicates(subset=[DATETIME_COLUMN], keep="first")
            .sort_values(DATETIME_COLUMN)
            .reset_index(drop=True)
        )

    report = build_quality_report(
        raw=raw,
        clean=frame,
        source_file=source_file or SOURCE_XLSX.name,
        duplicate_rows_removed=duplicate_rows_removed,
        duplicate_timestamps_removed=duplicate_timestamps_removed,
    )
    return CleaningResult(frame=frame, report=report)


def export_clean_csv(
    frame: pd.DataFrame,
    path: str | Path = CLEANED_CSV,
) -> Path:
    """Write the cleaned dataset to CSV and return the path written.

    The export keeps the original ``Date`` and ``Time`` columns for
    readability, adds the merged ``datetime`` column used for analysis, and
    writes the 13 measured variables with empty cells where a value is
    missing (CSV has no representation for ``NaN``).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Build the export explicitly so that each column gets a readable format:
    #   Date     -> 2004-03-10        (date only, no misleading midnight time)
    #   Time     -> 18:00:00
    #   datetime -> 2004-03-10 18:00:00  (the analysis key)
    export = pd.DataFrame(index=frame.index)
    export[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    export[TIME_COLUMN] = frame[TIME_COLUMN].astype(str)
    export[DATETIME_COLUMN] = pd.to_datetime(frame[DATETIME_COLUMN]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    for column in MEASURED_COLUMNS:
        export[column] = frame[column].astype("float64")

    export.to_csv(destination, index=False, float_format="%.10g")
    return destination


def load_clean_dataset(
    source: str | Path | None = None,
) -> CleaningResult:
    """Convenience helper: read the original workbook and clean it in one go.

    This is the single call the Streamlit app will use (wrapped in
    ``st.cache_data``), so the expensive work happens once per session.
    """
    raw = load_raw(source)
    return clean_frame(raw, source_file=Path(source or SOURCE_XLSX).name)


def _print_report(result: CleaningResult) -> None:
    """Print a readable summary of the cleaning step to the console."""
    report = result.report
    print("=" * 70)
    print("EcoPulse - data cleaning report")
    print("=" * 70)
    print(f"Source file           : {report.source_file}")
    print(f"Rows read             : {report.rows_read}")
    print(f"Rows after cleaning   : {report.rows_after_clean} "
          f"({report.rows_retained_pct:.2f}% retained)")
    print(f"Rows dropped          : {report.rows_dropped}")
    print(f"Duplicate rows removed: {report.duplicate_rows_removed}")
    print(f"Duplicate timestamps  : {report.duplicate_timestamps_removed}")
    if report.datetime_start is not None:
        print(f"Time range            : {report.datetime_start} -> {report.datetime_end}")
    print(f"Expected hourly slots : {report.expected_hours}")
    print(f"Missing hours (gaps)  : {report.missing_hours} "
          f"across {report.gap_count} gap(s)")
    print(f"Largest gap           : {report.largest_gap_hours} hour(s)")
    print()
    print(f"Missing-value groups (columns that fail together):")
    for index, group in enumerate(report.missing_pattern_groups, start=1):
        print(f"  {index}. {', '.join(group)}")
    print()
    print(f"{'column':16} {'-200':>7} {'missing %':>11} {'valid':>7} {'min':>12} {'max':>12}")
    for entry in report.columns:
        print(
            f"{entry.column:16} {entry.sentinel_count:7d} "
            f"{entry.sentinel_pct:10.2f}% {entry.valid_count:7d} "
            f"{entry.min_value:12.4g} {entry.max_value:12.4g}"
        )
    print()


def main() -> int:
    """Command line entry point: ``python -m ecopulse.cleaning``."""
    result = load_clean_dataset()
    written = export_clean_csv(result.frame)
    _print_report(result)
    print(f"Cleaned CSV written to: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
