"""Build a data-quality report for the air-quality dataset.

The report answers three questions that the dashboard shows to the user:

1. **What was missing before cleaning?** - the raw count of ``-200`` markers
   per column.
2. **What is missing afterwards?** - the same columns after the markers have
   been converted to real missing values (``NaN``).
3. **Why was it missing?** - the report groups columns that share the *same*
   missing pattern, which reveals the distinct failure mechanisms (device
   outage vs. reference-analyzer outage vs. a channel that was never filled).

Grouping is computed from the data rather than hard-coded, so the report stays
honest if the source file is ever swapped for a different vintage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    COLUMN_METADATA,
    DATETIME_COLUMN,
    MEASURED_COLUMNS,
    SENTINEL_VALUE,
)

#: Any measured value below this is treated as "not physically plausible".
#: The coldest legitimate reading in this dataset is T = -1.9 degC, so -50 is a
#: very safe floor that still catches other sentinel styles (-999, -1e9, ...).
NON_PHYSICAL_FLOOR = -50.0


@dataclass
class ColumnQuality:
    """Per-column quality statistics for one measured variable."""

    column: str
    label: str
    unit: str
    role: str
    sentinel_count: int
    sentinel_pct: float
    valid_count: int
    missing_after_clean: int
    missing_after_clean_pct: float
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    median_value: float | None
    non_physical_count: int


@dataclass
class OutageBlock:
    """A continuous run of hours during which a group of columns is missing."""

    start: pd.Timestamp
    end: pd.Timestamp
    hours: int
    columns: list[str]

    def as_row(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "hours": self.hours,
            "columns": ", ".join(self.columns),
        }


@dataclass
class QualityReport:
    """The complete before/after picture of the cleaning step."""

    source_file: str
    rows_read: int
    rows_after_clean: int
    rows_dropped: int
    duplicate_rows_removed: int
    duplicate_timestamps_removed: int
    datetime_start: pd.Timestamp | None
    datetime_end: pd.Timestamp | None
    expected_hours: int
    missing_hours: int
    gap_count: int
    largest_gap_hours: int
    sentinel_value: float
    columns: list[ColumnQuality] = field(default_factory=list)
    missing_pattern_groups: list[list[str]] = field(default_factory=list)
    outages: list[OutageBlock] = field(default_factory=list)

    @property
    def rows_retained_pct(self) -> float:
        """Percentage of the rows read that survive cleaning."""
        if self.rows_read == 0:
            return 0.0
        return 100.0 * self.rows_after_clean / self.rows_read

    def column(self, name: str) -> ColumnQuality | None:
        """Return the quality entry for one column, or ``None``."""
        for entry in self.columns:
            if entry.column == name:
                return entry
        return None


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity of two missing-value patterns (1.0 = identical)."""
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return intersection / union if union else 1.0


def group_columns_by_missing_pattern(
    raw: pd.DataFrame, similarity_threshold: float = 0.9
) -> list[list[str]]:
    """Group measured columns whose missing values occur at the same times.

    Columns that go missing together almost certainly fail together, so the
    grouping is the fastest way to explain *why* data is missing.

    Two columns join the same group when the *Jaccard similarity* of their
    missing-value patterns is at least ``similarity_threshold``. A strict
    "identical pattern" test is not enough here: ``NOx(GT)`` and ``NO2(GT)``
    share the same analyzer channel and miss 1,639 of the same hours, but
    three individual hours differ, which would incorrectly split them apart.
    """
    mask = raw[MEASURED_COLUMNS] == SENTINEL_VALUE
    if not mask.to_numpy().any():
        raise ValueError(
            "group_columns_by_missing_pattern() expects the RAW frame (with the "
            f"{SENTINEL_VALUE} markers still in place); the supplied frame has no "
            "missing markers left."
        )

    signatures = {column: mask[column].to_numpy() for column in MEASURED_COLUMNS}

    groups: list[list[str]] = []
    assigned: set[str] = set()

    for column in MEASURED_COLUMNS:
        if column in assigned:
            continue
        group = [column]
        assigned.add(column)
        for other in MEASURED_COLUMNS:
            if other in assigned:
                continue
            if _jaccard(signatures[column], signatures[other]) >= similarity_threshold:
                group.append(other)
                assigned.add(other)
        groups.append(group)

    # Largest group first - it is usually the most interesting one.
    groups.sort(key=len, reverse=True)
    return groups


def missing_pattern_overlaps(raw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard similarity of every pair of measured columns.

    ``1.00`` means the two columns are missing at exactly the same hours.
    """
    mask = raw[MEASURED_COLUMNS] == SENTINEL_VALUE
    rows = []
    for first in MEASURED_COLUMNS:
        for second in MEASURED_COLUMNS:
            if first >= second:
                continue
            rows.append(
                {
                    "Column A": first,
                    "Column B": second,
                    "Shared missing hours": int(
                        (mask[first] & mask[second]).sum()
                    ),
                    "Jaccard similarity": round(
                        _jaccard(mask[first].to_numpy(), mask[second].to_numpy()), 4
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "Jaccard similarity", ascending=False
    ).reset_index(drop=True)


def find_outage_blocks(
    raw: pd.DataFrame, columns: list[str]
) -> list[OutageBlock]:
    """Find continuous runs where *all* of ``columns`` carry the sentinel."""
    if not columns:
        return []

    mask = (raw[columns] == SENTINEL_VALUE).all(axis=1).to_numpy()
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []

    blocks: list[OutageBlock] = []
    # Split the sorted positions into continuous runs.
    splits = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
    times = raw[DATETIME_COLUMN]
    for run in splits:
        blocks.append(
            OutageBlock(
                start=times.iloc[run[0]],
                end=times.iloc[run[-1]],
                hours=int(run.size),
                columns=list(columns),
            )
        )
    return blocks


def build_quality_report(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    source_file: str,
    duplicate_rows_removed: int = 0,
    duplicate_timestamps_removed: int = 0,
) -> QualityReport:
    """Compare the raw frame with the cleaned frame and describe both.

    Parameters
    ----------
    raw
        Frame straight from :func:`ecopulse.loader.load_raw` (sentinels intact).
    clean
        Frame after :func:`ecopulse.cleaning.clean_frame` (sentinels -> NaN).
    source_file
        Human readable name of the source workbook, for display.
    """
    rows_read = len(raw)
    rows_after = len(clean)

    start = clean[DATETIME_COLUMN].iloc[0] if rows_after else None
    end = clean[DATETIME_COLUMN].iloc[-1] if rows_after else None

    expected_hours = 0
    missing_hours = 0
    gap_count = 0
    largest_gap = 0
    if rows_after:
        differences = clean[DATETIME_COLUMN].diff().dt.total_seconds() / 3600
        gaps = differences[differences > 1]
        gap_count = int(gaps.size)
        largest_gap = int(gaps.max() - 1) if gap_count else 0
        expected_hours = int((end - start).total_seconds() / 3600) + 1
        missing_hours = expected_hours - rows_after

    columns: list[ColumnQuality] = []
    for name in MEASURED_COLUMNS:
        meta = COLUMN_METADATA.get(name, {})
        raw_values = pd.to_numeric(raw[name], errors="coerce")
        sentinel_count = int((raw_values == SENTINEL_VALUE).sum())

        clean_values = pd.to_numeric(clean[name], errors="coerce")
        missing_after = int(clean_values.isna().sum())
        valid = clean_values.dropna()

        columns.append(
            ColumnQuality(
                column=name,
                label=meta.get("label", name),
                unit=meta.get("unit", ""),
                role=meta.get("role", ""),
                sentinel_count=sentinel_count,
                sentinel_pct=100.0 * sentinel_count / rows_read if rows_read else 0.0,
                valid_count=int(valid.size),
                missing_after_clean=missing_after,
                missing_after_clean_pct=(
                    100.0 * missing_after / rows_after if rows_after else 0.0
                ),
                min_value=float(valid.min()) if valid.size else None,
                max_value=float(valid.max()) if valid.size else None,
                mean_value=float(valid.mean()) if valid.size else None,
                median_value=float(valid.median()) if valid.size else None,
                non_physical_count=int((valid <= NON_PHYSICAL_FLOOR).sum()),
            )
        )

    groups: list[list[str]] = []
    outages: list[OutageBlock] = []
    if (raw[MEASURED_COLUMNS] == SENTINEL_VALUE).to_numpy().any():
        groups = group_columns_by_missing_pattern(raw)
        for group in groups:
            outages.extend(find_outage_blocks(raw, group))
        outages.sort(key=lambda block: block.start)

    return QualityReport(
        source_file=source_file,
        rows_read=rows_read,
        rows_after_clean=rows_after,
        rows_dropped=rows_read - rows_after,
        duplicate_rows_removed=duplicate_rows_removed,
        duplicate_timestamps_removed=duplicate_timestamps_removed,
        datetime_start=start,
        datetime_end=end,
        expected_hours=expected_hours,
        missing_hours=missing_hours,
        gap_count=gap_count,
        largest_gap_hours=largest_gap,
        sentinel_value=SENTINEL_VALUE,
        columns=columns,
        missing_pattern_groups=groups,
        outages=outages,
    )


def before_after_table(report: QualityReport) -> pd.DataFrame:
    """Return the before/after cleaning comparison as a tidy DataFrame.

    This is the table rendered on the *Data Quality* page of the dashboard.
    """
    rows = []
    for entry in report.columns:
        rows.append(
            {
                "Column": entry.column,
                "Label": entry.label,
                "Unit": entry.unit,
                "Role": entry.role,
                "Raw -200 markers": entry.sentinel_count,
                "Raw missing %": round(entry.sentinel_pct, 2),
                "Missing after cleaning": entry.missing_after_clean,
                "Missing after cleaning %": round(entry.missing_after_clean_pct, 2),
                "Valid values": entry.valid_count,
                "Min": entry.min_value,
                "Max": entry.max_value,
                "Mean": entry.mean_value,
                "Median": entry.median_value,
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(report: QualityReport) -> dict:
    """Headline numbers for the metric cards on the Data Quality page."""
    total_markers = sum(entry.sentinel_count for entry in report.columns)
    total_slots = report.rows_read * len(MEASURED_COLUMNS)
    return {
        "rows_read": report.rows_read,
        "rows_after_clean": report.rows_after_clean,
        "rows_retained_pct": report.rows_retained_pct,
        "duplicate_rows_removed": report.duplicate_rows_removed,
        "duplicate_timestamps_removed": report.duplicate_timestamps_removed,
        "total_sentinel_markers": total_markers,
        "sentinel_pct_of_cells": 100.0 * total_markers / total_slots if total_slots else 0.0,
        "missing_hours": report.missing_hours,
        "gap_count": report.gap_count,
        "largest_gap_hours": report.largest_gap_hours,
        "non_physical_total": sum(e.non_physical_count for e in report.columns),
        "date_start": report.datetime_start,
        "date_end": report.datetime_end,
    }


__all__ = [
    "ColumnQuality",
    "OutageBlock",
    "QualityReport",
    "NON_PHYSICAL_FLOOR",
    "group_columns_by_missing_pattern",
    "missing_pattern_overlaps",
    "find_outage_blocks",
    "build_quality_report",
    "before_after_table",
    "summary_metrics",
]
