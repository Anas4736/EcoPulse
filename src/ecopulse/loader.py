"""Read the original UCI workbook and turn it into a tidy, time-aware frame.

This module is deliberately the *only* place that touches the source file, and
it opens the workbook in read-only fashion. Nothing in this project ever writes
to ``Data/AirQualityUCI.xlsx``.

What the loader does
--------------------
1. Reads the single worksheet with :func:`pandas.read_excel`.
2. Parses the ``Date`` and ``Time`` columns into one ``datetime`` column.
3. Leaves the ``-200`` missing markers exactly as they are - converting them
   is the job of :mod:`ecopulse.cleaning`.

Why the parsing needs care
---------------------------
The UCI project ships the same data in two encodings:

* ``AirQualityUCI.xlsx`` - dates arrive as real ``datetime`` objects and times
  as real ``time`` objects (this is the file we were given).
* ``AirQualityUCI.csv``  - dates as ``DD/MM/YYYY`` strings and times as
  ``HH.MM.SS`` strings using a dot separator.

Both are handled here so the loader keeps working if the CSV sibling is ever
substituted.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from .config import (
    DATE_COLUMN,
    DATETIME_COLUMN,
    MEASURED_COLUMNS,
    SOURCE_XLSX,
    TIME_COLUMN,
)


class DataLoadError(RuntimeError):
    """Raised when the source workbook is missing or does not look as expected."""


def _parse_dates(values: pd.Series) -> pd.Series:
    """Return a timezone-naive datetime series from the ``Date`` column.

    Handles both the Excel representation (already ``datetime64``) and the
    CSV representation (``DD/MM/YYYY`` text, therefore ``dayfirst=True``).
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values)

    text = values.astype(str).str.strip()
    # dayfirst=True because the dataset uses the Italian DD/MM/YYYY format.
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if parsed.isna().any():
        bad = text[parsed.isna()].unique()[:5]
        raise DataLoadError(f"Could not parse {len(parsed[parsed.isna()])} date values, e.g. {list(bad)}")
    return parsed


def _time_to_seconds(value: object) -> float:
    """Convert one time value into seconds after midnight.

    Accepts ``datetime.time``, a ``datetime.datetime``/``pandas.Timestamp``,
    or a text time such as ``"18.00.00"`` / ``"18:00:00"``.
    """
    if isinstance(value, pd.Timestamp):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, dt.datetime):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, dt.time):
        return value.hour * 3600 + value.minute * 60 + value.second

    parts = [p for p in re.split(r"[.:]", str(value).strip()) if p != ""]
    if len(parts) < 2:
        raise DataLoadError(f"Unrecognised time value: {value!r}")
    hours, minutes = int(parts[0]), int(parts[1])
    seconds = int(parts[2]) if len(parts) > 2 else 0
    return hours * 3600 + minutes * 60 + seconds


def _parse_times(values: pd.Series) -> pd.Series:
    """Return a float series of seconds after midnight for the ``Time`` column."""
    try:
        seconds = values.map(_time_to_seconds)
    except DataLoadError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise DataLoadError(f"Failed to parse the Time column: {exc}") from exc

    if seconds.isna().any():
        raise DataLoadError(f"{int(seconds.isna().sum())} time values could not be parsed")
    return seconds.astype("float64")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the source workbook into a DataFrame with a merged ``datetime``.

    Parameters
    ----------
    path
        Workbook to read. Defaults to the original UCI file in ``Data/``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Date``, ``Time``, the 13 measured variables, plus a new
        ``datetime`` column. Rows are sorted oldest first. The ``-200``
        markers are still present at this stage.
    """
    source = Path(path) if path is not None else SOURCE_XLSX
    if not source.exists():
        raise DataLoadError(f"Source workbook not found: {source}")

    try:
        frame = pd.read_excel(source)
    except Exception as exc:  # pragma: no cover - defensive
        raise DataLoadError(f"Could not read {source}: {exc}") from exc

    expected = {DATE_COLUMN, TIME_COLUMN, *MEASURED_COLUMNS}
    missing = expected.difference(frame.columns)
    if missing:
        raise DataLoadError(
            f"{source.name} is missing expected columns: {sorted(missing)}"
        )

    # Keep only the columns we know about, in a stable order.
    frame = frame[[DATE_COLUMN, TIME_COLUMN, *MEASURED_COLUMNS]].copy()

    dates = _parse_dates(frame[DATE_COLUMN])
    seconds = _parse_times(frame[TIME_COLUMN])
    frame[DATETIME_COLUMN] = dates + pd.to_timedelta(seconds, unit="s")

    frame = frame.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    return frame


__all__ = ["DataLoadError", "load_raw"]
