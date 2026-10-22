"""Feature engineering for the EcoPulse dataset.

Every function here takes a cleaned frame (see :mod:`ecopulse.cleaning`) and
returns a **new** frame with extra columns. Nothing is modified in place.

What is derived, and why
------------------------
``add_time_features``
    Calendar parts (year, month, hour, weekday...) plus *cyclical* encodings
    for the hour of day and the day of year. A cyclical encoding is used
    because 23:00 and 00:00 are neighbours, not opposite ends of a scale -
    a plain integer hour would make them look maximally different.

``add_missing_indicators``
    One 0/1 flag per measured column plus a row-level count of how many
    variables are missing. Missingness is *informative* here: the device
    outages are not spread uniformly through the year, so the model should be
    allowed to see that.

``add_lag_features``
    Values of a column at earlier times (t-1, t-24, t-168). These are only
    valid when the thing being predicted happens at time *t* and the lags are
    strictly in the past - that is the retrospective setting used by the
    benzene regression. For a live forecast of t+1 the lags would have to be
    shifted, which is why :func:`add_rolling_features` exposes a ``shift``
    argument.

``add_rolling_features``
    Trailing window statistics (mean, median, standard deviation). These are
    used for smoothing time-series charts and for building a seasonal
    baseline for the sensor-drift investigation.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import DATETIME_COLUMN, MEASURED_COLUMNS

#: Northern-hemisphere seasons, which is where the study city is located.
SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

#: Lag values in hours that are justified for this dataset.
#:   1   -> the previous hour (short-term persistence)
#:   24  -> the same hour on the previous day (diurnal pattern)
#:   168 -> the same hour on the previous week (weekly pattern)
DEFAULT_LAGS: tuple[int, ...] = (1, 24, 168)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return ``columns`` after checking that the frame really has them.

    A missing column is a mistake in the caller, not something to work
    around silently, so this raises instead of quietly skipping the column.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing these columns: {missing}")
    return list(columns)


def indicator_name(column: str) -> str:
    """Return a safe column name for a missing-value indicator.

    ``"CO(GT)"`` becomes ``"missing_CO_GT"`` so the name can be used with
    ordinary attribute access and in Plotly traces.
    """
    return "missing_" + re.sub(r"[^0-9A-Za-z]+", "_", column).strip("_")


def lag_name(column: str, lag: int) -> str:
    """Return the column name used for a lagged feature."""
    return f"{re.sub(r'[^0-9A-Za-z]+', '_', column).strip('_')}_lag{lag}"


def rolling_name(column: str, window: int, stat: str, shift: int = 0) -> str:
    """Return the column name used for a rolling feature."""
    suffix = f"_shift{shift}" if shift else ""
    return f"{re.sub(r'[^0-9A-Za-z]+', '_', column).strip('_')}_roll{window}_{stat}{suffix}"


def add_time_features(
    frame: pd.DataFrame,
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """Add calendar features derived from the ``datetime`` column.

    Parameters
    ----------
    frame
        Cleaned frame containing ``datetime``.
    include_cyclical
        When ``True`` (default) also add ``hour_sin``/``hour_cos`` (24-hour
        cycle) and ``doy_sin``/``doy_cos`` (annual cycle).

    Returns
    -------
    pandas.DataFrame
        A copy of ``frame`` with the new columns appended.
    """
    if DATETIME_COLUMN not in frame.columns:
        raise ValueError(f"Frame is missing the '{DATETIME_COLUMN}' column")

    result = frame.copy()
    stamps = pd.to_datetime(result[DATETIME_COLUMN])

    result["year"] = stamps.dt.year
    result["month"] = stamps.dt.month
    result["month_name"] = stamps.dt.month_name()
    result["day"] = stamps.dt.day
    result["hour"] = stamps.dt.hour
    result["dayofweek"] = stamps.dt.dayofweek          # Monday = 0
    result["day_name"] = stamps.dt.day_name()
    result["quarter"] = stamps.dt.quarter
    result["dayofyear"] = stamps.dt.dayofyear
    result["is_weekend"] = stamps.dt.dayofweek >= 5
    result["season"] = stamps.dt.month.map(SEASON_BY_MONTH)
    result["date"] = stamps.dt.normalize()
    result["is_daytime"] = result["hour"].between(6, 21)

    if include_cyclical:
        # Angle in radians: 2*pi for one full turn.
        hour_angle = 2 * np.pi * result["hour"] / 24.0
        result["hour_sin"] = np.sin(hour_angle)
        result["hour_cos"] = np.cos(hour_angle)

        day_angle = 2 * np.pi * result["dayofyear"] / 365.25
        result["doy_sin"] = np.sin(day_angle)
        result["doy_cos"] = np.cos(day_angle)

    return result


def add_missing_indicators(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add a 0/1 flag for every column that has missing values, plus a total.

    Parameters
    ----------
    frame
        Cleaned frame.
    columns
        Which columns to flag. Defaults to all 13 measured variables.
    """
    result = frame.copy()
    targets = _require_columns(result, list(columns) if columns is not None else list(MEASURED_COLUMNS))

    missing = result[targets].isna()
    for column in targets:
        result[indicator_name(column)] = missing[column].astype("int8")
    result["missing_count"] = missing.sum(axis=1).astype("int16")
    result["is_complete"] = (result["missing_count"] == 0).astype("int8")
    return result


def add_lag_features(
    frame: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add strictly-past values of ``columns`` as new features.

    Lags are taken by **calendar time**, not by row position: the value at
    ``t - 24 hours`` is looked up by timestamp. That keeps the feature correct
    even if a gap is introduced into the timeline (the lookup simply returns
    missing for an hour that does not exist).

    Notes
    -----
    * A lag of 1 hour is only a valid predictor for a value measured at the
      *current* hour in a retrospective setting (the previous hour is already
      known). For a genuine forecast of the next hour the same feature would
      have to be shifted, which is what ``shift`` in
      :func:`add_rolling_features` is for.
    """
    result = frame.copy()
    stamps = pd.to_datetime(result[DATETIME_COLUMN])
    _require_columns(result, list(columns))

    for column in columns:
        # A timestamp-indexed series that we can look values up in.
        lookup = pd.Series(
            pd.to_numeric(result[column], errors="coerce").to_numpy(),
            index=stamps,
        )
        for lag in lags:
            wanted = stamps - pd.Timedelta(hours=lag)
            result[lag_name(column, lag)] = lookup.reindex(wanted).to_numpy()

    return result


def add_rolling_features(
    frame: pd.DataFrame,
    columns: list[str],
    window: int = 24,
    stats: tuple[str, ...] = ("mean",),
    shift: int = 0,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add trailing window statistics for ``columns``.

    The window is measured in **hours of calendar time**. The series is
    re-indexed onto a complete hourly timeline before rolling, so a window
    always covers exactly ``window`` hours even if the input contains gaps
    (missing hours become missing values instead of silently shrinking the
    window).

    Parameters
    ----------
    window
        Size of the trailing window in hours (24 = one day of hourly data).
    stats
        Any of ``"mean"``, ``"median"``, ``"std"``, ``"min"``, ``"max"``.
    shift
        Number of hours to shift the result backwards. Use ``shift=1`` for
        any feature that will be used to *predict* a later value, so the
        window cannot include the value being predicted.
    min_periods
        Minimum number of observations required. The default of 1 lets early
        rows use a partial window, which is fine for charting.
    """
    result = frame.copy()
    stamps = pd.to_datetime(result[DATETIME_COLUMN])
    _require_columns(result, list(columns))

    # Complete hourly timeline covering the whole frame.
    full_index = pd.date_range(
        stamps.min(), stamps.max(), freq="h", name=DATETIME_COLUMN
    )

    for column in columns:
        lookup = pd.Series(
            pd.to_numeric(result[column], errors="coerce").to_numpy(),
            index=stamps,
        )
        aligned = lookup.reindex(full_index)
        rolling = aligned.rolling(window=window, min_periods=min_periods)

        for stat in stats:
            if stat == "mean":
                computed = rolling.mean()
            elif stat == "median":
                computed = rolling.median()
            elif stat == "std":
                computed = rolling.std()
            elif stat == "min":
                computed = rolling.min()
            elif stat == "max":
                computed = rolling.max()
            else:
                raise ValueError(f"Unsupported rolling statistic: {stat!r}")

            if shift:
                computed = computed.shift(shift)

            # Map the computed window back onto the original rows.
            result[rolling_name(column, window, stat, shift)] = (
                computed.reindex(stamps).to_numpy()
            )

    return result


def build_feature_frame(
    frame: pd.DataFrame,
    lag_columns: list[str] | None = None,
    rolling_columns: list[str] | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_window: int = 24,
) -> pd.DataFrame:
    """Build the full feature set used across the dashboard.

    Parameters
    ----------
    lag_columns
        Columns to build lags for. Defaults to the four reference analytes
        that are actually modelled or charted.
    rolling_columns
        Columns to build a 24-hour rolling mean for. Defaults to the same
        pollutant set.
    """
    if lag_columns is None:
        lag_columns = ["CO(GT)", "C6H6(GT)", "NO2(GT)", "NOx(GT)"]
    if rolling_columns is None:
        rolling_columns = list(lag_columns)

    # The defaults describe the columns this project normally models. If a
    # caller works with a smaller frame, the defaults that the frame does
    # not have are simply skipped - but a column that was asked for
    # explicitly and is not there is still an error.
    lag_columns = [column for column in lag_columns if column in frame.columns]
    rolling_columns = [column for column in rolling_columns if column in frame.columns]

    result = add_time_features(frame)
    result = add_missing_indicators(
        result, columns=[c for c in MEASURED_COLUMNS if c in result.columns]
    )
    result = add_lag_features(result, lag_columns, lags=lags)
    result = add_rolling_features(
        result, rolling_columns, window=rolling_window, stats=("mean",)
    )
    return result


__all__ = [
    "SEASON_BY_MONTH",
    "DEFAULT_LAGS",
    "indicator_name",
    "lag_name",
    "rolling_name",
    "add_time_features",
    "add_missing_indicators",
    "add_lag_features",
    "add_rolling_features",
    "build_feature_frame",
]
