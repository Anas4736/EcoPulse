"""Exploratory analysis functions for the EcoPulse dataset.

Every function computes its statistics **at runtime** from the data it is
given. Nothing is pre-baked, and every function that aggregates over time
also reports *how many observations* the aggregate is based on, so a number
built from 30 hours is never confused with one built from 9,000.

The functions fall into five groups:

* describing the data       - :func:`describe_variables`, :func:`pollutant_distribution`
* finding patterns in time  - :func:`hourly_profile`, :func:`daily_profile`,
  :func:`monthly_aggregate`, :func:`seasonal_aggregate`
* relating variables        - :func:`correlation_matrix`, :func:`sensor_reference_summary`
* episodes and gaps         - :func:`find_episodes`, :func:`missingness_by_period`
* sensor behaviour          - see :mod:`ecopulse.drift`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATETIME_COLUMN, MEASURED_COLUMNS, POLLUTANT_REFERENCE_COLUMNS


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one column as float64, treating missing values as NaN."""
    return pd.to_numeric(frame[column], errors="coerce")


def _period_key(stamps: pd.Series, period: str) -> pd.Series:
    """Convert timestamps into a period label (month, year, date...)."""
    stamps = pd.to_datetime(stamps)
    if period == "month":
        return stamps.dt.to_period("M").astype(str)
    if period == "year":
        return stamps.dt.year.astype(str)
    if period == "date":
        return stamps.dt.normalize()
    if period == "week":
        return stamps.dt.to_period("W").astype(str)
    if period == "season":
        return stamps.dt.month.map(
            {12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer",
             9: "Autumn", 10: "Autumn", 11: "Autumn"}
        )
    raise ValueError(f"Unsupported period: {period!r}")


# ---------------------------------------------------------------------------
# Describing the data
# ---------------------------------------------------------------------------
def describe_variables(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summary statistics for each variable, computed on valid values only.

    Missing values are excluded from the statistics but always reported in the
    ``Valid`` / ``Missing`` columns, so nothing is hidden.
    """
    targets = list(columns) if columns is not None else list(MEASURED_COLUMNS)

    rows = []
    for column in targets:
        values = _numeric(frame, column)
        valid = values.dropna()
        rows.append(
            {
                "Column": column,
                "Valid": int(valid.size),
                "Missing": int(values.isna().sum()),
                "Missing %": round(100.0 * values.isna().mean(), 2),
                "Mean": float(valid.mean()) if valid.size else np.nan,
                "Std": float(valid.std()) if valid.size > 1 else np.nan,
                "Min": float(valid.min()) if valid.size else np.nan,
                "Q1": float(valid.quantile(0.25)) if valid.size else np.nan,
                "Median": float(valid.median()) if valid.size else np.nan,
                "Q3": float(valid.quantile(0.75)) if valid.size else np.nan,
                "Max": float(valid.max()) if valid.size else np.nan,
                "IQR": (
                    float(valid.quantile(0.75) - valid.quantile(0.25))
                    if valid.size
                    else np.nan
                ),
                "Skew": float(valid.skew()) if valid.size > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pollutant_distribution(
    frame: pd.DataFrame,
    column: str,
    bins: int = 30,
) -> dict:
    """Histogram and summary for one pollutant, computed at runtime.

    Returns a dictionary with the histogram ``edges``/``counts`` (ready for a
    Plotly bar chart) and the summary statistics behind them.
    """
    values = _numeric(frame, column).dropna()

    if values.empty:
        return {
            "column": column,
            "edges": np.array([]),
            "counts": np.array([]),
            "n_valid": 0,
            "stats": {},
        }

    counts, edges = np.histogram(values.to_numpy(), bins=bins)

    return {
        "column": column,
        "edges": edges,
        "counts": counts,
        "n_valid": int(values.size),
        "stats": {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "skew": float(values.skew()) if values.size > 2 else np.nan,
        },
    }


# ---------------------------------------------------------------------------
# Patterns in time
# ---------------------------------------------------------------------------
def hourly_profile(
    frame: pd.DataFrame,
    column: str,
    by_season: bool = False,
) -> pd.DataFrame:
    """Average value of a variable for each hour of the day.

    The diurnal (24-hour) cycle is the strongest repeating pattern in an
    urban air-quality record, so this is a core exploratory view.
    """
    work = pd.DataFrame(
        {
            "value": _numeric(frame, column),
            "hour": pd.to_datetime(frame[DATETIME_COLUMN]).dt.hour,
        }
    )
    if by_season:
        work["season"] = _period_key(frame[DATETIME_COLUMN], "season")

        grouped = (
            work.groupby(["season", "hour"])["value"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )
        return grouped.rename(columns={"count": "n"})

    grouped = (
        work.groupby("hour")["value"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    return grouped


def daily_profile(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Daily mean, minimum, maximum and observation count for one variable."""
    work = pd.DataFrame(
        {
            "value": _numeric(frame, column),
            "date": pd.to_datetime(frame[DATETIME_COLUMN]).dt.normalize(),
        }
    )
    grouped = (
        work.groupby("date")["value"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    return grouped.rename(columns={"mean": "daily_mean", "min": "daily_min",
                                    "max": "daily_max"})


def monthly_aggregate(
    frame: pd.DataFrame,
    column: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate one variable per calendar month.

    ``agg`` may be any aggregation accepted by pandas, for example ``"mean"``,
    ``"median"``, ``"max"`` or ``"sum"``. The number of valid observations
    behind each month is always included.
    """
    work = pd.DataFrame(
        {
            "value": _numeric(frame, column),
            "month": _period_key(frame[DATETIME_COLUMN], "month"),
        }
    )
    grouped = (
        work.groupby("month")["value"]
        .agg(value=agg, n="count")
        .reset_index()
    )
    return grouped.rename(columns={"value": f"value_{agg}"})


def seasonal_aggregate(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate one variable per season (winter, spring, summer, autumn)."""
    work = pd.DataFrame(
        {
            "value": _numeric(frame, column),
            "season": _period_key(frame[DATETIME_COLUMN], "season"),
        }
    )
    order = ["Winter", "Spring", "Summer", "Autumn"]
    grouped = (
        work.groupby("season")["value"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    grouped["season"] = pd.Categorical(grouped["season"], categories=order, ordered=True)
    return grouped.sort_values("season").reset_index(drop=True)


def missingness_by_period(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    period: str = "month",
) -> pd.DataFrame:
    """How much of each variable is missing, per period.

    Used on the Data Quality page to show *when* the instrument was offline.
    """
    targets = list(columns) if columns is not None else list(MEASURED_COLUMNS)
    work = pd.DataFrame(index=frame.index)
    work["period"] = _period_key(frame[DATETIME_COLUMN], period)
    for column in targets:
        work[column] = _numeric(frame, column).isna()

    grouped = work.groupby("period")[targets].mean().mul(100).round(2)
    grouped.columns = [f"missing_pct_{column}" for column in grouped.columns]
    grouped["rows"] = work.groupby("period").size()
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Relationships between variables
# ---------------------------------------------------------------------------
def correlation_matrix(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlation matrix computed on pairwise-complete observations.

    pandas' ``.corr()`` already drops any pair where either value is missing,
    which is the correct behaviour here. Pairwise sample sizes are reported by
    :func:`pairwise_counts` because they differ between pairs.
    """
    targets = list(columns) if columns is not None else list(POLLUTANT_REFERENCE_COLUMNS)
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method=method)


def pairwise_counts(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Number of observations used for each pair in a correlation."""
    targets = list(columns) if columns is not None else list(POLLUTANT_REFERENCE_COLUMNS)
    numeric = frame[targets].apply(pd.to_numeric, errors="coerce")
    counts = pd.DataFrame(index=numeric.columns, columns=numeric.columns, dtype=int)
    for first in numeric.columns:
        for second in numeric.columns:
            counts.loc[first, second] = int(
                (numeric[first].notna() & numeric[second].notna()).sum()
            )
    return counts


def sensor_reference_summary(
    frame: pd.DataFrame,
    sensor: str,
    reference: str,
) -> dict:
    """How well does a sensor response track the reference concentration?

    This is the core measurement question of the dataset: can the cheap metal
    oxide sensor stand in for the certified analyser?

    The sensor column is a raw device response, not a concentration, so the
    relationship is described with a fitted line (``slope``/``intercept``)
    rather than a ratio.
    """
    data = pd.DataFrame(
        {
            "sensor": _numeric(frame, sensor),
            "reference": _numeric(frame, reference),
        }
    ).dropna()

    n = len(data)
    result = {
        "sensor": sensor,
        "reference": reference,
        "n_pairs": n,
        "n_sensor_only": int(_numeric(frame, sensor).notna().sum()),
        "n_reference_only": int(_numeric(frame, reference).notna().sum()),
        "n_both": n,
    }

    if n < 3:
        result.update(
            {"pearson_r": np.nan, "spearman_rho": np.nan, "slope": np.nan,
             "intercept": np.nan, "r_squared": np.nan, "rmse": np.nan,
             "mae": np.nan, "bias": np.nan, "p_value": np.nan}
        )
        return result

    regression = stats.linregress(data["sensor"], data["reference"])
    predicted = regression.intercept + regression.slope * data["sensor"]
    errors = predicted - data["reference"]

    result.update(
        {
            "pearson_r": float(regression.rvalue),
            "spearman_rho": float(stats.spearmanr(data["sensor"], data["reference"]).statistic),
            "slope": float(regression.slope),
            "intercept": float(regression.intercept),
            "r_squared": float(regression.rvalue ** 2),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "p_value": float(regression.pvalue),
        }
    )
    return result


def absolute_humidity_check(
    frame: pd.DataFrame,
    temperature: str = "T",
    humidity: str = "RH",
    absolute: str = "AH",
) -> dict:
    """Check how the dataset's ``AH`` column relates to temperature and humidity.

    ``AH`` is not in a standard unit of absolute humidity. Comparing it with
    the Magnus saturation vapour pressure shows it is that quantity scaled by
    the relative humidity and multiplied by a constant, which means it adds
    almost nothing to a model that already has ``T`` and ``RH``.

    This is checked at runtime rather than taken on trust, and the numbers
    are reported so the Methodology page can show its working.
    """
    t = _numeric(frame, temperature)
    rh = _numeric(frame, humidity) / 100.0
    ah = _numeric(frame, absolute)

    # Magnus saturation vapour pressure over water, temperature in Celsius.
    saturation = 6.112 * np.exp(17.625 * t / (243.04 + t))
    expected = saturation * rh

    valid = ah.notna() & expected.notna() & (expected != 0)
    ratio = ah[valid] / expected[valid]

    return {
        "n_pairs": int(valid.sum()),
        "pearson_r": float(ah[valid].corr(expected[valid])),
        "scale_factor": float(ratio.mean()),
        "scale_cv_pct": float(100.0 * ratio.std() / ratio.mean()),
        "correlation_with_temperature": float(ah[valid].corr(t[valid])),
        "correlation_with_humidity": float(ah[valid].corr(rh[valid])),
    }


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
def find_episodes(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
    min_duration: int = 1,
) -> pd.DataFrame:
    """Find continuous runs where a pollutant stays above ``threshold``.

    Rules, chosen so the result is defensible:

    * Only *valid* readings count. A missing value ends the run, because we
      cannot claim the pollutant was still high during an outage.
    * A run also ends when the timeline itself has a **gap**. If the record
      jumps from 01:00 to 05:00, the pollutant was not observed in between,
      so those hours cannot be part of a continuous episode. Without this
      rule an episode would be credited with hours that were never measured.
    * An episode therefore needs at least ``min_duration`` consecutive
      observed hours above the threshold.

    The threshold is supplied by the caller so the dashboard can let the user
    explore; it is never hard-coded behind the user's back.
    """
    values = _numeric(frame, column)
    stamps = pd.to_datetime(frame[DATETIME_COLUMN])

    above = (values > threshold).fillna(False).to_numpy()

    # True wherever the timeline jumps forward by more than one hour.
    time_gaps = np.diff(stamps.to_numpy()) > np.timedelta64(1, "h")

    episodes: list[tuple[int, int]] = []
    start: int | None = None
    for position, flag in enumerate(above):
        if flag:
            if start is None:
                start = position
            elif time_gaps[position - 1]:
                # The previous hour was never recorded, so the run stops.
                episodes.append((start, position - 1))
                start = position
        elif start is not None:
            episodes.append((start, position - 1))
            start = None
    if start is not None:
        episodes.append((start, len(above) - 1))

    rows = []
    for first, last in episodes:
        duration = last - first + 1
        if duration < min_duration:
            continue
        segment = values.iloc[first : last + 1]
        peak_value = float(segment.max())
        rows.append(
            {
                "start": stamps.iloc[first],
                "end": stamps.iloc[last],
                "duration_hours": duration,
                "peak_value": peak_value,
                "mean_value": float(segment.mean()),
                "peak_time": stamps.iloc[first + int(segment.values.argmax())],
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "start", "end", "duration_hours", "peak_value",
            "mean_value", "peak_time",
        ],
    )


def episode_summary(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
    min_duration: int = 1,
) -> dict:
    """Headline numbers for the episode analysis of one pollutant."""
    episodes = find_episodes(frame, column, threshold, min_duration=min_duration)
    values = _numeric(frame, column)
    valid = values.dropna()
    stamps = pd.to_datetime(frame[DATETIME_COLUMN])

    exceedance_hours = int((values > threshold).fillna(False).sum())

    per_year: dict[str, int] = {}
    if exceedance_hours:
        hours_above = stamps[values > threshold]
        per_year = {str(year): int(count) for year, count in hours_above.groupby(hours_above.dt.year).size().items()}

    return {
        "column": column,
        "threshold": threshold,
        "n_valid": int(valid.size),
        "n_exceedance_hours": exceedance_hours,
        "exceedance_pct_of_valid": (
            100.0 * exceedance_hours / valid.size if valid.size else 0.0
        ),
        "n_episodes": int(len(episodes)),
        "max_episode_hours": int(episodes["duration_hours"].max()) if len(episodes) else 0,
        "median_episode_hours": (
            float(episodes["duration_hours"].median()) if len(episodes) else 0.0
        ),
        "peak_value": float(valid.max()) if valid.size else np.nan,
        "exceedance_hours_by_year": per_year,
        "episodes": episodes,
    }


__all__ = [
    "describe_variables",
    "pollutant_distribution",
    "hourly_profile",
    "daily_profile",
    "monthly_aggregate",
    "seasonal_aggregate",
    "missingness_by_period",
    "correlation_matrix",
    "pairwise_counts",
    "sensor_reference_summary",
    "absolute_humidity_check",
    "find_episodes",
    "episode_summary",
]
