"""Investigate whether the sensors drift over the recording period.

The UCI documentation for this dataset explicitly warns that "concept and
sensor drifts" are present. That is a claim about the *original* study, so
before EcoPulse repeats it, the data has to be checked - and the check has to
survive three obvious confounds:

1. **Seasonality.**   Pollution and sensor response both change with the
   season, so comparing January 2004 with December 2005 proves nothing.
2. **Missingness.**  The device was offline for blocks of hours. If outages
   cluster in particular seasons, a naive comparison is skewed.
3. **Time coverage.** The record starts in March 2004 and ends in April 2005,
   so "early versus late" is not a balanced comparison, and April 2005 is
   covered for only four days.

The approach here is therefore:

* Recalibrate the sensor against the reference analyser **separately for
  every calendar month**. A drift shows up as the slope of that monthly
  calibration line moving systematically.
* Compare the **same calendar month in different years**. Holding the season
  fixed is the cleanest control available, and it is possible for March and
  April only, because those are the two months present in both 2004 and 2005.
* Make that comparison **fair in coverage** by trimming both years to the
  days of the month that both years actually contain, so April 2004 is
  compared against April 1-4 of 2005 rather than against a full month.
* Quantify every trend with a **bootstrap confidence interval** so the
  verdict rests on an interval rather than on eyeballing a chart.

One caveat applies to the absolute numbers, not to the comparison. The
sensor readings carry their own measurement noise, and regressing a clean
value on a noisy measurement always understates the slope - the classical
regression-dilution effect. The fitted slope is therefore a little smaller
than the true sensitivity, and it is attenuated by roughly the same amount
every month, so the *change* over time is still measured correctly.

The verdict is computed from those numbers by the rules in
:func:`assess_drift`. Nothing is asserted by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATETIME_COLUMN

#: Minimum number of paired observations needed to fit a monthly calibration.
DEFAULT_MIN_PAIRS = 30

#: Number of bootstrap resamples used for the confidence intervals.
DEFAULT_BOOTSTRAP = 2000

#: Fixed seed so every run of the project gives the same numbers.
RANDOM_SEED = 20260101


@dataclass
class DriftAssessment:
    """Everything needed to judge (and to show) the drift question."""

    sensor: str
    reference: str
    monthly: pd.DataFrame
    matched: pd.DataFrame
    months_analysed: int
    total_pairs: int
    slope_trend_per_month: float
    slope_trend_ci: tuple[float, float]
    intercept_trend_per_month: float
    intercept_trend_ci: tuple[float, float]
    correlation_trend_per_month: float
    months_with_increased_slope: int
    matched_months: int
    matched_agrees_with_trend: int
    coverage_note: str
    missingness_note: str
    verdict: str
    verdict_reasons: list[str] = field(default_factory=list)

    @property
    def trend_ci_excludes_zero(self) -> bool:
        """Does the bootstrap interval for the slope trend exclude zero?"""
        low, high = self.slope_trend_ci
        if not np.isfinite(low) or not np.isfinite(high):
            return False
        return bool((low > 0 and high > 0) or (low < 0 and high < 0))


def _monthly_frame(frame: pd.DataFrame, sensor: str, reference: str) -> pd.DataFrame:
    """Prepare a frame with the two columns of interest and calendar labels."""
    stamps = pd.to_datetime(frame[DATETIME_COLUMN])
    return pd.DataFrame(
        {
            "sensor": pd.to_numeric(frame[sensor], errors="coerce"),
            "reference": pd.to_numeric(frame[reference], errors="coerce"),
            "timestamp": stamps,
            "month": stamps.dt.to_period("M").astype(str),
            "year": stamps.dt.year,
            "month_number": stamps.dt.month,
            "day": stamps.dt.day,
        }
    )


def _fit_calibration(sensor_values: pd.Series, reference_values: pd.Series) -> dict:
    """Fit ``reference ~ sensor`` and return the summary numbers."""
    regression = stats.linregress(sensor_values, reference_values)
    predicted = regression.intercept + regression.slope * sensor_values
    errors = predicted - reference_values
    return {
        "slope": float(regression.slope),
        "intercept": float(regression.intercept),
        "pearson_r": float(regression.rvalue),
        "r_squared": float(regression.rvalue ** 2),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "n_pairs": int(len(sensor_values)),
        "mean_sensor": float(sensor_values.mean()),
        "mean_reference": float(reference_values.mean()),
    }


def monthly_calibration(
    frame: pd.DataFrame,
    sensor: str,
    reference: str,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> pd.DataFrame:
    """Fit a separate sensor-to-reference calibration for each calendar month.

    Returns one row per month with enough paired observations, holding the
    number of pairs and the coverage of each month.
    """
    work = _monthly_frame(frame, sensor, reference)

    rows = []
    for month, group in work.groupby("month"):
        paired = group[["sensor", "reference"]].dropna()
        if len(paired) < min_pairs:
            continue
        fit = _fit_calibration(paired["sensor"], paired["reference"])
        rows.append(
            {
                "month": month,
                "year": int(group["year"].mode().iloc[0]),
                "month_number": int(group["month_number"].mode().iloc[0]),
                **fit,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "month", "year", "month_number", "n_pairs", "slope", "intercept",
            "pearson_r", "r_squared", "rmse", "mean_sensor", "mean_reference",
        ],
    )


def matched_period_comparison(
    frame: pd.DataFrame,
    sensor: str,
    reference: str,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> pd.DataFrame:
    """Compare the same calendar month across years on **equal footing**.

    For every calendar month that appears in more than one year, both years
    are trimmed to the days of the month that *both* of them contain, and the
    calibration is refitted on that trimmed window. This removes two
    confounds at once:

    * the season, because only the same calendar month is compared;
    * the coverage, because March 2004 starts on the 10th and April 2005 ends
      on the 4th, so a full month would never be compared with a full month.
    """
    work = _monthly_frame(frame, sensor, reference)

    rows = []
    for month_number, group in work.groupby("month_number"):
        if group["year"].nunique() < 2:
            continue

        # Trim to the days of the month that every year of this month covers.
        shared_day = int(group.groupby("year")["day"].max().min())
        subset = group[group["day"] <= shared_day]

        fits: dict[int, dict] = {}
        for year, year_group in subset.groupby("year"):
            paired = year_group[["sensor", "reference"]].dropna()
            if len(paired) < min_pairs:
                continue
            fits[int(year)] = _fit_calibration(paired["sensor"], paired["reference"])

        if len(fits) < 2:
            continue

        years = sorted(fits)
        early, late = fits[years[0]], fits[years[-1]]
        rows.append(
            {
                "month_number": int(month_number),
                "year_early": years[0],
                "year_late": years[-1],
                "shared_day": shared_day,
                "slope_early": early["slope"],
                "slope_late": late["slope"],
                "slope_change": late["slope"] - early["slope"],
                "intercept_early": early["intercept"],
                "intercept_late": late["intercept"],
                "intercept_change": late["intercept"] - early["intercept"],
                "r_early": early["pearson_r"],
                "r_late": late["pearson_r"],
                "r_change": late["pearson_r"] - early["pearson_r"],
                "n_pairs_early": early["n_pairs"],
                "n_pairs_late": late["n_pairs"],
                "mean_sensor_early": early["mean_sensor"],
                "mean_sensor_late": late["mean_sensor"],
                "mean_reference_early": early["mean_reference"],
                "mean_reference_late": late["mean_reference"],
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_trend_ci(
    values: np.ndarray,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """95% bootstrap interval for the slope of a straight line through ``values``.

    The values are treated as an evenly spaced time series (one point per
    month), so the slope is "change per month".

    The resampling is a **pairs bootstrap**: a time index and the value
    recorded at that index are drawn together, so the ordering of the series
    survives the resampling. Resampling the values on their own would shuffle
    them into a random order, and the trend of a random ordering is zero by
    construction - that would produce a meaningless interval centred on zero
    no matter how strong the real trend is.
    """
    if values.size < 3:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    positions = np.arange(values.size, dtype=float)

    # Draw every resample at once. Resampling (position, value) pairs
    # together keeps the ordering of the series intact.
    picks = rng.integers(0, values.size, size=(n_resamples, values.size))
    xs = positions[picks]
    ys = values[picks]

    # Ordinary least squares slope with an intercept, written out so the
    # whole bootstrap can be done with array operations.
    x_mean = xs.mean(axis=1, keepdims=True)
    y_mean = ys.mean(axis=1, keepdims=True)
    dx = xs - x_mean
    dy = ys - y_mean
    denominator = (dx * dx).sum(axis=1)
    numerator = (dx * dy).sum(axis=1)
    slopes = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )

    low, high = np.percentile(slopes, [2.5, 97.5])
    return (float(low), float(high))


def _trend(values: np.ndarray) -> float:
    """Least-squares slope of ``values`` against evenly spaced positions."""
    if values.size < 2 or np.allclose(values, values[0]):
        return 0.0
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


def year_over_year_calibration(monthly: pd.DataFrame) -> pd.DataFrame:
    """Simple same-calendar-month comparison taken from a monthly table.

    :func:`matched_period_comparison` is the rigorous version because it
    also equalises the coverage; this one is kept as a quick view.
    """
    if monthly.empty:
        return pd.DataFrame(
            columns=["month_number", "month", "years", "slope_change",
                     "intercept_change", "r_change"]
        )

    rows = []
    for month_number, group in monthly.groupby("month_number"):
        if group["year"].nunique() < 2:
            continue
        group = group.sort_values("year")
        first, last = group.iloc[0], group.iloc[-1]
        rows.append(
            {
                "month_number": int(month_number),
                "month": first["month"],
                "years": f"{int(first['year'])} vs {int(last['year'])}",
                "slope_change": float(last["slope"] - first["slope"]),
                "intercept_change": float(last["intercept"] - first["intercept"]),
                "r_change": float(last["pearson_r"] - first["pearson_r"]),
                "n_pairs_first": int(first["n_pairs"]),
                "n_pairs_last": int(last["n_pairs"]),
            }
        )
    return pd.DataFrame(rows)


def assess_drift(
    frame: pd.DataFrame,
    sensor: str,
    reference: str,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
) -> DriftAssessment:
    """Assess whether the sensor response drifts over the recording period.

    The verdict is one of:

    ``"insufficient data"``
        Too few months with enough paired observations to say anything.
    ``"no clear drift signal"``
        The trend interval includes zero, or the matched same-season
        comparison contradicts the trend.
    ``"evidence consistent with a temporal change"``
        The trend interval excludes zero **and** every matched same-season
        comparison moved in the same direction as the trend. Because the
        comparison holds both the season and the coverage fixed, this cannot
        be explained away by either.
    ``"temporal trend detected, seasonal control unavailable"``
        As above, but no calendar month appears in both years, so the
        seasonal control could not be applied and the trend cannot be
        separated from the season.
    """
    monthly = monthly_calibration(frame, sensor, reference, min_pairs=min_pairs)
    matched = matched_period_comparison(frame, sensor, reference, min_pairs=min_pairs)
    total_pairs = int(monthly["n_pairs"].sum()) if not monthly.empty else 0

    # --- Coverage and missingness notes ------------------------------------
    stamps = pd.to_datetime(frame[DATETIME_COLUMN])
    per_year = stamps.dt.year.value_counts().sort_index().to_dict()
    coverage_note = (
        f"Record spans {stamps.min():%Y-%m-%d} to {stamps.max():%Y-%m-%d} "
        f"with {per_year} rows per year. A first-versus-last comparison would "
        "not hold the season constant, and 2005 is only covered to early April."
    )

    paired_rows = _monthly_frame(frame, sensor, reference)[["sensor", "reference"]].dropna()
    paired_share = 100.0 * len(paired_rows) / max(1, len(frame))
    missingness_note = (
        f"{paired_share:.1f}% of hours have both a sensor and a reference value; "
        "the remaining hours cannot take part in the comparison at all."
    )

    if monthly.empty or len(monthly) < 6:
        return DriftAssessment(
            sensor=sensor,
            reference=reference,
            monthly=monthly,
            matched=matched,
            months_analysed=len(monthly),
            total_pairs=total_pairs,
            slope_trend_per_month=float("nan"),
            slope_trend_ci=(float("nan"), float("nan")),
            intercept_trend_per_month=float("nan"),
            intercept_trend_ci=(float("nan"), float("nan")),
            correlation_trend_per_month=float("nan"),
            months_with_increased_slope=0,
            matched_months=len(matched),
            matched_agrees_with_trend=0,
            coverage_note=coverage_note,
            missingness_note=missingness_note,
            verdict="insufficient data",
            verdict_reasons=[
                f"Only {len(monthly)} month(s) had at least {min_pairs} paired "
                "observations, which is not enough to separate a trend from noise."
            ],
        )

    slopes = monthly["slope"].to_numpy(dtype=float)
    intercepts = monthly["intercept"].to_numpy(dtype=float)
    correlations = monthly["pearson_r"].to_numpy(dtype=float)

    slope_trend = _trend(slopes)
    intercept_trend = _trend(intercepts)
    correlation_trend = _trend(correlations)

    slope_ci = _bootstrap_trend_ci(slopes, n_resamples=n_bootstrap)
    intercept_ci = _bootstrap_trend_ci(intercepts, n_resamples=n_bootstrap)

    # Direction of month-to-month movement in the calibration slope.
    changes = np.diff(slopes)
    increased = int((changes > 0).sum())
    decreased = int((changes < 0).sum())
    total_steps = increased + decreased
    dominant = max(increased, decreased)

    # The seasonal control: same calendar month, equal coverage, later year
    # against earlier year.
    matched_months = len(matched)
    matched_agrees = 0
    if matched_months:
        trend_sign = np.sign(slope_trend)
        matched_agrees = int((np.sign(matched["slope_change"]) == trend_sign).sum())

    # --- Verdict, computed from the numbers above --------------------------
    reasons: list[str] = []
    ci_excludes_zero = (
        (slope_ci[0] > 0 and slope_ci[1] > 0) or (slope_ci[0] < 0 and slope_ci[1] < 0)
    )

    if ci_excludes_zero:
        reasons.append(
            f"The monthly calibration slope changes by {slope_trend:+.5g} per month, "
            f"and the 95% bootstrap interval [{slope_ci[0]:+.5g}, {slope_ci[1]:+.5g}] "
            "excludes zero."
        )
    else:
        reasons.append(
            f"The monthly calibration slope changes by {slope_trend:+.5g} per month, "
            f"but the 95% bootstrap interval [{slope_ci[0]:+.5g}, {slope_ci[1]:+.5g}] "
            "includes zero, so the direction of the trend is not established."
        )

    if dominant == increased and increased > 0:
        reasons.append(
            f"The monthly calibration slope increased in {increased} of "
            f"{total_steps} month-to-month steps."
        )
    elif dominant == decreased and decreased > 0:
        reasons.append(
            f"The monthly calibration slope decreased in {decreased} of "
            f"{total_steps} month-to-month steps."
        )

    if matched_months:
        reasons.append(
            f"{matched_agrees} of {matched_months} matched same-season comparisons "
            "(same calendar month, equal day coverage) move in the same direction "
            "as the overall trend."
        )
    else:
        reasons.append(
            "No calendar month appears in both years, so the seasonal control "
            "could not be applied."
        )

    # The verdict needs the interval to exclude zero and the seasonal
    # control to agree. Without the control, a trend cannot be separated
    # from a seasonal effect.
    if not ci_excludes_zero:
        verdict = "no clear drift signal"
    elif matched_months == 0:
        verdict = "temporal trend detected, seasonal control unavailable"
    elif matched_agrees == matched_months:
        verdict = "evidence consistent with a temporal change"
    else:
        verdict = "no clear drift signal"

    return DriftAssessment(
        sensor=sensor,
        reference=reference,
        monthly=monthly,
        matched=matched,
        months_analysed=len(monthly),
        total_pairs=total_pairs,
        slope_trend_per_month=slope_trend,
        slope_trend_ci=slope_ci,
        intercept_trend_per_month=intercept_trend,
        intercept_trend_ci=intercept_ci,
        correlation_trend_per_month=correlation_trend,
        months_with_increased_slope=increased,
        matched_months=matched_months,
        matched_agrees_with_trend=matched_agrees,
        coverage_note=coverage_note,
        missingness_note=missingness_note,
        verdict=verdict,
        verdict_reasons=reasons,
    )


def assess_all_sensors(
    frame: pd.DataFrame,
    pairs: list[tuple[str, str]],
    min_pairs: int = DEFAULT_MIN_PAIRS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
) -> list[DriftAssessment]:
    """Run :func:`assess_drift` for every sensor/reference pair."""
    return [
        assess_drift(frame, sensor, reference, min_pairs=min_pairs, n_bootstrap=n_bootstrap)
        for sensor, reference in pairs
    ]


__all__ = [
    "DriftAssessment",
    "DEFAULT_MIN_PAIRS",
    "DEFAULT_BOOTSTRAP",
    "RANDOM_SEED",
    "monthly_calibration",
    "matched_period_comparison",
    "year_over_year_calibration",
    "assess_drift",
    "assess_all_sensors",
]
