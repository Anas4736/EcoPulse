"""Retrospective regression of benzene, without leaking the answer into the model.

The question
------------
The UCI Air Quality dataset measures benzene with a certified reference
analyser (``C6H6(GT)``). That analyser is the ground truth, but it is also
expensive and only produces a value for a subset of hours. So the practical
question is:

    *Using only signals that are genuinely available at prediction time, how
    well can benzene be estimated, and does that relationship hold up over
    time?*

The three rules that make the answer trustworthy
------------------------------------------------
1. **No target leakage.**  The target itself is never a feature, and neither
   is anything computed from it.  This is enforced by
   :func:`build_ml_frame` and verified by the test suite.
2. **No proxy for the answer.**  ``PT08.S2(NMHC)`` is a cheap titania sensor
   whose response correlates at r = 0.982 with the benzene reference.  Using
   it would be legal arithmetic but useless science: the model would just be
   reading the answer off another instrument.  It is therefore excluded from
   the headline model.  :func:`run_leakage_diagnostic` deliberately re-runs
   the model *with* that column so the cost of the exclusion is a measured
   number rather than an assertion.
3. **A chronological split.**  The record runs from March 2004 to April 2005.
   Splitting it randomly would put neighbouring hours on both sides of the
   split and let the model interpolate its way through the test set.  The
   training set is the *earlier* part of the record and the test set is the
   *later* part, so the model is scored on the future.  Cross-validation uses
   :class:`~sklearn.model_selection.TimeSeriesSplit`, which is chronological
   too.

Missing values
--------------
Several variables are missing in long blocks because the monitoring device
was offline.  Rows whose *target* is missing cannot be trained or scored, so
they are dropped.  Missing *features* are handled inside the pipeline by a
median imputer that is fitted on the training fold only - never on the test
fold - together with binary indicators that tell the model which readings
were imputed.

Which model is the headline
---------------------------
The headline model is the **sensor-only** tier (:data:`PRIMARY_FEATURE_SET`):
the weather measurements, the four non-hydrocarbon metal-oxide sensor
responses, the calendar features and the missing-count taken over those same
variables.  Every one of those is
available to a cheap monitoring station, which is the scenario this project is
about, so the headline number is the one a low-cost station could actually
reproduce.

The ``full`` tier (:data:`REFERENCE_ASSISTED_FEATURE_SET`) additionally uses
``CO(GT)``, ``NOx(GT)`` and ``NO2(GT)`` - measurements from the co-located
certified reference analysers.  It answers a different question: *how much does
adding an expensive analyser to a cheap station actually buy?*  It is therefore
reported as a clearly labelled reference-assisted comparison and is never
presented as a low-cost sensor model.

The headline number
-------------------
Only :func:`run_benzene_benchmark` produces a reported metric, and it produces
it by actually fitting the model and scoring it on held-out data.  Nothing in
this module contains a hard-coded accuracy figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATETIME_COLUMN, MEASURED_COLUMNS, PRIMARY_TARGET
from .features import add_time_features

#: The column being modelled.
TARGET: str = PRIMARY_TARGET

#: Fixed seed so every run of the project gives the same numbers.
RANDOM_SEED: int = 20260101

#: Number of trees used by the Random Forest in the reported benchmark.
DEFAULT_N_ESTIMATORS: int = 300

#: The hydrocarbon columns.  ``NMHC(GT)`` is excluded by the project brief
#: (about 90% of its values are missing) and ``PT08.S2(NMHC)`` is excluded
#: because it acts as a proxy measurement of the target.
NMHC_REFERENCE = "NMHC(GT)"
NMHC_SENSOR = "PT08.S2(NMHC)"

#: Every column that is barred from the headline feature set, and why.
EXCLUDED_COLUMNS: dict[str, str] = {
    TARGET: "the modelling target itself",
    NMHC_REFERENCE: (
        "excluded by the project brief, and about 90% of its values are missing"
    ),
    NMHC_SENSOR: (
        "a near-perfect proxy for the target (r = 0.982 with C6H6(GT)); using it "
        "would measure the answer with another instrument instead of predicting it"
    ),
}

#: Weather measurements.
WEATHER_FEATURES: list[str] = ["T", "RH", "AH"]

#: The metal-oxide sensor responses that are *not* hydrocarbon sensors.
SENSOR_FEATURES: list[str] = [
    "PT08.S1(CO)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
]

#: The other reference analysers, measured on the same train at the same time.
OTHER_REFERENCE_FEATURES: list[str] = ["CO(GT)", "NOx(GT)", "NO2(GT)"]

#: Calendar features.  Hour and day-of-year are encoded cyclically so that
#: 23:00 and 00:00 are treated as neighbours rather than opposites.
TIME_FEATURES: list[str] = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_daytime",
]

#: A count of how many of the variables a tier uses are missing in each row.
#: It is a useful signal because the device outages are not spread evenly
#: through the year.  It is counted over the *tier's own* measured columns,
#: so a tier never sees the availability of an instrument it does not use, and
#: it never counts the target or the excluded hydrocarbon columns.
MISSING_COUNT_FEATURE = "missing_count"

#: The feature tiers, from the least to the most informative.  "sensors" is the
#: headline tier: weather, the four non-hydrocarbon metal-oxide sensors, the
#: calendar features and the missing-count over those same variables.
#: Everything it uses is available to a low-cost monitoring station.  "full"
#: adds the co-located reference analysers and is therefore a
#: *reference-assisted* comparison, not a low-cost sensor model.
FEATURE_SETS: dict[str, list[str]] = {
    "weather": WEATHER_FEATURES + TIME_FEATURES + [MISSING_COUNT_FEATURE],
    "sensors": (
        WEATHER_FEATURES + SENSOR_FEATURES + TIME_FEATURES + [MISSING_COUNT_FEATURE]
    ),
    "full": (
        WEATHER_FEATURES
        + SENSOR_FEATURES
        + OTHER_REFERENCE_FEATURES
        + TIME_FEATURES
        + [MISSING_COUNT_FEATURE]
    ),
}

#: The tier used for the reported benchmark: the **sensor-only** model.  Every
#: predictor in it is a cheap weather measurement or a metal-oxide sensor
#: response, so it describes what a low-cost station could do.
PRIMARY_FEATURE_SET = "sensors"

#: The **reference-assisted** tier.  It is the sensor-only tier plus the
#: ``CO(GT)``, ``NOx(GT)`` and ``NO2(GT)`` reference-analyser measurements, so
#: it needs the expensive co-located instruments.  It is reported separately and
#: labelled as such; it is never the headline model.
REFERENCE_ASSISTED_FEATURE_SET = "full"

#: Short labels for the tiers, used wherever a tier is named on screen.
FEATURE_SET_LABELS: dict[str, str] = {
    "weather": "Weather only",
    "sensors": "Sensor-only (headline)",
    "full": "Reference-assisted (comparison)",
}

#: One-line statements of what each tier is allowed to use.  These keep the
#: distinction between the low-cost sensor model and the reference-assisted
#: comparison visible everywhere the tiers are named.
FEATURE_SET_DESCRIPTIONS: dict[str, str] = {
    "weather": (
        "Temperature, humidity and the calendar, plus a count of how many of "
        "those readings are missing. No sensor and no analyser."
    ),
    "sensors": (
        "Weather, the four non-hydrocarbon metal-oxide sensors and the calendar, "
        "plus a count of how many of those readings are missing. Everything here "
        "is available to a low-cost station."
    ),
    "full": (
        "The sensor-only predictors plus CO(GT), NOx(GT) and NO2(GT) from the "
        "certified reference analysers, so it needs the expensive instruments."
    ),
}

#: The models that are compared, in the order they are reported.
MODEL_NAMES: tuple[str, ...] = ("mean_baseline", "linear_baseline", "random_forest")

#: Human-readable labels for the dashboard.
MODEL_LABELS: dict[str, str] = {
    "mean_baseline": "Mean baseline (predict the training average)",
    "linear_baseline": "Linear regression",
    "random_forest": "Random Forest",
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ModellingData:
    """The modelling table: features, target and timestamps, all aligned."""

    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def is_chronological(self) -> bool:
        """Are the rows ordered by time?"""
        return bool(self.timestamps.is_monotonic_increasing)


@dataclass
class Split:
    """A chronological split into an earlier training part and a later test part."""

    train: ModellingData
    test: ModellingData
    cutoff: pd.Timestamp
    test_size: float

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train.timestamps.iloc[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test.timestamps.iloc[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test.timestamps.iloc[-1]


@dataclass
class ModelResult:
    """The measured performance of one model on one feature set."""

    name: str
    feature_set: str
    r2: float
    mae: float
    rmse: float
    n_train: int
    n_test: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    cv_r2_mean: float
    cv_r2_std: float
    cv_mae_mean: float
    cv_rmse_mean: float
    feature_names: list[str] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)
    actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    test_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.name, self.name)

    def summary(self) -> dict:
        """A flat row describing this result, for a table."""
        return {
            "model": self.label,
            "feature_set": self.feature_set,
            "r2": self.r2,
            "mae": self.mae,
            "rmse": self.rmse,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "cv_r2_mean": self.cv_r2_mean,
            "cv_r2_std": self.cv_r2_std,
        }


@dataclass
class BenchmarkResult:
    """The full set of results from one benchmark run."""

    results: list[ModelResult]
    feature_sets: dict[str, list[str]]
    n_rows: int
    n_dropped: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    target: str = TARGET

    def get(self, name: str, feature_set: str = PRIMARY_FEATURE_SET) -> ModelResult:
        """Look up one result by model name and feature set."""
        for result in self.results:
            if result.name == name and result.feature_set == feature_set:
                return result
        raise KeyError(f"No result for model {name!r} on feature set {feature_set!r}")

    def summary_frame(self) -> pd.DataFrame:
        """Every result as a table."""
        if not self.results:
            return pd.DataFrame(
                columns=["model", "feature_set", "r2", "mae", "rmse", "n_train", "n_test"]
            )
        return pd.DataFrame([result.summary() for result in self.results])


@dataclass
class LeakageReport:
    """What the excluded proxy column would have done to the headline number."""

    proxy_column: str
    proxy_target_correlation: float
    primary_r2: float
    proxy_r2: float
    primary_rmse: float
    proxy_rmse: float
    excluded_columns: dict[str, str] = field(default_factory=dict)

    @property
    def r2_inflation(self) -> float:
        """How much R2 the excluded column would have added."""
        return self.proxy_r2 - self.primary_r2

    @property
    def rmse_reduction_pct(self) -> float:
        """How much smaller the RMSE would have been, as a percentage."""
        if self.primary_rmse == 0:
            return 0.0
        return 100.0 * (self.primary_rmse - self.proxy_rmse) / self.primary_rmse


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true, y_pred) -> dict:
    """Return R2, MAE and RMSE for a set of predictions.

    Parameters
    ----------
    y_true
        Observed target values.
    y_pred
        Predicted target values, in the same order.

    Returns
    -------
    dict
        ``{"r2": float, "mae": float, "rmse": float, "n": int}``

    Notes
    -----
    RMSE is computed directly as the square root of the mean squared error
    rather than through a helper whose keyword arguments have changed between
    scikit-learn releases.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty set of predictions")

    errors = y_pred - y_true
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = r2_score(y_true, y_pred) if np.ptp(y_true) > 0 else float("nan")
    return {
        "r2": float(r2),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "n": int(y_true.size),
    }


# ---------------------------------------------------------------------------
# Building the modelling table
# ---------------------------------------------------------------------------


def build_ml_frame(
    frame: pd.DataFrame,
    feature_set: str = PRIMARY_FEATURE_SET,
    target: str = TARGET,
    allow_hydrocarbon_proxy: bool = False,
) -> ModellingData:
    """Build the modelling table, keeping the target strictly out of it.

    Parameters
    ----------
    frame
        Cleaned dataset containing ``datetime``, the target and the features.
    feature_set
        Which tier from :data:`FEATURE_SETS` to use.
    target
        The column to model.
    allow_hydrocarbon_proxy
        Add ``PT08.S2(NMHC)`` to the features.  This exists **only** for
        :func:`run_leakage_diagnostic`; the headline model never sets it.

    Returns
    -------
    ModellingData
        Features, target and timestamps, sorted by time, with rows whose
        target is missing removed.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}. "
            f"Choose one of: {sorted(FEATURE_SETS)}"
        )
    if target not in frame.columns:
        raise ValueError(f"Frame is missing the target column {target!r}")

    features = list(FEATURE_SETS[feature_set])
    if allow_hydrocarbon_proxy:
        features.append(NMHC_SENSOR)

    # Measured columns must exist in the input frame.
    measured_missing = [
        column
        for column in features
        if column in MEASURED_COLUMNS and column not in frame.columns
    ]
    if measured_missing:
        raise ValueError(f"Frame is missing these columns: {measured_missing}")

    work = add_time_features(frame)

    # How many of the variables *this tier uses* are missing in each row.  The
    # count is taken over the tier's own measured columns, so a tier that
    # claims to be sensor-only never sees the availability of an instrument it
    # does not own, and the excluded hydrocarbon columns are never counted
    # either.  The feature therefore cannot reveal whether the target was
    # missing, and cannot leak the proxy's presence.
    measured = [
        column
        for column in features
        if column in MEASURED_COLUMNS and column != target
    ]
    work[MISSING_COUNT_FEATURE] = work[measured].isna().sum(axis=1).astype("float64")

    # The calendar features and the missing count are derived here, so they
    # only exist once the frame has been prepared.
    derived_missing = [column for column in features if column not in work.columns]
    if derived_missing:
        raise ValueError(f"Frame is missing these columns: {derived_missing}")

    X = work[features].astype("float64")
    y = pd.to_numeric(work[target], errors="coerce")
    stamps = pd.to_datetime(work[DATETIME_COLUMN])

    keep = y.notna() & stamps.notna()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    stamps = stamps.loc[keep].reset_index(drop=True)

    # Sort by time. A stable sort keeps the original order for equal stamps.
    order = np.argsort(stamps.to_numpy(), kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    stamps = stamps.iloc[order].reset_index(drop=True)

    return ModellingData(X=X, y=y, timestamps=stamps, feature_names=features)


def chronological_split(data: ModellingData, test_size: float = 0.2) -> Split:
    """Split the data so the training set is earlier than the test set.

    Parameters
    ----------
    data
        The modelling table.
    test_size
        Fraction of rows held out for testing, taken from the **end** of the
        record.

    Returns
    -------
    Split
        The two parts plus the timestamp that separates them.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size!r}")

    n = len(data)
    cutoff_index = int(n * (1.0 - test_size))
    if cutoff_index < 2 or n - cutoff_index < 1:
        raise ValueError(
            f"Cannot split {n} rows into a chronological train/test split "
            f"with test_size={test_size!r}"
        )

    def part(sl: slice) -> ModellingData:
        return ModellingData(
            X=data.X.iloc[sl].reset_index(drop=True),
            y=data.y.iloc[sl].reset_index(drop=True),
            timestamps=data.timestamps.iloc[sl].reset_index(drop=True),
            feature_names=list(data.feature_names),
        )

    return Split(
        train=part(slice(0, cutoff_index)),
        test=part(slice(cutoff_index, n)),
        cutoff=data.timestamps.iloc[cutoff_index],
        test_size=test_size,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_model(
    name: str,
    random_state: int = RANDOM_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> Pipeline:
    """Return the full pipeline for one model.

    The pipeline is always *impute -> scale -> estimate*.  The imputer is
    fitted inside the pipeline, which means it is fitted on the training fold
    only whenever the pipeline is fitted - that is what keeps the missing
    values from leaking information backwards in time.
    """
    if name == "mean_baseline":
        estimator = DummyRegressor(strategy="mean")
    elif name == "linear_baseline":
        estimator = LinearRegression()
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unknown model {name!r}. Choose one of: {list(MODEL_NAMES)}"
        )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _rows(data, index):
    """Select rows from a DataFrame, a Series or a plain array."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[index]
    return np.asarray(data)[index]


def cross_validate_chronologically(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Score a pipeline with expanding-window, chronological cross-validation.

    :class:`~sklearn.model_selection.TimeSeriesSplit` is used instead of
    ``KFold`` so that every validation block is *later* than the block it is
    scored against.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []
    for train_index, test_index in splitter.split(X):
        fold = clone(pipeline)
        fold.fit(_rows(X, train_index), _rows(y, train_index))
        predictions = fold.predict(_rows(X, test_index))
        scores = regression_metrics(_rows(y, test_index), predictions)
        scores.update(
            {
                "train_start": int(train_index.min()),
                "train_end": int(train_index.max()),
                "test_start": int(test_index.min()),
                "test_end": int(test_index.max()),
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
            }
        )
        rows.append(scores)

    return {
        "r2_mean": float(np.mean([row["r2"] for row in rows])),
        "r2_std": float(np.std([row["r2"] for row in rows])),
        "mae_mean": float(np.mean([row["mae"] for row in rows])),
        "rmse_mean": float(np.mean([row["rmse"] for row in rows])),
        "folds": rows,
    }


def evaluate_model(
    name: str,
    data: ModellingData,
    feature_set: str = PRIMARY_FEATURE_SET,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> ModelResult:
    """Fit one model and score it on the later part of the record."""
    split = chronological_split(data, test_size=test_size)
    pipeline = build_model(name, random_state=random_state, n_estimators=n_estimators)

    pipeline.fit(split.train.X, split.train.y)
    predictions = pipeline.predict(split.test.X)
    metrics = regression_metrics(split.test.y.to_numpy(), predictions)

    cross = cross_validate_chronologically(
        pipeline, split.train.X, split.train.y, n_splits=cv_splits
    )

    importances: dict[str, float] = {}
    coefficients: dict[str, float] = {}

    transformed_names = _transformed_feature_names(pipeline)
    fitted = pipeline.named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        for name_, value in zip(transformed_names, fitted.feature_importances_):
            importances[name_] = float(value)
    if hasattr(fitted, "coef_"):
        for name_, value in zip(transformed_names, np.ravel(fitted.coef_)):
            coefficients[name_] = float(value)

    return ModelResult(
        name=name,
        feature_set=feature_set,
        r2=metrics["r2"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        n_train=len(split.train),
        n_test=len(split.test),
        train_end=split.train_end,
        test_start=split.test_start,
        test_end=split.test_end,
        cv_r2_mean=cross["r2_mean"],
        cv_r2_std=cross["r2_std"],
        cv_mae_mean=cross["mae_mean"],
        cv_rmse_mean=cross["rmse_mean"],
        feature_names=list(data.feature_names),
        feature_importances=importances,
        coefficients=coefficients,
        actuals=split.test.y.to_numpy(),
        predictions=predictions,
        test_timestamps=split.test.timestamps.to_numpy(),
    )


def _transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns the final estimator actually sees.

    The imputer appends one binary indicator per feature that was missing in
    the training data, so the model's own feature list is longer than the
    column list that was handed in.
    """
    try:
        return list(pipeline[:-1].get_feature_names_out())
    except (AttributeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


def run_benzene_benchmark(
    frame: pd.DataFrame,
    feature_sets: tuple[str, ...] = (PRIMARY_FEATURE_SET,),
    models: tuple[str, ...] = MODEL_NAMES,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> BenchmarkResult:
    """Fit and score every model on every feature tier.

    This is the only function that produces the numbers reported on the
    benzene page.  Every metric in the result comes from a model that was
    actually fitted and scored on held-out data.
    """
    results: list[ModelResult] = []
    used_features: dict[str, list[str]] = {}
    n_rows = 0
    n_dropped = 0
    train_end = test_start = test_end = None

    for feature_set in feature_sets:
        data = build_ml_frame(frame, feature_set=feature_set)
        used_features[feature_set] = list(data.feature_names)
        n_dropped += len(frame) - len(data)

        for name in models:
            result = evaluate_model(
                name,
                data,
                feature_set=feature_set,
                test_size=test_size,
                cv_splits=cv_splits,
                random_state=random_state,
                n_estimators=n_estimators,
            )
            results.append(result)
            if feature_set == PRIMARY_FEATURE_SET and name == models[0]:
                n_rows = len(data)
                train_end = result.train_end
                test_start = result.test_start
                test_end = result.test_end

    return BenchmarkResult(
        results=results,
        feature_sets=used_features,
        n_rows=n_rows,
        n_dropped=n_dropped,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def run_leakage_diagnostic(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> LeakageReport:
    """Measure what the excluded proxy column would have done to the score.

    The headline model leaves out ``PT08.S2(NMHC)`` because it correlates at
    r = 0.982 with the target.  Rather than asserting that excluding it is
    the right call, this function fits the Random Forest both ways and reports
    the difference.

    The comparison is made against the headline **sensor-only** model
    (:data:`PRIMARY_FEATURE_SET`), so the number here is the cost of the
    exclusion for the low-cost sensor scenario and not for the
    reference-assisted one.
    """
    primary = build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
    with_proxy = build_ml_frame(
        frame, feature_set=PRIMARY_FEATURE_SET, allow_hydrocarbon_proxy=True
    )

    primary_result = evaluate_model(
        "random_forest",
        primary,
        feature_set=PRIMARY_FEATURE_SET,
        test_size=test_size,
        cv_splits=cv_splits,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    proxy_result = evaluate_model(
        "random_forest",
        with_proxy,
        feature_set=f"{PRIMARY_FEATURE_SET}+proxy",
        test_size=test_size,
        cv_splits=cv_splits,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    paired = frame[[TARGET, NMHC_SENSOR]].apply(pd.to_numeric, errors="coerce").dropna()
    correlation = float(paired[TARGET].corr(paired[NMHC_SENSOR]))

    return LeakageReport(
        proxy_column=NMHC_SENSOR,
        proxy_target_correlation=correlation,
        primary_r2=primary_result.r2,
        proxy_r2=proxy_result.r2,
        primary_rmse=primary_result.rmse,
        proxy_rmse=proxy_result.rmse,
        excluded_columns=dict(EXCLUDED_COLUMNS),
    )


def feature_importance_table(result: ModelResult, top: int | None = None) -> pd.DataFrame:
    """Turn a model's feature importances into a sorted table."""
    if not result.feature_importances:
        return pd.DataFrame(columns=["feature", "importance"])
    table = pd.DataFrame(
        sorted(result.feature_importances.items(), key=lambda item: item[1], reverse=True),
        columns=["feature", "importance"],
    )
    if top is not None:
        table = table.head(top).reset_index(drop=True)
    return table


__all__ = [
    "TARGET",
    "RANDOM_SEED",
    "DEFAULT_N_ESTIMATORS",
    "NMHC_REFERENCE",
    "NMHC_SENSOR",
    "EXCLUDED_COLUMNS",
    "WEATHER_FEATURES",
    "SENSOR_FEATURES",
    "OTHER_REFERENCE_FEATURES",
    "TIME_FEATURES",
    "MISSING_COUNT_FEATURE",
    "FEATURE_SETS",
    "PRIMARY_FEATURE_SET",
    "REFERENCE_ASSISTED_FEATURE_SET",
    "FEATURE_SET_LABELS",
    "FEATURE_SET_DESCRIPTIONS",
    "MODEL_NAMES",
    "MODEL_LABELS",
    "ModellingData",
    "Split",
    "ModelResult",
    "BenchmarkResult",
    "LeakageReport",
    "regression_metrics",
    "build_ml_frame",
    "chronological_split",
    "build_model",
    "cross_validate_chronologically",
    "evaluate_model",
    "run_benzene_benchmark",
    "run_leakage_diagnostic",
    "feature_importance_table",
]
