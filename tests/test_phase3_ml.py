"""Phase 3 tests: the machine-learning layer.

Five properties are checked here, and they are the properties that decide
whether the numbers on the benzene page mean anything:

1. **Metric calculation** - the reported R2, MAE and RMSE are computed
   correctly from the predictions that were actually made.
2. **Chronology** - the training set is earlier than the test set, and the
   cross-validation folds respect time as well.
3. **Missing values** - rows with a missing target are dropped, missing
   features are imputed from the *training* data only, and the model still
   runs.
4. **Feature leakage** - the target, ``NMHC(GT)`` and the near-perfect
   ``PT08.S2(NMHC)`` proxy never reach the model, and the feature matrix
   provably does not change when the target is scrambled.
5. **Feature sets** - the headline model is the *sensor-only* tier, the
   reference-assisted tier is kept apart from it and labelled as such, and
   the headline features do not depend on any reference analyser.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from ecopulse.config import MEASURED_COLUMNS
from ecopulse.ml import (
    EXCLUDED_COLUMNS,
    FEATURE_SET_DESCRIPTIONS,
    FEATURE_SET_LABELS,
    FEATURE_SETS,
    MODEL_NAMES,
    NMHC_REFERENCE,
    NMHC_SENSOR,
    MISSING_COUNT_FEATURE,
    OTHER_REFERENCE_FEATURES,
    PRIMARY_FEATURE_SET,
    REFERENCE_ASSISTED_FEATURE_SET,
    SENSOR_FEATURES,
    TARGET,
    TIME_FEATURES,
    WEATHER_FEATURES,
    build_ml_frame,
    build_model,
    chronological_split,
    cross_validate_chronologically,
    evaluate_model,
    feature_importance_table,
    regression_metrics,
    run_benzene_benchmark,
    run_leakage_diagnostic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_ml_frame(n=200, start="2004-01-01", seed=0):
    """A synthetic frame with every column the modelling table expects."""
    rng = np.random.default_rng(seed)
    data = {
        "datetime": pd.date_range(start, periods=n, freq="h"),
        "T": 10 + rng.normal(0, 3, n),
        "RH": 50 + rng.normal(0, 8, n),
        "AH": 0.8 + rng.normal(0, 0.05, n),
        "PT08.S1(CO)": 1000 + rng.normal(0, 100, n),
        "PT08.S3(NOx)": 1000 + rng.normal(0, 100, n),
        "PT08.S4(NO2)": 1000 + rng.normal(0, 100, n),
        "PT08.S5(O3)": 1000 + rng.normal(0, 100, n),
        "CO(GT)": 2.0 + rng.normal(0, 0.5, n),
        "NOx(GT)": 150 + rng.normal(0, 20, n),
        "NO2(GT)": 90 + rng.normal(0, 15, n),
        "NMHC(GT)": 2.0 + rng.normal(0, 0.4, n),
        "PT08.S2(NMHC)": 900 + rng.normal(0, 80, n),
    }
    data[TARGET] = 2.5 + 0.3 * data["CO(GT)"] + rng.normal(0, 0.2, n)
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def real_benchmark(cleaned_result):
    """The reported benchmark, fitted once on the real data."""
    return run_benzene_benchmark(
        cleaned_result.frame, feature_sets=(PRIMARY_FEATURE_SET,), cv_splits=5
    )


@pytest.fixture(scope="session")
def real_leakage_report(cleaned_result):
    """The measured cost of allowing the proxy column in."""
    return run_leakage_diagnostic(cleaned_result.frame, cv_splits=5)


@pytest.fixture(scope="session")
def real_tier_benchmark(cleaned_result):
    """The Random Forest fitted on each of the three feature tiers."""
    return run_benzene_benchmark(
        cleaned_result.frame,
        feature_sets=("weather", "sensors", REFERENCE_ASSISTED_FEATURE_SET),
        models=("random_forest",),
        cv_splits=5,
    )


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------
class TestRegressionMetrics:
    def test_perfect_prediction_scores_perfectly(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = regression_metrics(y, y)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["n"] == 4

    def test_rmse_is_the_root_mean_squared_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 4.0, 6.0])
        expected = np.sqrt(np.mean((y_pred - y_true) ** 2))
        assert regression_metrics(y_true, y_pred)["rmse"] == pytest.approx(expected)

    def test_mae_is_the_mean_absolute_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 4.0, 6.0])
        expected = np.mean(np.abs(y_pred - y_true))
        assert regression_metrics(y_true, y_pred)["mae"] == pytest.approx(expected)

    def test_r2_agrees_with_scikit_learn(self):
        rng = np.random.default_rng(3)
        y_true = rng.normal(10, 2, 200)
        y_pred = y_true + rng.normal(0, 1, 200)
        assert regression_metrics(y_true, y_pred)["r2"] == pytest.approx(
            r2_score(y_true, y_pred)
        )

    def test_r2_is_negative_for_a_bad_model(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        assert regression_metrics(y_true, y_pred)["r2"] < 0

    def test_constant_actual_gives_an_undefined_r2_not_a_crash(self):
        y_true = np.array([5.0, 5.0, 5.0])
        metrics = regression_metrics(y_true, y_true)
        assert np.isnan(metrics["r2"])
        assert metrics["mae"] == pytest.approx(0.0)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            regression_metrics(np.zeros(5), np.zeros(4))

    def test_empty_predictions_are_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            regression_metrics(np.array([]), np.array([]))

    def test_reported_metrics_match_a_recomputation(self, real_benchmark):
        for result in real_benchmark.results:
            recomputed = regression_metrics(result.actuals, result.predictions)
            assert result.r2 == pytest.approx(recomputed["r2"])
            assert result.mae == pytest.approx(recomputed["mae"])
            assert result.rmse == pytest.approx(recomputed["rmse"])


# ---------------------------------------------------------------------------
# Chronology
# ---------------------------------------------------------------------------
class TestChronology:
    def test_train_is_entirely_earlier_than_test(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        split = chronological_split(data, test_size=0.2)
        assert split.train_end < split.test_start
        assert split.train.timestamps.max() < split.test.timestamps.min()

    def test_no_row_is_lost_or_used_twice(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        split = chronological_split(data, test_size=0.2)
        assert len(split.train) + len(split.test) == len(data)
        train_stamps = set(split.train.timestamps)
        test_stamps = set(split.test.timestamps)
        assert not (train_stamps & test_stamps)

    def test_the_test_block_is_the_later_portion_of_the_record(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        split = chronological_split(data, test_size=0.2)
        assert len(split.train) + len(split.test) == len(data)
        assert len(split.test) == pytest.approx(len(data) * 0.2, abs=1)
        assert split.test.timestamps.iloc[0] > data.timestamps.median()

    def test_the_modelling_table_is_sorted_by_time(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        assert data.is_chronological
        assert data.timestamps.is_monotonic_increasing

    def test_an_unordered_input_frame_is_sorted_before_being_split(self):
        frame = make_ml_frame(n=120)
        shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
        data = build_ml_frame(shuffled)
        assert data.is_chronological

    def test_invalid_test_size_is_rejected(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        for bad in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError, match="test_size"):
                chronological_split(data, test_size=bad)

    def test_a_record_too_short_to_split_is_rejected(self):
        data = build_ml_frame(make_ml_frame(n=2))
        with pytest.raises(ValueError, match="Cannot split"):
            chronological_split(data, test_size=0.2)

    def test_cross_validation_folds_respect_time(self, cleaned_result):
        """Every validation block must come strictly after the block it scores.

        The row indices are only meaningful as time because the modelling
        table is sorted by timestamp, which is checked here as well.  A
        shuffling splitter would interleave the indices and fail this test.
        """
        data = build_ml_frame(cleaned_result.frame, feature_set="weather")
        assert data.is_chronological

        cross = cross_validate_chronologically(
            build_model("linear_baseline"), data.X, data.y, n_splits=5
        )
        folds = cross["folds"]
        assert len(folds) == 5

        for fold in folds:
            assert fold["test_start"] > fold["train_end"]
            assert fold["n_train"] > 0 and fold["n_test"] > 0

        # The training window grows and the validation blocks march forward.
        for earlier, later in zip(folds, folds[1:]):
            assert later["train_end"] > earlier["train_end"]
            assert later["test_start"] > earlier["test_start"]

    def test_a_random_split_would_not_satisfy_the_chronology_check(self):
        """The same check applied to a shuffling splitter must fail.

        This guards the test above: it shows the assertion is actually
        sensitive to the splitter and not passing for every input.
        """
        x = np.arange(1, 201, dtype=float)
        y = np.where(x <= 100, x, x * 10.0)
        X = pd.DataFrame({"x": x})

        random_folds = []
        for train_index, test_index in KFold(n_splits=4).split(X):
            random_folds.append(
                {
                    "train_start": int(train_index.min()),
                    "train_end": int(train_index.max()),
                    "test_start": int(test_index.min()),
                    "test_end": int(test_index.max()),
                }
            )
        interleaved = [fold for fold in random_folds if fold["test_start"] < fold["train_end"]]
        assert interleaved, "a random split interleaves the blocks, as expected"

    def test_the_reported_test_period_is_in_the_future(self, real_benchmark):
        result = real_benchmark.get("random_forest")
        assert result.train_end < result.test_start
        assert result.test_end > result.test_start


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
class TestMissingValues:
    def test_rows_with_a_missing_target_are_dropped(self, cleaned_result):
        frame = cleaned_result.frame
        data = build_ml_frame(frame)
        expected = int(frame[TARGET].notna().sum())
        assert len(data) == expected
        assert data.y.notna().all()

    def test_a_missing_target_does_not_stop_the_remaining_rows(self):
        frame = make_ml_frame(n=100)
        frame.loc[frame.index[:10], TARGET] = np.nan
        data = build_ml_frame(frame)
        assert len(data) == 90
        assert data.y.notna().all()

    def test_the_model_trains_with_missing_features(self):
        frame = make_ml_frame(n=200)
        for column in ["T", "CO(GT)", "PT08.S1(CO)"]:
            frame.loc[frame.index[::7], column] = np.nan
        data = build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
        assert data.X.isna().any().any()
        result = evaluate_model(
            "random_forest", data, n_estimators=10, cv_splits=3
        )
        assert np.isfinite(result.predictions).all()
        assert np.isfinite(result.r2)

    def test_the_imputer_is_fitted_on_the_training_data_only(self):
        """The later block has a wild offset; the imputer must ignore it."""
        frame = make_ml_frame(n=400)
        frame.loc[frame.index[300:], "T"] = 1000.0
        data = build_ml_frame(frame, feature_set="weather")
        split = chronological_split(data, test_size=0.25)

        pipeline = build_model("linear_baseline")
        pipeline.fit(split.train.X, split.train.y)

        statistics = pipeline.named_steps["imputer"].statistics_
        assert statistics[0] == pytest.approx(split.train.X["T"].median())
        assert statistics[0] != pytest.approx(split.test.X["T"].median())

    def test_a_fully_missing_feature_column_does_not_break_the_model(self):
        frame = make_ml_frame(n=150)
        frame["T"] = np.nan
        data = build_ml_frame(frame, feature_set="weather")
        result = evaluate_model(
            "linear_baseline", data, cv_splits=3
        )
        assert np.isfinite(result.predictions).all()

    def test_missing_count_does_not_count_the_target(self):
        """The target's own missingness must never become a feature."""
        frame = make_ml_frame(n=20)
        frame.loc[frame.index[0], TARGET] = np.nan

        data = build_ml_frame(frame, feature_set="weather")
        assert (data.X["missing_count"] == 0).all()

        # The naive count over every measured column would have been 1.
        naive = frame[list(MEASURED_COLUMNS)].isna().sum(axis=1)
        assert int(naive.max()) == 1

    def test_blanking_the_target_leaves_the_features_untouched(self, cleaned_result):
        frame = cleaned_result.frame
        altered = frame.copy()
        target_values = altered[TARGET].to_numpy().copy()
        present = ~pd.isna(target_values)
        target_values[present] = (
            altered[TARGET].dropna().sample(frac=1.0, random_state=11).to_numpy()
        )
        altered[TARGET] = target_values

        before = build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
        after = build_ml_frame(altered, feature_set=PRIMARY_FEATURE_SET)

        # The same rows survive and their features are byte-for-byte identical.
        assert list(before.timestamps) == list(after.timestamps)
        pd.testing.assert_frame_equal(before.X, after.X)
        # The target really did change, so the test is not vacuous.
        assert not np.allclose(before.y.to_numpy(), after.y.to_numpy())


# ---------------------------------------------------------------------------
# Feature leakage
# ---------------------------------------------------------------------------
class TestFeatureLeakage:
    def test_the_target_is_not_a_feature(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        assert TARGET not in data.feature_names

    def test_the_nmhc_reference_is_not_a_feature(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        assert NMHC_REFERENCE not in data.feature_names

    def test_the_nmhc_sensor_proxy_is_not_a_feature(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame)
        assert NMHC_SENSOR not in data.feature_names

    def test_every_feature_tier_excludes_the_hydrocarbons(self):
        for name, columns in FEATURE_SETS.items():
            assert TARGET not in columns, name
            assert NMHC_REFERENCE not in columns, name
            assert NMHC_SENSOR not in columns, name

    def test_the_exclusions_are_documented_with_a_reason(self):
        assert set(EXCLUDED_COLUMNS) == {TARGET, NMHC_REFERENCE, NMHC_SENSOR}
        for reason in EXCLUDED_COLUMNS.values():
            assert isinstance(reason, str) and len(reason) > 10

    def test_scrambling_the_target_does_not_change_the_features(self, cleaned_result):
        frame = cleaned_result.frame
        scrambled = frame.copy()
        values = scrambled[TARGET].to_numpy().copy()
        present = ~pd.isna(values)
        values[present] = (
            scrambled[TARGET].dropna().sample(frac=1.0, random_state=7).to_numpy()
        )
        scrambled[TARGET] = values

        before = build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
        after = build_ml_frame(scrambled, feature_set=PRIMARY_FEATURE_SET)
        pd.testing.assert_frame_equal(before.X, after.X)

    def test_the_proxy_only_enters_when_explicitly_requested(self, cleaned_result):
        frame = cleaned_result.frame
        normal = build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
        with_proxy = build_ml_frame(
            frame, feature_set=PRIMARY_FEATURE_SET, allow_hydrocarbon_proxy=True
        )
        assert NMHC_SENSOR not in normal.feature_names
        assert NMHC_SENSOR in with_proxy.feature_names
        assert len(with_proxy.feature_names) == len(normal.feature_names) + 1

    def test_the_leakage_diagnostic_shows_the_proxy_would_help(self, real_leakage_report):
        report = real_leakage_report
        assert report.proxy_target_correlation > 0.95
        assert report.proxy_r2 > report.primary_r2
        assert report.r2_inflation > 0.05

    def test_the_leakage_diagnostic_names_every_exclusion(self, real_leakage_report):
        assert set(real_leakage_report.excluded_columns) == {
            TARGET,
            NMHC_REFERENCE,
            NMHC_SENSOR,
        }


# ---------------------------------------------------------------------------
# Feature sets: the sensor-only headline and the reference-assisted tier
# ---------------------------------------------------------------------------
class TestFeatureSets:
    """The headline model must be the low-cost one.

    ``PRIMARY_FEATURE_SET`` is what every reported number is computed on.  If
    it were ever pointed at the reference-assisted tier, the dashboard would
    be claiming that a cheap station can do something that needs the
    certified analysers, so these tests pin the two apart.
    """

    def test_the_headline_tier_is_the_sensor_only_tier(self):
        assert PRIMARY_FEATURE_SET == "sensors"
        assert REFERENCE_ASSISTED_FEATURE_SET == "full"
        assert PRIMARY_FEATURE_SET != REFERENCE_ASSISTED_FEATURE_SET

    def test_the_sensor_tier_uses_the_four_metal_oxide_sensors(self):
        columns = FEATURE_SETS["sensors"]
        assert len(SENSOR_FEATURES) == 4
        for sensor in SENSOR_FEATURES:
            assert sensor in columns
        assert NMHC_SENSOR not in columns

    def test_the_sensor_tier_also_uses_weather_and_calendar_features(self):
        columns = FEATURE_SETS["sensors"]
        for column in WEATHER_FEATURES + TIME_FEATURES + [MISSING_COUNT_FEATURE]:
            assert column in columns

    def test_the_sensor_tier_uses_no_reference_analyser(self):
        """This is the whole point of the headline model."""
        columns = FEATURE_SETS["sensors"]
        for analyser in OTHER_REFERENCE_FEATURES:
            assert analyser not in columns

    def test_the_reference_assisted_tier_is_the_sensor_tier_plus_the_three_analysers(self):
        sensors = set(FEATURE_SETS["sensors"])
        full = set(FEATURE_SETS[REFERENCE_ASSISTED_FEATURE_SET])
        assert full - sensors == set(OTHER_REFERENCE_FEATURES)
        assert sensors < full

    def test_no_tier_contains_the_target_or_the_hydrocarbons(self):
        for name, columns in FEATURE_SETS.items():
            assert TARGET not in columns, name
            assert NMHC_REFERENCE not in columns, name
            assert NMHC_SENSOR not in columns, name

    def test_every_tier_has_a_label_and_a_description(self):
        for name in FEATURE_SETS:
            assert name in FEATURE_SET_LABELS
            assert name in FEATURE_SET_DESCRIPTIONS
            assert FEATURE_SET_LABELS[name].strip()
            assert len(FEATURE_SET_DESCRIPTIONS[name]) > 20

    def test_the_labels_name_the_two_models_differently(self):
        sensor_label = FEATURE_SET_LABELS["sensors"]
        reference_label = FEATURE_SET_LABELS[REFERENCE_ASSISTED_FEATURE_SET]
        assert "Sensor-only" in sensor_label
        assert "Reference-assisted" in reference_label
        assert sensor_label != reference_label
        assert "headline" in sensor_label.lower()

    def test_the_descriptions_say_what_each_tier_is_allowed_to_use(self):
        sensor_text = FEATURE_SET_DESCRIPTIONS["sensors"].lower()
        reference_text = FEATURE_SET_DESCRIPTIONS[REFERENCE_ASSISTED_FEATURE_SET].lower()
        assert "low-cost" in sensor_text
        for analyser in OTHER_REFERENCE_FEATURES:
            assert analyser.lower() in reference_text
        assert "low-cost" not in reference_text

    def test_the_sensor_only_feature_matrix_ignores_the_reference_analysers(
        self, cleaned_result
    ):
        """Scrambling the analysers must leave the headline features alone."""
        frame = cleaned_result.frame
        perturbed = frame.copy()
        perturbed["CO(GT)"] = 999.0
        perturbed["NOx(GT)"] = 999.0
        perturbed["NO2(GT)"] = 999.0

        before = build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
        after = build_ml_frame(perturbed, feature_set=PRIMARY_FEATURE_SET)
        pd.testing.assert_frame_equal(before.X, after.X)

        # The reference-assisted tier, by contrast, does move.
        full_before = build_ml_frame(
            frame, feature_set=REFERENCE_ASSISTED_FEATURE_SET
        )
        full_after = build_ml_frame(
            perturbed, feature_set=REFERENCE_ASSISTED_FEATURE_SET
        )
        assert not full_before.X["CO(GT)"].equals(full_after.X["CO(GT)"])

    def test_the_sensor_only_feature_matrix_ignores_the_excluded_hydrocarbons(
        self, cleaned_result
    ):
        """The excluded columns must not sneak back in through the missing-count.

        The missing-count is taken over the tier's own variables, so the
        availability of a column the model is not allowed to see cannot
        change a single feature value.
        """
        frame = cleaned_result.frame
        perturbed = frame.copy()
        perturbed[NMHC_SENSOR] = 999.0
        perturbed[NMHC_REFERENCE] = 999.0

        before = build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
        after = build_ml_frame(perturbed, feature_set=PRIMARY_FEATURE_SET)
        pd.testing.assert_frame_equal(before.X, after.X)

    def test_the_sensor_only_model_is_fitted_without_any_analyser(self, cleaned_result):
        data = build_ml_frame(cleaned_result.frame, feature_set=PRIMARY_FEATURE_SET)
        for analyser in OTHER_REFERENCE_FEATURES:
            assert analyser not in data.feature_names
        assert set(data.feature_names) == set(FEATURE_SETS["sensors"])

    def test_the_benchmark_reports_the_sensor_only_tier_by_default(self, real_benchmark):
        assert {result.feature_set for result in real_benchmark.results} == {
            PRIMARY_FEATURE_SET
        }
        assert real_benchmark.get("random_forest").feature_set == "sensors"

    def test_the_reference_assisted_tier_beats_the_sensor_only_tier(
        self, real_tier_benchmark
    ):
        """The analysers carry real signal, but they need real instruments."""
        weather = real_tier_benchmark.get("random_forest", feature_set="weather")
        sensors = real_tier_benchmark.get("random_forest", feature_set="sensors")
        reference = real_tier_benchmark.get(
            "random_forest", feature_set=REFERENCE_ASSISTED_FEATURE_SET
        )
        assert reference.r2 > sensors.r2
        assert sensors.r2 > weather.r2

    def test_the_leakage_diagnostic_is_measured_against_the_sensor_only_tier(
        self, real_leakage_report, real_tier_benchmark
    ):
        """The cost of excluding the proxy is reported for the cheap model.

        The diagnostic fits the headline model again, so its "without the
        proxy" score must be the same number the benzene page shows.
        """
        sensors = real_tier_benchmark.get("random_forest", feature_set="sensors")
        assert real_leakage_report.primary_r2 == pytest.approx(sensors.r2)
        assert real_leakage_report.primary_rmse == pytest.approx(sensors.rmse)


# ---------------------------------------------------------------------------
# The reported benchmark on the real data
# ---------------------------------------------------------------------------
class TestBenchmarkOnRealData:
    def test_every_row_with_a_target_is_used_exactly_once(self, real_benchmark, cleaned_result):
        result = real_benchmark.get("random_forest")
        assert result.n_train + result.n_test == int(cleaned_result.frame[TARGET].notna().sum())

    def test_all_three_models_were_fitted(self, real_benchmark):
        assert {result.name for result in real_benchmark.results} == set(MODEL_NAMES)

    def test_the_random_forest_beats_both_baselines(self, real_benchmark):
        forest = real_benchmark.get("random_forest")
        assert forest.r2 > real_benchmark.get("linear_baseline").r2
        assert forest.r2 > real_benchmark.get("mean_baseline").r2
        assert forest.rmse < real_benchmark.get("linear_baseline").rmse

    def test_the_mean_baseline_is_worse_than_zero_information(self, real_benchmark):
        """R2 below zero means the later period is genuinely different.

        This is the honest consequence of a chronological split: the training
        average is not a good guess for the test period, because the level of
        benzene changed between March 2004 and early 2005.
        """
        baseline = real_benchmark.get("mean_baseline")
        assert baseline.r2 < 0.0

    def test_the_random_forest_metrics_are_finite_and_plausible(self, real_benchmark):
        forest = real_benchmark.get("random_forest")
        assert np.isfinite(forest.r2) and np.isfinite(forest.mae) and np.isfinite(forest.rmse)
        assert 0.5 < forest.r2 < 1.0
        assert 0.0 < forest.mae < forest.rmse < 10.0

    def test_the_chronological_score_is_not_optimistic(self, real_benchmark):
        """The held-out score should be close to the cross-validated score.

        If the test score were far above the cross-validated score it would
        mean the test period was easier than the record as a whole.
        """
        forest = real_benchmark.get("random_forest")
        assert forest.r2 <= forest.cv_r2_mean + 0.10
        assert forest.rmse >= forest.cv_rmse_mean - 0.50

    def test_the_importances_form_a_distribution(self, real_benchmark):
        forest = real_benchmark.get("random_forest")
        assert forest.feature_importances
        assert sum(forest.feature_importances.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(value >= 0 for value in forest.feature_importances.values())

    def test_the_importance_table_is_sorted(self, real_benchmark):
        forest = real_benchmark.get("random_forest")
        table = feature_importance_table(forest)
        assert list(table["importance"]) == sorted(table["importance"], reverse=True)

    def test_no_feature_importance_can_come_from_the_target(self, real_benchmark):
        forest = real_benchmark.get("random_forest")
        for name in forest.feature_importances:
            assert TARGET not in name
            assert NMHC_REFERENCE not in name
            assert NMHC_SENSOR not in name

    def test_the_summary_frame_reports_every_model(self, real_benchmark):
        table = real_benchmark.summary_frame()
        assert len(table) == len(MODEL_NAMES)
        for column in ["r2", "mae", "rmse", "n_train", "n_test"]:
            assert column in table.columns
            assert table[column].notna().all()

    def test_unknown_models_and_feature_sets_are_rejected(self, cleaned_result):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("magic")
        with pytest.raises(ValueError, match="Unknown feature set"):
            build_ml_frame(cleaned_result.frame, feature_set="magic")

    def test_a_missing_feature_column_is_reported_not_ignored(self):
        frame = make_ml_frame(n=50).drop(columns=["PT08.S4(NO2)"])
        with pytest.raises(ValueError, match="missing these columns"):
            build_ml_frame(frame, feature_set=PRIMARY_FEATURE_SET)
