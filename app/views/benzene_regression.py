"""Page 6 - Benzene Regression.

How well the reference benzene concentration can be estimated from signals
that are genuinely available at prediction time, and why the evaluation is
built the way it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ecopulse.ml import (
    EXCLUDED_COLUMNS,
    FEATURE_SET_DESCRIPTIONS,
    FEATURE_SET_LABELS,
    FEATURE_SETS,
    MODEL_LABELS,
    MODEL_NAMES,
    NMHC_SENSOR,
    PRIMARY_FEATURE_SET,
    REFERENCE_ASSISTED_FEATURE_SET,
    feature_importance_table,
)

from app import theme


def render(context) -> None:
    theme.page_header(
        "Benzene Regression",
        "Can benzene be estimated from the other signals in the record? This page "
        "answers that with a model that is actually fitted and scored on the real "
        "data - and it is careful about the two ways such a model usually cheats. "
        "The headline model uses **only low-cost sensors and weather**; a "
        "reference-assisted model is reported next to it, clearly labelled.",
    )
    theme.disclaimer()

    try:
        benchmark = context.benchmark
        leakage = context.leakage
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(
            "**The benzene models could not be fitted.** The error below explains why."
        )
        st.exception(exc)
        return

    # The headline model: cheap sensors and weather only.
    forest = benchmark.get("random_forest", PRIMARY_FEATURE_SET)
    linear = benchmark.get("linear_baseline", PRIMARY_FEATURE_SET)
    baseline = benchmark.get("mean_baseline", PRIMARY_FEATURE_SET)

    # The reference-assisted comparison: the same pipeline with the co-located
    # certified analysers added to the predictors.
    assisted = benchmark.get("random_forest", REFERENCE_ASSISTED_FEATURE_SET)

    # -- the question and the rules ----------------------------------------
    theme.section(
        "The question, and the two ways to cheat at it",
        "Benzene is measured by a certified reference analyser. The task is to "
        "estimate that measurement from signals a cheaper station could also "
        "collect.",
    )
    st.markdown(
        "A naive version of this task is easy to do badly. Two mistakes inflate the "
        "score without making the model useful:"
    )
    st.markdown(
        "1. **Leaking the answer.** If the target, or anything computed from it, "
        "slips into the features, the model is graded on information it will not "
        "have in real use. EcoPulse excludes the target and verifies the exclusion "
        "by scrambling the target column and checking the feature matrix does not "
        "change.\n"
        f"2. **Using a proxy for the answer.** `{NMHC_SENSOR}` is a cheap hydrocarbon "
        "sensor that correlates at **r = 0.982** with the benzene reference. Adding "
        "it raises the score a lot, but the extra accuracy comes from measuring the "
        "same quantity with a second instrument. It is excluded from the headline "
        "model, and the cost of excluding it is measured below rather than assumed."
    )

    theme.notice(
        "**A third rule: the split is chronological.** The model is trained on the "
        "earlier part of the record and scored on the later part, because that is the "
        "only split that resembles using the model on a future hour. A random split "
        "would put neighbouring hours on both sides and let the model interpolate its "
        "way to a flattering score.",
        kind="warn",
    )

    # -- which model is the headline ----------------------------------------
    theme.section(
        "Two models, two different questions",
        "The headline model is the low-cost sensor model. The reference-assisted "
        "model is reported beside it, and never as a substitute for it.",
    )
    st.markdown(
        "Everything below is fitted with the same pipeline, on the same "
        "chronological split, and scored on the same held-out hours. The only "
        "difference between them is **which instruments are allowed to be "
        "predictors**:"
    )
    st.markdown(
        f"- **{FEATURE_SET_LABELS['sensors']}** - {FEATURE_SET_DESCRIPTIONS['sensors']} "
        "This is the headline model on this page, and the number below is the one a "
        "low-cost station could reproduce.\n"
        f"- **{FEATURE_SET_LABELS['full']}** - {FEATURE_SET_DESCRIPTIONS['full']} "
        "It answers a different question: how much does adding the expensive "
        "analysers to a cheap station actually buy? It is a comparison, not the "
        "headline, and its score must not be quoted as a low-cost sensor result."
    )
    theme.notice(
        "**Why this distinction matters.** A model that is allowed to read the "
        "co-located certified analysers is not measuring anything a cheap station "
        "could measure: it is being handed the expensive instrument's answer for "
        "CO, NOx and NO2. That is a perfectly legitimate upper bound, but it is a "
        "different claim from 'benzene can be estimated from cheap sensors'. "
        "Reporting only the reference-assisted number would overstate what the low-cost "
        "case achieves; reporting only the sensor-only number would hide how much the "
        "analysers contribute. EcoPulse reports both, labelled.",
        kind="info",
    )

    # -- the split ---------------------------------------------------------
    theme.section("The train / test split")
    left, right = st.columns([3, 2])
    with left:
        theme.metric_cards(
            [
                {
                    "label": "Training hours",
                    "value": theme.fmt_number(forest.n_train, 0),
                    "note": f"up to {pd.Timestamp(forest.train_end):%d %b %Y %H:%M}",
                    "help": "The earliest part of the record, used to fit the model.",
                },
                {
                    "label": "Test hours",
                    "value": theme.fmt_number(forest.n_test, 0),
                    "note": (
                        f"{pd.Timestamp(forest.test_start):%d %b %Y} to "
                        f"{pd.Timestamp(forest.test_end):%d %b %Y}"
                    ),
                    "help": "The latest part of the record, never seen during fitting.",
                },
                {
                    "label": "Benzene readings",
                    "value": theme.fmt_number(forest.n_train + forest.n_test, 0),
                    "note": "of 9,357 hours",
                    "help": "Rows whose benzene target is present. Rows without a target cannot be trained on or scored.",
                },
            ]
        )
    with right:
        st.markdown("**The split in one picture**")
        split_figure = go.Figure()
        split_figure.add_trace(
            go.Bar(
                x=[forest.n_train, forest.n_test],
                y=["Training", "Test"],
                orientation="h",
                marker_color=[theme.SECONDARY, theme.PRIMARY],
                text=[f"{forest.n_train:,} h", f"{forest.n_test:,} h"],
                textposition="inside",
                hovertemplate="%{y}: %{x:,} hours<extra></extra>",
            )
        )
        split_figure.update_xaxes(title="Hours")
        split_figure.update_yaxes(title="")
        theme.style_figure(split_figure, height=180, showlegend=False)
        st.plotly_chart(split_figure, width="stretch")

    # -- the models --------------------------------------------------------
    theme.section(
        "The models, scored on the future",
        "All three models use the sensor-only predictor set, see exactly the same "
        "training rows and are scored on exactly the same test rows.",
    )

    rows = []
    for name in MODEL_NAMES:
        result = benchmark.get(name, PRIMARY_FEATURE_SET)
        rows.append(
            {
                "Model": MODEL_LABELS.get(name, name),
                "R²": result.r2,
                "MAE (µg/m³)": result.mae,
                "RMSE (µg/m³)": result.rmse,
                "CV R² (mean)": result.cv_r2_mean,
                "CV R² (std)": result.cv_r2_std,
            }
        )
    comparison = pd.DataFrame(rows)

    st.dataframe(
        comparison.style.format(
            {
                "R²": "{:.3f}",
                "MAE (µg/m³)": "{:.3f}",
                "RMSE (µg/m³)": "{:.3f}",
                "CV R² (mean)": "{:.3f}",
                "CV R² (std)": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Sensor-only predictor set: weather, the four non-hydrocarbon metal-oxide "
        "sensors, calendar features and the missing-count taken over those same "
        "variables. MAE and RMSE are in "
        "µg/m³, the unit of the benzene target. CV R² is the "
        "mean and standard deviation across five expanding-window, chronological "
        "cross-validation folds on the training data."
    )

    theme.metric_cards(
        [
            {
                "label": "Sensor-only R²",
                "value": theme.fmt_number(forest.r2, 3),
                "note": "headline model, on the held-out future",
                "help": "The share of variance in the test-period benzene readings explained by the Random Forest built from low-cost sensors and weather only.",
            },
            {
                "label": "Sensor-only RMSE",
                "value": f"{theme.fmt_number(forest.rmse, 2)} ug/m3",
                "note": f"MAE {theme.fmt_number(forest.mae, 2)} ug/m3",
                "help": "Root mean squared error on the test period. A typical error size in the target's own unit.",
            },
            {
                "label": "Error reduction vs mean",
                "value": f"{theme.fmt_number(baseline.rmse - forest.rmse, 2)} µg/m³",
                "note": "lower RMSE",
                "help": "How much the Random Forest reduces the error compared with predicting the training average. Positive means better than the mean baseline.",
            },
            {
                "label": "Chronological CV R²",
                "value": theme.fmt_number(forest.cv_r2_mean, 3),
                "note": f"sd {theme.fmt_number(forest.cv_r2_std, 3)}",
                "help": "The average score across five time-ordered folds of the training data. Close to the held-out score, which is a good sign.",
            },
        ]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="R²",
            x=comparison["Model"],
            y=comparison["R²"],
            marker_color=[theme.MUTED, theme.ACCENT, theme.PRIMARY],
            text=comparison["R²"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            hovertemplate="%{x}<br>R²: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line=dict(color=theme.INK, width=1))
    figure.update_yaxes(title="R² on the held-out test period", range=[-0.3, 1.0])
    figure.update_xaxes(title="")
    theme.style_figure(figure, height=330, showlegend=False)
    st.plotly_chart(figure, width="stretch")

    theme.notice(
        f"**The mean baseline scores R² = {baseline.r2:.3f}, below zero.** That looks "
        "wrong until you remember the split is chronological: the average benzene "
        "level in the later test period is different from the training average, so "
        "predicting the training mean is worse than predicting nothing. This is "
        "exactly the kind of effect a random split would have hidden, and it is why "
        "the baseline is included at all - it shows how much of the Random Forest's "
        "score is real skill rather than a lucky test period.",
        kind="info",
    )

    # -- observed vs predicted ---------------------------------------------
    theme.section(
        "What the model actually predicts",
        "Observed benzene against the headline sensor-only Random Forest's estimate, "
        "over the held-out test period.",
    )

    timestamps = pd.to_datetime(forest.test_timestamps)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=forest.actuals,
            mode="lines",
            name="Observed benzene",
            line=dict(color=theme.INK, width=1.4),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Observed: %{y:.2f} ug/m3<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=forest.predictions,
            mode="lines",
            name="Predicted benzene",
            line=dict(color=theme.PRIMARY, width=1.6),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Predicted: %{y:.2f} ug/m3<extra></extra>",
        )
    )
    figure.update_yaxes(title="Benzene (ug/m3)")
    theme.style_figure(figure, height=380)
    st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Predicted against observed**")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=forest.actuals,
                y=forest.predictions,
                mode="markers",
                name="Test hours",
                marker=dict(size=4, color=theme.PRIMARY, opacity=0.35),
                hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
            )
        )
        limits = [
            float(min(forest.actuals.min(), forest.predictions.min())),
            float(max(forest.actuals.max(), forest.predictions.max())),
        ]
        figure.add_trace(
            go.Scatter(
                x=limits,
                y=limits,
                mode="lines",
                name="Perfect prediction",
                line=dict(color=theme.DANGER, width=2, dash="dash"),
                hovertemplate="Perfect: %{y:.2f}<extra></extra>",
            )
        )
        figure.update_xaxes(title="Observed benzene (ug/m3)")
        figure.update_yaxes(title="Predicted benzene (ug/m3)")
        theme.style_figure(figure, height=360)
        st.plotly_chart(figure, width="stretch")

    with right:
        st.markdown("**Residuals over time**")
        residuals = forest.predictions - forest.actuals
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=timestamps,
                y=residuals,
                mode="markers",
                name="Residual",
                marker=dict(size=4, color=theme.SECONDARY, opacity=0.4),
                hovertemplate="%{x|%d %b %Y %H:%M}<br>Error: %{y:+.2f} ug/m3<extra></extra>",
            )
        )
        figure.add_hline(y=0, line=dict(color=theme.INK, width=1))
        figure.update_yaxes(title="Predicted minus observed (ug/m3)")
        theme.style_figure(figure, height=360, showlegend=False)
        st.plotly_chart(figure, width="stretch")

    theme.metric_cards(
        [
            {
                "label": "Mean bias",
                "value": f"{theme.fmt_number(float(np.mean(residuals)), 2)} ug/m3",
                "note": "predicted minus observed",
                "help": "Positive means the model over-predicts on average; negative means it under-predicts.",
            },
            {
                "label": "Error standard deviation",
                "value": f"{theme.fmt_number(float(np.std(residuals)), 2)} ug/m3",
                "note": "spread of the errors",
                "help": "How widely the individual errors scatter around their mean.",
            },
            {
                "label": "Within 2 ug/m3",
                "value": theme.fmt_pct(float(np.mean(np.abs(residuals) <= 2.0)), 1),
                "note": "of test hours",
                "help": "The share of test hours where the estimate was within 2 ug/m3 of the real reading.",
            },
        ]
    )

    # -- feature importance -------------------------------------------------
    theme.section(
        "What drives the estimate",
        "Random Forest feature importance for the headline sensor-only model, "
        "which measures how much each variable reduces prediction error when it is "
        "used to split the data. These sum to 1.",
    )

    importances = feature_importance_table(forest, top=15)
    importances["Readable"] = importances["feature"].map(
        lambda name: name.replace("missingindicator_", "missing flag: ")
    )

    figure = go.Figure(
        go.Bar(
            x=importances["importance"],
            y=importances["Readable"],
            orientation="h",
            marker_color=[
                theme.PRIMARY if "missing" not in name else theme.MUTED
                for name in importances["Readable"]
            ],
            hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Importance")
    figure.update_yaxes(title="", autorange="reversed")
    theme.style_figure(figure, height=max(320, 30 * len(importances) + 90), showlegend=False)
    st.plotly_chart(figure, width="stretch")

    top_feature = importances.iloc[0]["Readable"]
    theme.notice(
        f"**`{top_feature}` is the most important single input.** None of the "
        "importances come from the target or from the excluded hydrocarbon sensor: "
        "the exclusions are enforced in code and checked by the test suite. The grey "
        f"bars are the imputation flags for `{NMHC_SENSOR}`-free features, which tell "
        "the model when a reading was filled in rather than measured.",
        kind="info",
    )

    # -- feature tiers ------------------------------------------------------
    theme.section(
        "Sensor-only against reference-assisted",
        "The same Random Forest fitted on three nested sets of predictors. The "
        "difference between the rows is what each group of instruments contributes.",
    )

    tier_rows = []
    for tier in ("weather", "sensors", REFERENCE_ASSISTED_FEATURE_SET):
        result = benchmark.get("random_forest", tier)
        tier_rows.append(
            {
                "Model": FEATURE_SET_LABELS.get(tier, tier),
                "Predictors": FEATURE_SET_DESCRIPTIONS.get(tier, tier),
                "R²": result.r2,
                "MAE": result.mae,
                "RMSE": result.rmse,
            }
        )
    tier_table = pd.DataFrame(tier_rows)
    st.dataframe(
        tier_table.style.format({"R²": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )

    figure = go.Figure(
        go.Bar(
            x=tier_table["Model"],
            y=tier_table["R²"],
            marker_color=[theme.MUTED, theme.PRIMARY, theme.SECONDARY],
            text=tier_table["R²"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            hovertemplate="%{x}<br>R²: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_yaxes(title="R² on the held-out test period", range=[0, 1.0])
    figure.update_xaxes(title="")
    theme.style_figure(figure, height=320, showlegend=False)
    st.plotly_chart(figure, width="stretch")

    theme.notice(
        f"**The low-cost sensor model is the headline, and it is not far behind.** "
        f"Weather, the four non-hydrocarbon sensors and the calendar reach "
        f"R² = {forest.r2:.3f} - that is the number for the low-cost sensor case. "
        f"Adding the reference analysers lifts the score only to "
        f"R² = {assisted.r2:.3f}, an increment of {assisted.r2 - forest.r2:+.3f}. "
        "The operational lesson is that most of the benzene signal is already "
        "present in inexpensive instruments, and the expensive analysers add a "
        "comparatively small amount on top. The reference-assisted score is a "
        "ceiling for a station that already owns those analysers - it is not a "
        "low-cost sensor result.",
        kind="info",
    )

    # -- leakage diagnostic -------------------------------------------------
    theme.section(
        "The measured cost of excluding the proxy",
        "The same Random Forest, fitted once without and once with the hydrocarbon "
        "sensor that correlates at r = 0.982 with the target. Both fits use the "
        "headline sensor-only predictor set.",
    )

    theme.metric_cards(
        [
            {
                "label": "Without the proxy",
                "value": theme.fmt_number(leakage.primary_r2, 3),
                "note": f"RMSE {theme.fmt_number(leakage.primary_rmse, 2)} ug/m3",
                "help": "The headline sensor-only model on this page.",
            },
            {
                "label": "With the proxy",
                "value": theme.fmt_number(leakage.proxy_r2, 3),
                "note": f"RMSE {theme.fmt_number(leakage.proxy_rmse, 2)} ug/m3",
                "help": "The same sensor-only model with PT08.S2(NMHC) added back in.",
            },
            {
                "label": "R² gained",
                "value": f"+{theme.fmt_number(leakage.proxy_r2 - leakage.primary_r2, 3)}",
                "note": "by measuring the answer twice",
                "help": "The improvement that comes from adding the near-perfect proxy.",
            },
            {
                "label": "RMSE reduction",
                "value": theme.fmt_pct(
                    1.0 - (leakage.proxy_rmse / leakage.primary_rmse), 1
                ),
                "note": "from the proxy",
                "help": "How much smaller the error would look if the proxy were allowed in.",
            },
        ]
    )

    st.markdown(
        f"The proxy correlates at **r = {leakage.proxy_target_correlation:.3f}** with "
        "the benzene reference. Adding it would raise the score from "
        f"{leakage.primary_r2:.3f} to {leakage.proxy_r2:.3f}. That is a real "
        "improvement in the arithmetic and a meaningless one in practice: the model "
        "would be reading the answer from a second instrument rather than estimating "
        "it. Reporting the excluded model is the honest choice, and reporting the size "
        "of the exclusion is what makes that choice checkable."
    )

    # -- what is excluded ---------------------------------------------------
    with st.expander("What the model is not allowed to see, and what it does use"):
        excluded_table = pd.DataFrame(
            [
                {"Column": column, "Why it is excluded": reason}
                for column, reason in EXCLUDED_COLUMNS.items()
            ]
        )
        st.markdown("**Excluded columns**")
        st.dataframe(excluded_table, width="stretch", hide_index=True)

        st.markdown("**The predictors the headline model does use**")
        st.caption(
            f"{FEATURE_SET_LABELS[PRIMARY_FEATURE_SET]}: "
            f"{FEATURE_SET_DESCRIPTIONS[PRIMARY_FEATURE_SET]}"
        )
        st.code("\n".join(FEATURE_SETS[PRIMARY_FEATURE_SET]), language="text")

        st.markdown(
            f"**The extra predictors of the {FEATURE_SET_LABELS[REFERENCE_ASSISTED_FEATURE_SET]} "
            "model**"
        )
        st.caption(
            f"{FEATURE_SET_DESCRIPTIONS[REFERENCE_ASSISTED_FEATURE_SET]} These are "
            "the certified reference analysers, so this model is not a low-cost "
            "sensor model."
        )
        st.code(
            "\n".join(
                column
                for column in FEATURE_SETS[REFERENCE_ASSISTED_FEATURE_SET]
                if column not in FEATURE_SETS[PRIMARY_FEATURE_SET]
            ),
            language="text",
        )

    if linear.coefficients:
        with st.expander("The linear baseline, for comparison"):
            st.caption(
                "The linear model scores lower than the Random Forest, which tells you "
                "the relationship is not simply linear. Its coefficients are shown so "
                "the direction of each effect can be read."
            )
            coef_rows = [
                {"Feature": name, "Coefficient": value}
                for name, value in sorted(
                    linear.coefficients.items(), key=lambda item: abs(item[1]), reverse=True
                )
            ]
            st.dataframe(
                pd.DataFrame(coef_rows).style.format({"Coefficient": "{:+.4f}"}),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Because the features are standardised inside the pipeline, each "
                "coefficient is the change in benzene (ug/m3) per one standard deviation "
                "of that feature."
            )

    # -- honest limitations ------------------------------------------------
    theme.section("How to read these numbers")
    theme.notice(
        "**The headline number is the sensor-only one.** The Random Forest result "
        f"quoted above (R² = {forest.r2:.3f}) is the low-cost sensor model: weather, "
        "the four non-hydrocarbon metal-oxide sensors and the calendar. The "
        "reference-assisted figure (R² = "
        f"{assisted.r2:.3f}) additionally uses the co-located certified analysers "
        "for CO, NOx and NO2, so it describes a station that already owns those "
        "instruments and must not be quoted as a low-cost sensor result.",
        kind="info",
    )
    theme.notice(
        "**This is a retrospective estimate, not a forecast.** The model uses "
        "contemporaneous readings - other instruments measuring at the same hour - to "
        "estimate benzene at that same hour. It is not predicting tomorrow. A genuine "
        "forecast would need every feature lagged into the past and would score "
        "worse. **The test period is also unusually hard:** it sits in early 2005, "
        "after the sensor sensitivity had begun to shift, and the training average is "
        "a poor guide to it. That is the point of a chronological split, and it is "
        "why the baseline R² is negative. Finally, the figures describe one monitoring "
        "site in one city across fourteen months, and they should not be generalised "
        "to other places or other years.",
        kind="warn",
    )
