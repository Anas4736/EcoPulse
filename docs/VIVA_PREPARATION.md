# EcoPulse — Viva Preparation Guide

A question-and-answer companion to `PROJECT_REPORT.md`. Every number below is
computed by the implementation; the same figures appear on the dashboard and are
re-derived by the test suite.

---

## How to use this guide

1. Read **Section 0** and the **numbers table** until you can answer from memory.
2. Work through **Sections A–F** and try to answer before reading the response.
3. Rehearse **Section G** out loud. Examiners reward boundaries more than
   enthusiasm; a candidate who says "this analysis cannot show that" is stronger
   than one who overclaims.
4. Keep the **one-sentence answers** in mind for rapid-fire questioning.

If you do not know an answer, say so and say what you *would* do to find out. That
is a better response than inventing a number. Never quote a figure you cannot
trace to the dashboard or the tests.

---

## 0. The 60-second summary

> EcoPulse analyses a fixed historical air-quality record from one urban monitoring
> site: 9,357 hourly observations from March 2004 to April 2005. The data combines
> five certified reference analysers, five cheap metal-oxide sensors and weather
> measurements. We built a reproducible cleaning pipeline, an exploratory analysis,
> a controlled test for sensor drift, and a leakage-safe model that estimates
> benzene from the other signals. The dashboard presents all of it and recomputes
> every number on load. The main findings are that the timeline is complete, that
> nitrogen dioxide has a strong evening and winter cycle, that only one of five
> sensors shows evidence consistent with a temporal change, and that benzene can be
> estimated with a held-out R² of 0.86 — provided the near-perfect proxy sensor is
> excluded and the cost of excluding it is reported.

---

## Numbers you must know

| Fact | Value |
| --- | ---: |
| Rows | 9,357 |
| Period | 2004-03-10 18:00 → 2005-04-04 14:00 |
| Missing hours / gaps / duplicate timestamps | 0 / 0 / 0 |
| Sentinel markers (`-200`) replaced | 16,701 (13.73 % of cells) |
| Valid `CO(GT)` | 7,674 (82.01 %) |
| Valid `NO2(GT)` | 7,715 (82.45 %) |
| Valid `NOx(GT)` | 7,718 (82.48 %) |
| Valid `NMHC(GT)` | 914 (9.77 %) |
| Valid device/weather columns | 8,991 (96.09 %) |
| Missingness groups / outage blocks | 4 / 553 |
| Longest device outage | 366 hours across 16 blocks |
| NO2 mean / median / max | 113.08 / 109.0 / 339.7 µg/m³ |
| NO2 peak hour / trough hour | 19:00 (150.08) / 04:00 (58.60) |
| NO2 winter / summer mean | 139.91 / 91.46 µg/m³ |
| NO2 hours above 200 µg/m³ | 388 (5.03 % of valid) |
| NO2 episodes / longest / median | 137 / 18 h / 2 h |
| NO2 exceedance hours 2004 / 2005 | 115 / 273 |
| `PT08.S2(NMHC)` vs benzene | r = 0.982 |
| `PT08.S4(NO2)` vs NO2 | r = 0.158 |
| AH vs Magnus formula | r = 0.99993, scale CV 0.80 % |
| Drift verdict, `PT08.S3(NOx)` | slope −0.0394/month, CI [−0.0591, −0.0250] |
| Drift verdicts, other four sensors | no clear drift signal |
| Model rows: train / test | 7,192 / 1,799 |
| Train end / test start | 2005-01-16 01:00 / 2005-01-16 02:00 |
| Mean baseline R² / RMSE | −0.1822 / 7.0836 µg/m³ |
| Linear R² / RMSE (sensor-only, headline) | 0.5146 / 4.5390 µg/m³ |
| Random Forest R² / MAE / RMSE (sensor-only, headline) | 0.8404 / 2.2366 / 2.6029 µg/m³ |
| RF cross-validation R² (sensor-only) | 0.9086 ± 0.0296 |
| RF weather tier R² | 0.3078 |
| RF sensors tier R² (headline) | 0.8404 |
| RF reference-assisted tier R² (comparison only) | 0.8580 |
| Leakage: with proxy R² / RMSE | 0.9695 / 1.1382 µg/m³ |
| Leakage: R² inflation / RMSE reduction | +0.1291 / 56.3 % |
| Tests | 281 |
| Source SHA-256 (XLSX) | `C92D102B…3124BD5D` |

---

## A. Dataset and provenance

**A1. What is the dataset and where did it come from?**
The UCI Air Quality dataset (ID 360), contributed by De Vito et al. It is a single
Excel workbook of hourly readings from a gas multi-sensor device co-located with a
certified reference analyser at a road site in an Italian city.

**A2. How do you know the data is the original, unmodified data?**
Two ways. The test suite asserts the SHA-256 digest of both files in `Data/`, and
the dashboard's Methodology page recomputes the digests at runtime and shows whether
they match. If either file changed, both would say so.

**A3. Why does that matter?**
Because every claim is derived from those files. If the input can change silently,
the results are not reproducible and the tests would pass against different data
than the one the report describes.

**A4. What are the three kinds of variable?**
Five reference analysers (CO, NMHC, benzene, NOx, NO2), five metal-oxide sensors
(PT08 series), and three weather variables (temperature, relative humidity, absolute
humidity). Thirteen measured columns in total.

**A5. How many rows, and what period?**
9,357 hourly rows, from 10 March 2004 18:00 to 4 April 2005 14:00.

**A6. Is the record continuous?**
Yes. There are 9,357 expected hours and 9,357 rows: zero gaps, zero duplicate
timestamps, and 100 % retention after cleaning.

**A7. Why is continuity important?**
It makes "hours" well defined. When we say 388 hours above a threshold, or an
episode of 18 hours, there is no ambiguity about what an hour is or whether a gap
was silently skipped.

**A8. What is unusual about the missing values?**
They are encoded as `-200`, a sentinel, rather than as blanks. If you compute
statistics without handling it, the means and correlations are wrong. There are
16,701 such markers, 13.73 % of all cells.

---

## B. Cleaning and data quality

**B1. Walk me through the cleaning.**
Read the sheet; combine `Date` and `Time` into one hourly timestamp; replace every
`-200` with a true missing value; remove exact duplicate rows and duplicate
timestamps; validate the hourly index; reject non-physical values; export the CSV.

**B2. How many rows did cleaning delete?**
None. 9,357 in, 9,357 out. What changed is that sentinel values became honest
missing values.

**B3. Why not drop rows with missing values?**
Because the row still carries real information in the columns that were recorded,
and dropping rows would bias the record toward periods when every instrument
happened to be working. Missingness here is in long device outages, so dropping
would disproportionately remove whole days.

**B4. How many columns are affected by missingness?**
All thirteen measured columns have some, but they organise into exactly **four**
groups based on which columns go missing together.

**B5. What are the four groups?**
A nine-column device outage group; NOx+NO2 together; CO alone; and NMHC alone.

**B6. How many outage blocks are there in total?**
553. Of those, 16 blocks knock out nine columns simultaneously for a total of 366
hours; the rest are per-analyser blocks.

**B7. Why is that number of blocks important?**
Because it shows the missingness is not random. Random missingness would be spread
thinly; here it comes in long, structurally coherent episodes, which means the
missing hours may differ systematically from the observed hours.

**B8. Which column is effectively unusable?**
`NMHC(GT)`: only 914 valid hours, 9.77 % of the record. Any statistic about it
describes a small, non-representative subset, so it is excluded from modelling.

**B9. What did you do about imputation?**
Almost nothing, deliberately. Descriptive statistics use observed values only and
report the pairwise sample size. Imputation happens only inside the benzene model's
predictor pipeline, using the training median plus missing-value indicators.

**B10. Why inside the pipeline?**
So the imputation statistics are learned from training rows only. If you impute
before splitting, information from the test period leaks into training.

**B11. What is the difference between a missing row and a missing value?**
A missing row would be an absent hour. There are none. A missing value is a
recorded hour where one instrument did not report. All of our missingness is
value-level.

---

## C. Exploratory analysis

**C1. What is the most striking diurnal pattern?**
All three main pollutants peak around 19:00 and bottom out in the small hours
(04:00–05:00). NO2 goes from about 150 µg/m³ at 19:00 to about 59 at 04:00.

**C2. What is the seasonal pattern?**
Nitrogen oxides are strongly seasonal: NO2 averages 139.91 µg/m³ in winter and
91.46 in summer. NOx nearly triples from summer to winter.

**C3. Does benzene follow the same seasonal shape?**
No. Benzene peaks in autumn (12.55 µg/m³) and is lowest in winter (8.72), which
differs from NO2. That is interesting because a single "air quality" narrative
would have predicted they move together.

**C4. How strongly are the pollutants correlated?**
CO and benzene are the strongest pair at r = 0.931. CO–NOx is 0.795, NOx–NO2 is
0.763. All correlations use pairwise-complete hours.

**C5. What did you discover about absolute humidity?**
It is essentially a deterministic function of temperature and relative humidity:
r = 0.99993 with the Magnus-formula product, and the scale factor varies by only
0.80 %. It adds almost no independent information.

**C6. Does AH track humidity or temperature more?**
Temperature, at r = 0.656, against only 0.168 for relative humidity. That corrects
a natural assumption that "absolute humidity is a humidity measure".

**C7. Why does that matter for the model?**
It means including AH alongside T and RH adds little; the model's weather tier is
effectively temperature and humidity with a rotation. It also warns against
interpreting AH coefficients causally.

**C8. How well does each sensor track its reference?**
`PT08.S2(NMHC)` is excellent at r = 0.982. `PT08.S1(CO)` is good at 0.879.
`PT08.S5(O3)` reaches 0.708 against NO2. `PT08.S3(NOx)` is negatively correlated at
−0.656 by design. `PT08.S4(NO2)` is poor at 0.158.

**C9. Why is `PT08.S3(NOx)` negatively correlated?**
The sensor's electrical response falls as NOx concentration rises, so the
relationship has a negative slope. The sign is a property of the device, not an
error in the data.

**C10. Which sensor is the weak link?**
`PT08.S4(NO2)`: r = 0.158 and R² = 0.025 against the NO2 reference. Whatever that
channel measures, it is not a good NO2 proxy in this record.

---

## D. Anomaly detection

**D1. How do you define an episode?**
A run of consecutive hours in which the pollutant is strictly above a chosen
threshold. Consecutive is unambiguous because the timeline has no gaps.

**D2. What is the default NO2 threshold and why?**
200 µg/m³, the European hourly limit value for NO2 (Directive 2008/50/EC). It is
clamped to the observed data so a narrow date range cannot make the widget invalid.

**D3. How many episodes are there at that threshold?**
137 episodes covering 388 exceedance hours, 5.03 % of the 7,715 valid NO2 readings.

**D4. How long is a typical episode?**
Two hours. The longest is 18 hours, from 12 February 2005 08:00 to 13 February
2005 01:00.

**D5. When is the peak reading?**
339.7 µg/m³, on 11 February 2005 at 16:00, inside a 15-hour episode.

**D6. Is the number of exceedances rising over time?**
Be careful here. 273 of the 388 exceedance hours fall in 2005, which is 70 %. But
2005 is only three and a half months of winter-heavy data. The correct statement is
that exceedances cluster in winter, and the winter of 2005 is unusually well
represented. It is not evidence that air quality deteriorated.

**D7. Why not report a simple "exceedances per year"?**
Because the two years are not comparable in length or season. 2004 has a full year;
2005 has about three and a half months, all of them winter. A per-year rate would
be misleading.

**D8. What does the duration distribution show?**
Most episodes are short; the distribution has a long right tail. The long episodes
matter more for exposure because people stay above the threshold for many hours.

---

## E. Sensor drift

**E1. What is drift?**
A slow change in a sensor's response over time, so that the same true concentration
produces a different reading later than it did earlier.

**E2. Why is it hard to test?**
Because the obvious test is confounded by season. If you compare spring 2004 with
spring 2005 you span a winter, and any seasonal effect that has not fully cancelled
looks like drift.

**E3. What are your four checks?**
Per-month re-calibration with a trend over months; matched same-calendar-month
comparison; a pairs-bootstrap confidence interval on the trend; and a sensitivity
reading of whether the response actually weakened.

**E4. Why all four?**
Because each alone is weak. A trend alone ignores season; a matched comparison
alone uses only two months; a confidence interval alone can be wide; and a slope
change alone does not say whether sensitivity was lost. Requiring agreement makes a
false positive much less likely.

**E5. What is a pairs bootstrap and why pairs?**
You resample the observed (sensor, reference) hour-pairs with replacement, keeping
each pair intact, and recompute the slope trend many times. If you resampled the
two columns independently you would destroy the very relationship you are
estimating.

**E6. What did you find?**
Only `PT08.S3(NOx)` shows evidence consistent with a temporal change: the monthly
slope trend is −0.0394, with a 95 % interval of [−0.0591, −0.0250], which excludes
zero, and both matched same-month comparisons agree with the trend.

**E7. What exactly changed for `PT08.S3(NOx)`?**
In the March comparison the slope moves from −0.263 to −0.681, the sensor's mean
reading falls from about 1,015 to 704 while the NOx reference mean rises from about
143 to 309 µg/m³, and the correlation weakens from −0.828 to −0.764.

**E8. So does that sensor drift?**
We say *evidence consistent with a temporal change in the sensor-reference
relationship*. We do not say "the sensor drifted", because the data cannot
distinguish a sensor change from an environmental, inlet, or reference-instrument
change.

**E9. What about the other four?**
Their slope-trend intervals all include zero, so the direction of any change is not
established. We report "no clear drift signal", not "no drift".

**E10. Why not say "no drift"?**
Because absence of evidence is not evidence of absence. With 14 monthly points and
two matched calendar months, the test has limited power. "No clear drift signal" is
the honest description of what the test can support.

**E11. What about `PT08.S4(NO2)`? Its correlation clearly fell.**
It did: from 0.835 to 0.506 in the same-month March comparison. But the slope trend
interval includes zero and only one of two matched comparisons agrees, so it does
not meet the threshold. We report the observation and say why it is not a verdict.
That contrast is exactly why the four-check rule exists.

**E12. Could there be drift you missed?**
Yes. A change too small or too gradual for 14 monthly points to resolve, or one
disguised by a coincident seasonal shift, would not be detected. That is a stated
limitation.

**E13. Is this a general claim about metal-oxide sensors?**
No. It is a statement about five sensor channels at one site over fourteen months.

---

## F. Benzene regression

**F1. What is the task?**
Estimate the reference benzene concentration from other signals available at the
same hour — a retrospective estimator, not a forecast.

**F2. Why benzene?**
It is toxic, it is measured by an expensive reference analyser, and the dataset
contains a cheap sensor that tracks it closely. That makes it the natural test case
for "can cheap instruments estimate what the expensive one measures?"

**F3. What are the two ways this task is usually cheated?**
Target leakage — letting the target or something derived from it enter the features
— and using a near-perfect proxy as a predictor, which measures the answer a second
time instead of estimating it.

**F3b. Which model is the headline, and what is the other one?**
The headline is the **sensor-only** model: weather, the four non-hydrocarbon
metal-oxide sensors, calendar features and the missing-count taken over those same
variables. Everything it uses is
available to a low-cost station, so its score is the low-cost sensor result. Beside
it we report a **reference-assisted** model that also reads `CO(GT)`, `NOx(GT)` and
`NO2(GT)` from the co-located certified analysers. That one is a comparison, not the
headline: it describes a station that already owns the expensive instruments, so it
must never be quoted as a low-cost sensor result.

**F4. Which column is the proxy?**
`PT08.S2(NMHC)`, at r = 0.982 with benzene and a Spearman ρ of 1.000, meaning it is
a monotone transformation of the target.

**F5. Why exclude it?**
Because including it would raise the apparent score without demonstrating that
benzene can be inferred from *other* information. We measure the cost of excluding
it rather than asserting it: R² falls from 0.9695 to 0.8404, and RMSE would be
56.3 % lower with it.

**F6. Why also exclude `NMHC(GT)`?**
The brief excludes it, and it is present for only 9.77 % of hours, so it could not
support a general model anyway.

**F7. Explain the chronological split.**
Train on the earlier 80 % of rows and test on the later 20 %. Training ends on
16 January 2005 at 01:00; testing runs from 02:00 that day to 4 April 2005.

**F8. Why not a random split?**
Air quality is highly autocorrelated. A random split puts adjacent hours on both
sides of the boundary, so the model can interpolate between neighbouring training
points. That produces a flattering score that says nothing about performance on a
genuinely unseen future hour.

**F9. What are the results?**
Held-out test period, sensor-only (headline): mean baseline R² = −0.1822, linear
regression 0.5146, Random Forest 0.8404. The forest's MAE is 2.2366 µg/m³ and RMSE
2.6029 µg/m³. The reference-assisted comparison reaches R² = 0.8580 (MAE 2.1020,
RMSE 2.4547) but needs the certified analysers, so it is not a low-cost figure.

**F10. Why is the baseline R² negative?**
Because it predicts the *training* mean, while R² is measured against the *test*
mean. The test period is lower in benzene than the training period, so a constant
based on training is systematically biased high and accumulates more error than
predicting the test mean. That is a negative R².

**F11. Isn't a negative R² just a bug?**
No, and it is informative. It shows the test period genuinely differs from the
training period, which is exactly what a random split would have hidden. It also
sets the bar: the forest's 0.840 is meaningful because the trivial approach scores
−0.182 on the same rows.

**F12. Why is the forest so much better than linear regression?**
Because the relationships are non-linear: sensor responses saturate, effects
interact, and the target is skewed. On the sensor-only predictors linear regression
captures about half of the variance (0.515); the forest captures about five-sixths
(0.840).

**F13. What do the predictor tiers show?**
Weather alone gives R² 0.3078. Adding the four cheap sensors raises it to 0.8404 —
that is the headline, the low-cost sensor case. Adding the reference analysers
raises it only to 0.8580, and that tier is labelled a reference-assisted comparison
because it needs the certified instruments. Most of the signal is already in the
inexpensive instruments.

**F14. How did you cross-validate?**
Five expanding-window, chronological folds on the training data only. The test
period is never touched during model selection. The forest's mean CV R² is
0.9086 ± 0.0296, close to the held-out 0.8404, which suggests the result is not a
fluke of one period.

**F15. What are the most important features?**
The four cheap sensors dominate: `PT08.S5(O3)` 0.244, `PT08.S1(CO)` 0.230,
`PT08.S3(NOx)` 0.189, `PT08.S4(NO2)` 0.176. Together the four sensors account for
about 0.84 of total importance in the sensor-only model.

**F16. Does importance mean causation?**
No. Importance measures how much a variable reduces prediction error when used to
split the data. It does not mean the variable causes benzene.

**F17. Is this a forecast?**
No. It uses same-hour readings from other instruments. A forecast would require
lagging every predictor and would score worse.

**F18. Can this model be used elsewhere?**
Not without re-fitting. It describes one site over fourteen months.

**F19. How would you know the model is not leaking?**
Target and proxy are excluded by name in code, and the exclusion is tested: the
feature matrix is verified not to contain the target, and scrambling the target does
not change the feature matrix. The leakage diagnostic then quantifies what the
exclusion costs.

**F20. What would you do with more time?**
Prospective validation on unseen future data, lag-based forecasting, prediction
intervals, and multi-site replication.

---

## G. Limitations, honesty and boundaries

**G1. What is the single most important limitation?**
Everything describes one site over fourteen months of historical data. It does not
generalise, and it is not live monitoring.

**G2. Is this an air-quality index?**
No. We deliberately do not compute an AQI. We report concentrations and episodes
against a stated threshold.

**G3. Would you call this a real-time system?**
No. It has no live feed. It reads a fixed historical file.

**G4. Is the missingness a problem for your conclusions?**
Yes, and we state how. Missingness comes in long device outages, so the observed
hours may differ systematically from the missing ones. Every statistic describes
the observed hours.

**G5. Can you claim the sensors drift?**
No. We can say one sensor channel shows evidence consistent with a temporal change
under a strict test, and that the other four do not have an established direction.

**G6. Can you claim air quality got worse in 2005?**
No. 2005 is winter-heavy and shorter. The exceedance clustering is seasonal.

**G7. What is the weakest part of the analysis?**
The drift test's statistical power, and the short test period for the model. Both
are stated rather than hidden.

**G8. If an examiner says "your model's R² is inflated", what do you say?**
The opposite concern applies: our split is chronological and our headline model
excludes the near-perfect proxy, and we report the 0.107 R² cost of that exclusion.
The baseline is negative, which shows the split is not flattering the model.

**G9. If an examiner says "your negative R² means your model is broken", what do you say?**
The negative R² belongs to the *mean baseline*, not the model. It is a consequence
of predicting the training mean on a test period with a different level. It is
expected, it is explained on the dashboard, and it is the evidence that the split
is honest.

**G10. What did you deliberately choose not to do?**
Forecast, compute an AQI, claim regulatory validity, generalise beyond the site,
or use the proxy sensor to inflate the score.

---

## H. Dashboard and engineering

**H1. How is the code organised?**
A library layer (`src/ecopulse`: loader, cleaning, quality, features, analysis,
drift, ml) with no Streamlit dependency, and an application layer (`app`: main,
theme, data, views) that only renders. The tests verify the library directly and
the pages through Streamlit's test harness.

**H2. Why `app/views/` and not `app/pages/`?**
Streamlit reserves a `pages/` directory next to the entry script for its own
multipage mechanism, which conflicted with our custom sidebar navigation.

**H3. How do you know every page works?**
Fifteen automated dashboard tests launch the real app and navigate to each page,
asserting no exception and no error element. They also exercise the date filter, an
empty range, and the threshold control.

**H4. How are expensive results handled?**
The cleaned frame, the quality report, the model benchmark and the leakage
diagnostic are cached. The model objects are computed lazily, so opening the
Overview page does not fit a Random Forest.

**H5. What happens if a page fails?**
The error is caught and shown inside the page with an explanation and the
traceback, instead of taking the whole app down.

**H6. What happens with an empty date range or a pollutant with no readings?**
The page shows an explanation and returns; it does not crash or draw an empty chart
without comment.

**H7. Give an example of a bug your tests caught.**
The threshold widget crashed when a narrowed date range put the NO2 default of 200
above the observed maximum. The control now clamps its default into the observed
span, and the fix is shared by the two pages that use it.

**H8. How many tests are there?**
281, across seven modules, covering loading, cleaning, features, analysis, drift,
modelling and the dashboard.

**H9. How do you stop the tests from passing on the wrong data?**
The scaffold tests assert the SHA-256 digests of the source files. If either file
changes, the suite fails.

**H10. Are the numbers hard-coded anywhere?**
No. The dashboard computes every figure from the cleaned frame. The tests
re-derive the headline figures, so a change in the analysis cannot quietly change a
claim.

---

## I. Mapping questions to the demo

| If asked about… | Show… |
| --- | --- |
| Data completeness | Data Quality page: before/after table and outage blocks |
| Diurnal or seasonal patterns | Pollution Explorer: hourly and monthly views |
| Sensor quality | Pollutant Relationships: sensor-versus-reference panels |
| Episodes | Anomaly Detection: threshold control and episode table |
| The model | Benzene Regression: model table, scatter, importances, tiers, leakage |
| Reproducibility | Methodology: live SHA-256 verification |
| Honesty of the split | Benzene Regression: the baseline R² note |

---

## J. Ten one-sentence answers to memorise

1. 9,357 hourly rows, March 2004 to April 2005, zero gaps and zero duplicates.
2. Missing values are `-200`; 16,701 of them, 13.73 % of cells, and they form four
   structural groups.
3. The device was off for 366 hours across 16 blocks, taking out nine columns at
   once.
4. NO2 averages 113.08 µg/m³, peaks at 19:00, and is highest in winter.
5. 388 hours exceed 200 µg/m³, in 137 episodes, the longest 18 hours.
6. Only `PT08.S3(NOx)` shows evidence consistent with a temporal change; the other
   four have no established direction.
7. The benzene model is trained on the earlier rows and tested on the later ones,
   because a random split would let it interpolate between neighbours.
8. The chronological split is why the mean baseline's R² is −0.182: the model
   predicts the training mean, but the later test period sits at a lower benzene
   level, so that constant is systematically biased.
9. `PT08.S2(NMHC)` is excluded because it correlates at r = 0.982 with the target;
   excluding it costs 0.129 R², which we report.
10. The sensor-only Random Forest reaches R² 0.840 and RMSE 2.60 µg/m³; the
    reference-assisted comparison reaches 0.858 but needs the certified analysers, so
    it is reported separately and never as a low-cost sensor result.
