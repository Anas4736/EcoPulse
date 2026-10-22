# EcoPulse

**Air Quality Monitoring and Data Analysis System**

An interactive Streamlit dashboard for exploring and modelling the
[UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
(UCI Machine Learning Repository, Dataset ID 360).

> **This is a historical dataset analysis, not a live monitoring system.**
> The source readings were recorded between March 2004 and April 2005 by a
> single air-quality sensor device. The dashboard analyses that archive; it does
> not receive, display or imply any live sensor feed.

---

## Team

| Name | Roll Number |
|---|---|
| Mohd Anas | 2026010138 |
| Sumayah Lateef | 2026010125 |

### Contributions

<!-- EDITABLE: replace the placeholder bullets below with your own descriptions
     of what each member did. Please keep them accurate. -->

- **Mohd Anas (2026010138)** — _TODO: describe your contributions, e.g. data cleaning, feature engineering, specific pages of the dashboard, parts of the machine-learning experiment._
- **Sumayah Lateef (2026010125)** — _TODO: describe your contributions, e.g. exploratory analysis and charts, the anomaly detection page, documentation, testing, and the project report._

---

## The dataset

| | |
|---|---|
| Source | UCI Machine Learning Repository, Dataset ID 360 |
| Citation | Vito, S. (2008). *Air Quality* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C59K5F> |
| Original paper | Vito, S. et al. (2008). "On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario." *Sensors and Actuators B*, 129(2). |
| Device | 5 metal-oxide sensors in a chemical multisensor, deployed at road level in a significantly polluted area of an Italian city |
| Ground truth | Co-located certified reference analyzer for CO, NMHC, Benzene, NOx and NO2 |
| Period | 2004-03-10 18:00 to 2005-04-04 14:00 (hourly) |
| Rows | 9,357 hourly observations |
| Columns | 15 = 13 measured variables + `Date` + `Time` |
| Missing values | Tagged with `-200` (never a real reading) |

**No particulate matter (PM2.5 / PM10) is present in this dataset**, so a
composite Air Quality Index cannot be computed. Anything in the dashboard that
looks like a "pollution level" is a pollutant-specific concentration or a
clearly labelled user-defined threshold. See `PROJECT_REPORT.md` for details.

### Licence note

The UCI dataset page lists the dataset under **CC BY 4.0**, while the dataset
description states that it may be used for **research purposes only and that
commercial use is excluded**. Both are noted here and in the report.

---

## Project structure

```
EcoPulse/
├── Data/                       # ORIGINAL dataset - read only, never modified
│   ├── AirQualityUCI.xlsx
│   └── Mohd_Anas_2026010138_Project_Synopsis.pdf
├── output/                     # generated artifacts
│   ├── airquality_clean.csv    # cleaned dataset (deliverable)
│   └── models/                 # cached scikit-learn models
├── src/ecopulse/               # the analysis package (importable, testable)
│   ├── config.py               # paths, constants, column registry
│   ├── loader.py               # read the workbook into a tidy DataFrame
│   ├── cleaning.py             # -200 handling, timeline validation, CSV export
│   ├── quality.py              # data-quality report (before/after, outage blocks)
│   ├── features.py             # time / lag / rolling feature engineering
│   ├── analysis.py             # exploratory statistics
│   ├── drift.py                # sensor-drift investigation with seasonal control
│   └── ml.py                   # leakage-safe benzene regression and evaluation
├── app/
│   ├── main.py                 # Streamlit entry point and navigation
│   ├── theme.py                # colours, CSS and shared UI helpers
│   ├── data.py                 # cached access to the frame and analysis results
│   ├── static/                 # favicon
│   └── views/                  # one module per dashboard page
│       ├── overview.py
│       ├── pollution_explorer.py
│       ├── relationships.py
│       ├── data_quality.py
│       ├── anomaly_detection.py
│       ├── benzene_regression.py
│       └── methodology.py
├── tests/                      # pytest suite
│   ├── conftest.py
│   ├── test_phase0_scaffold.py
│   ├── test_phase1_cleaning.py
│   ├── test_phase2_features.py
│   ├── test_phase2_analysis.py
│   ├── test_phase2_drift.py
│   ├── test_phase3_ml.py
│   └── test_phase4_app.py
├── docs/                       # viva guide and demo script
│   ├── VIVA_PREPARATION.md
│   └── DEMO_SCRIPT.md
├── requirements.txt
├── PROJECT_REPORT.md           # full project report
└── run_app.bat / run_app.sh
```

The dashboard is organised under `app/views/` rather than `app/pages/` because
Streamlit reserves a `pages/` directory next to the entry script for its own
multipage mechanism, which would conflict with the custom sidebar navigation.

The written deliverables — the full project report and the viva/demo guides — are
complete, and the analysis, models and dashboard they describe all exist and are
tested.

---

## Setup and running

### Windows

```bat
run_app.bat
```

### macOS / Linux

```bash
chmod +x run_app.sh
./run_app.sh
```

### Manual setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
pip install -e .              # makes the `ecopulse` package importable
python -m streamlit run app/main.py
```

The dashboard is then served at <http://localhost:8501>.

### Rebuilding the cleaned dataset

The cleaning pipeline reads `Data/AirQualityUCI.xlsx` (never writes to it) and
produces `output/airquality_clean.csv` together with a console report:

```bash
python -m ecopulse.cleaning
```

It prints the row counts, the before/after missing-value comparison and the
groups of columns that fail together.

### Running the tests

```bash
pytest -q
```

The suite currently has **281 tests**, including a smoke test that launches the
real dashboard and navigates to every page.

---

## The dashboard

EcoPulse is a seven-page Streamlit application. Every figure on screen is
computed when the page loads; there are no stored results. The sidebar carries
navigation, a date-range filter and a pollutant selector, and a historical-data
notice is always present.

| Page | What it shows |
| --- | --- |
| **Overview** | The dataset at a glance: period, row count, completeness, and the headline patterns. |
| **Pollution Explorer** | Hourly and daily series for any reference pollutant, with a user-set threshold and exceedance counts. |
| **Pollutant Relationships** | The correlation structure, pairwise sample counts, and how each metal-oxide sensor tracks its reference analyser. |
| **Data Quality** | Missingness before and after cleaning, the groups of columns that fail together, and every outage block. |
| **Anomaly Detection** | Episodes above a threshold: how many, how long, how high, and where they sit in the record. |
| **Benzene Regression** | The leakage-safe benzene model: the **sensor-only** headline model, model comparison, observed-versus-predicted, feature importance, the labelled reference-assisted comparison and the measured cost of excluding the proxy sensor. |
| **Methodology & About** | Provenance with live SHA-256 verification, the full column dictionary, the missing-data and drift policies, the modelling method, limitations and credits. |

A failed page is caught and reported in the interface rather than taking the
whole app down, and an empty date range or a pollutant with no readings produces
an explanation instead of a crash.

---

## Verified findings

Every number below is computed at runtime by `ecopulse.analysis`,
`ecopulse.drift` and `ecopulse.ml` from the cleaned dataset. Nothing is
hard-coded, and the test suite re-derives the headline figures so a change in
the analysis cannot quietly change a claim.

### The dataset

- 9,357 hourly observations, 13 measured variables, **no gaps and no
  duplicate timestamps** in the timeline.
- `-200` is the missing marker. It is never a real reading.
- Valid readings: `C6H6(GT)` 8,991 (96.1%); `CO(GT)` 7,674 (82.0%);
  `NO2(GT)` 7,715 (82.5%); `NOx(GT)` 7,718 (82.5%); `NMHC(GT)` only 914
  (9.8%) — `NMHC(GT)` is therefore described but kept out of the models.
- Missing values arrive in **four distinct groups**, which means the gaps are
  instrument behaviour and not random noise:
  1. nine columns together (16 outage blocks, 366 hours) — the device was off;
  2. `NOx(GT)` and `NO2(GT)` together (near-identical patterns);
  3. `CO(GT)` alone;
  4. `NMHC(GT)` alone.

### Pollution patterns

- **Diurnal cycle**: mean `NO2(GT)` is lowest at 04:00–05:00 (about 59 µg/m³)
  and highest at 19:00 (about 150 µg/m³), a traffic-driven rush-hour peak.
- **Seasonal cycle**: winter is the most polluted season (about 140 µg/m³
  mean `NO2(GT)`) and summer the least (about 91 µg/m³).
- **Threshold exceedances**: 388 hours above 200 µg/m³ of `NO2(GT)`
  (5.0% of valid readings) in 137 separate episodes, with 273 of those hours
  in 2005 against 115 in 2004. The peak reading is 339.7 µg/m³.
- **Relationships**: `C6H6(GT)` and the NMHC sensor response track each
  other almost perfectly (Pearson r ≈ 0.98, Spearman rho ≈ 1.00 on 8,991
  paired hours); `CO(GT)` and the CO sensor reach r ≈ 0.88. The `NO2(GT)`
  sensor is the weakest of the five (r ≈ 0.16) and the NOx sensor is
  *negatively* correlated with the reference (r ≈ −0.66), which the sensor
  literature explains by its tungsten oxide sensing chemistry.
- `AH` is not in a standard unit of absolute humidity. It is the Magnus
  saturation vapour pressure scaled by relative humidity and multiplied by a
  constant (Pearson r ≈ 0.99993 on 8,991 paired hours, scale factor 0.099
  with a coefficient of variation of 0.8%), so it adds almost no
  independent information beyond `T` and `RH`.

### Sensor drift

The UCI documentation states that concept and sensor drifts are present.
That claim is checked here rather than repeated, because three confounds
could easily imitate it: seasonality, the outage pattern, and the fact that
the record starts in March 2004 and ends in early April 2005.

The method recalibrates each sensor against its reference analyser **per
calendar month**, then compares **the same calendar month in different
years** with both years trimmed to the days they actually share (March and
April are the only two months present in both years, and April 2005 covers
only four days). Trends are quantified with a pairs-bootstrap confidence
interval.

| Sensor | Reference | Verdict |
|---|---|---|
| `PT08.S1(CO)` | `CO(GT)` | no clear drift signal |
| `PT08.S2(NMHC)` | `C6H6(GT)` | no clear drift signal |
| `PT08.S3(NOx)` | `NOx(GT)` | **evidence consistent with a temporal change** |
| `PT08.S4(NO2)` | `NO2(GT)` | no clear drift signal |
| `PT08.S5(O3)` | `NO2(GT)` | no clear drift signal |

Only the **NOx sensor** shows a change that survives the seasonal control:
its calibration slope moves by −0.039 per month with a 95% bootstrap
interval of [−0.059, −0.025], which excludes zero, and both matched
same-season comparisons agree in direction. The practical reading is that
the NOx sensor became *less* sensitive over the record, so the same raw
sensor value corresponds to a larger true concentration later on.

A naive "first 60 days versus last 60 days" comparison of the CO sensor
does suggest a change: the sensor-to-reference correlation weakens from
r = 0.92 to r = 0.84, and the reference concentration falls faster
(−14.7%) than the raw sensor reading does (−6.6%). The calibration *slope*
itself barely moves (+2.5%), though, and the month-by-month comparison with
a bootstrap interval does not confirm the change. The dashboard therefore
reports **no clear drift signal** for the CO sensor, and the same caution
applies to the four other sensors.

### Benzene regression

`ecopulse.ml` estimates the reference benzene concentration `C6H6(GT)` from
signals that are genuinely available at prediction time. Three rules keep the
result honest:

- **The target never becomes a feature**, and neither does anything computed
  from it. The test suite scrambles the target column and checks that the
  feature matrix comes back byte-for-byte identical.
- **`PT08.S2(NMHC)` is excluded.** It is the cheap titania sensor for
  non-methane hydrocarbons, and it correlates at **r = 0.982** with the
  benzene reference. Using it would be legal arithmetic but useless science:
  the model would be reading the answer off another instrument. The cost of
  the exclusion is measured rather than assumed (see below).
- **The split is chronological.** The training set is the *earlier* part of
  the record and the test set is the *later* part, so the model is scored on
  the future. Cross-validation uses `TimeSeriesSplit`, which is chronological
  too.

#### Two models, two questions

The benzene model is reported twice, and the two results are **not**
interchangeable:

| Model | Predictors | What it describes |
|---|---|---|
| **Sensor-only (headline)** | weather, the four non-hydrocarbon metal-oxide sensors, calendar features, missing-count over those variables | the **low-cost sensor scenario**: every predictor is available to a cheap monitoring station |
| Reference-assisted (comparison) | the sensor-only set **plus** `CO(GT)`, `NOx(GT)`, `NO2(GT)` from the co-located certified reference analysers | a station that **already owns the expensive analysers** — it is handed their answers for CO, NOx and NO2 |

The reference-assisted score is a legitimate upper bound, but it is **not** a
low-cost sensor result and is never presented as one. The headline number
below is the sensor-only one.

Of the 9,357 hours, 8,991 have a benzene reading; those are split 7,192
train (2004-03-10 18:00 → 2005-01-16 01:00) and 1,799 test (2005-01-16 02:00
→ 2005-04-04 14:00).

**Headline — sensor-only predictors** (weather + the four non-hydrocarbon
sensors + calendar + missing-count over those variables):

| Model | R² | MAE (µg/m³) | RMSE (µg/m³) | CV R² (mean ± sd) |
|---|---|---|---|---|
| Mean baseline (training average) | **−0.182** | 6.067 | 7.084 | −0.016 ± 0.029 |
| Linear regression | 0.515 | 4.164 | 4.539 | 0.831 ± 0.111 |
| **Random Forest** | **0.840** | **2.237** | **2.603** | 0.909 ± 0.030 |

The **negative** R² of the mean baseline is the point of the chronological
split. The average benzene level in the later test period is different from
the training average, so predicting the training mean is worse than useless
there. A random split would have mixed both periods and hidden this.

**Reference-assisted comparison** — the same pipeline, the same split, the
same test hours, with the reference analysers added to the predictors:

| Model | R² | MAE (µg/m³) | RMSE (µg/m³) | CV R² (mean ± sd) |
|---|---|---|---|---|
| Linear regression | 0.582 | 3.877 | 4.210 | 0.849 ± 0.100 |
| Random Forest | 0.858 | 2.102 | 2.455 | 0.928 ± 0.023 |

How much each group of predictors contributes (Random Forest, same split):

| Feature tier | Predictors | R² | Role |
|---|---|---|---|
| Weather only | `T`, `RH`, `AH` + calendar | 0.308 | context |
| **Sensors only (headline)** | the four non-hydrocarbon sensors + weather | **0.840** | the low-cost sensor scenario |
| Reference-assisted | sensors + `CO(GT)`, `NOx(GT)`, `NO2(GT)` | 0.858 | comparison only |

The middle row is the operationally interesting one: the four cheap
metal-oxide sensors and the weather alone reach R² = 0.840, so adding the
co-located analysers lifts the score by only 0.018 on top of what the
sensors already give. Most of the benzene signal is present in inexpensive
instruments.

**Measured cost of the exclusion.** Re-fitting the sensor-only Random Forest
with `PT08.S2(NMHC)` added raises R² from 0.840 to **0.969** and cuts RMSE from
2.60 to 1.14 (56.3 % lower). That is exactly why the column is kept out of the
headline model: the extra accuracy is bought by measuring the same quantity
twice, not by predicting better.

These figures are the output of `run_benzene_benchmark` and
`run_leakage_diagnostic` on the real data. The reported metrics are computed
by fitting the models and scoring them on held-out data; none of them is a
hard-coded number.

---

## Documentation

| File | Purpose |
|---|---|
| `README.md` | This file: setup, structure, dataset provenance, verified findings |
| `PROJECT_REPORT.md` | Full project report: problem, data, method, results, limitations |
| `docs/VIVA_PREPARATION.md` | Question-and-answer guide for the viva |
| `docs/DEMO_SCRIPT.md` | Step-by-step script for the presentation/demo |

---

## Deployment status

The application is **not deployed to any public host** at the time of writing.
It runs locally with the commands above. A deployment section will be added
here only once a deployment has actually been performed.

### Current build status

| Layer | Status |
| --- | --- |
| Data loading and cleaning | Complete, tested |
| Data-quality report | Complete, tested |
| Feature engineering | Complete, tested |
| Exploratory analysis | Complete, tested |
| Sensor-drift investigation | Complete, tested |
| Benzene regression | Complete, tested |
| Streamlit dashboard (7 pages) | Complete, tested, runs locally |
| `PROJECT_REPORT.md`, `docs/VIVA_PREPARATION.md`, `docs/DEMO_SCRIPT.md` | Complete |
