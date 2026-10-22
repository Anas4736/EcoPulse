# EcoPulse — Project Report

**Air Quality Monitoring and Data Analysis System**

| | |
| --- | --- |
| **Project** | EcoPulse |
| **Dataset** | UCI Machine Learning Repository — *Air Quality* (ID 360) |
| **Team** | Mohd Anas (2026010138), Sumayah Lateef (2026010125) |
| **Report status** | Final for the implemented scope |
| **Evidence base** | Source hash verification, 281 automated tests, and results recomputed from the cleaned dataset at the time of writing |

> **How to read this report.** Section 13 separates *verified findings* (reproduced
> from the data by the implementation and re-checked by the test suite) from
> *exploratory observations* (patterns seen in the data that are not claims of
> cause) and from *limitations* (what the evidence cannot support). Numbers are
> given to the precision the pipeline produces them. Nothing in this report is
> hard-coded in the application: every figure is recomputed on load.

---

## 1. Abstract

EcoPulse is an end-to-end analysis and visualisation system built on a fixed
historical record from a single urban air-quality monitoring station. The record
covers 9,357 hourly observations between 10 March 2004 and 4 April 2005 and
contains both certified reference analyser readings and the raw responses of a
metal-oxide gas sensor array.

The project delivers five things: a reproducible cleaning pipeline, a data-quality
assessment, an exploratory analysis of pollutant behaviour, a controlled
investigation of whether the sensors drift over time, and a leakage-safe
regression model that estimates benzene from signals available at prediction
time. It wraps them in a seven-page Streamlit dashboard in which every statistic
is computed, not stored.

The headline results are as follows. The timestamp record is complete, with 9,357
expected hours, zero gaps and zero duplicates; all missingness is value-level and
takes only four distinct column groupings. Nitrogen dioxide shows a clear diurnal
cycle (peak at 19:00, trough at 04:00) and a clear seasonal cycle (winter mean
139.9 µg/m³ against summer 91.5). Of the five sensors examined, four show no clear
drift signal under a strict four-part test; one `PT08.S3(NOx)` shows evidence
consistent with a temporal change (monthly slope −0.0394, 95 % bootstrap interval
[−0.0591, −0.0250]). A Random Forest built from **low-cost sensors and weather
only** estimates benzene with a held-out R² of 0.8404 and RMSE of 2.603 µg/m³,
against a mean baseline whose R² is −0.182 — a negative value that is itself an
important result about the difficulty of the chronological split. The same model
extended with the co-located certified reference analysers reaches R² = 0.8580;
that figure is reported as a clearly labelled reference-assisted comparison, not
as a low-cost sensor result. Excluding the near-perfect `PT08.S2(NMHC)` proxy costs
0.129 in R², a cost that is measured rather than assumed.

The report is explicit about scope: this is a retrospective analysis of historical
data from one site, not a live monitoring service, and it makes no general claim
that the sensors in this dataset drift.

---

## 2. Problem statement

Urban air quality is measured with two very different kinds of instrument.
Reference analysers are accurate but expensive, bulky and need maintenance; they
are deployed at relatively few sites. Metal-oxide gas sensors are cheap and can be
deployed densely, but their readings drift, cross-react between gases, and depend
on temperature and humidity. A monitoring network that could trust cheap sensors
would have far better spatial coverage; one that cannot must either accept sparse
coverage or spend heavily on reference instruments.

This produces three concrete problems, all of which this project addresses on a
real dataset:

1. **Trusting the record.** Before any analysis, the dataset must be shown to be
   internally consistent. Missing values in this dataset are encoded as a sentinel
   (`-200`) rather than as blanks, so naive statistics are silently wrong. The
   timestamps must be shown to be complete and unique, not merely assumed to be.
2. **Separating signal from artefact.** Pollutant concentrations vary by hour,
   by season and by day of week. A sensor's apparent change over time may be a
   real change in its response, or may simply be the seasonal cycle. The project
   must control for season before saying anything about drift.
3. **Estimating what cannot be measured cheaply.** Benzene is toxic and is measured
   by a reference analyser. Can it be estimated from signals a cheap station could
   collect — and, crucially, can that estimate be evaluated honestly? Two standard
   mistakes make this task look easy when it is not: letting the target leak into
   the features, and using a second instrument that measures the same thing as a
   "predictor". Both are addressed explicitly rather than avoided silently.

A fourth, practical problem runs through all three: the results must be presented
in a form a reader can check. A number quoted in a slide is worth little if the
code that produced it cannot be re-run and if the source files cannot be shown to
be unmodified.

---

## 3. Objectives

| # | Objective | How it was met | Status |
| --- | --- | --- | --- |
| O1 | Ingest the original workbook without altering it | Read-only loader; SHA-256 verification of both source files | Met |
| O2 | Clean the data reproducibly and audit the result | Sentinel replacement, duplicate removal, continuity check, CSV export, quality report | Met |
| O3 | Quantify and characterise missingness | Per-column counts, co-missingness grouping, outage-block extraction | Met |
| O4 | Describe pollutant behaviour over time | Distributions, diurnal, seasonal and monthly profiles, correlations | Met |
| O5 | Test for sensor drift without confounding by season | Monthly re-calibration, matched same-month comparison, pairs bootstrap, sensitivity reading | Met |
| O6 | Build a leakage-safe benzene estimator | Chronological split, excluded target/proxy, three baseline models, cross-validation | Met |
| O7 | Present everything in an interactive dashboard | Seven Streamlit pages, all figures computed live | Met |
| O8 | Make every claim testable | 281 automated tests, including source-hash and dashboard smoke tests | Met |
| O9 | Write the project documentation | This report, a viva guide and a demo script | Met |

**Non-objectives.** The project does not forecast future concentrations, does not
compute an air-quality index, does not claim regulatory validity for the sensor
readings, and does not generalise its findings beyond this record. These are
deliberate boundaries, not omissions, and they are stated again in Section 14.

---

## 4. Dataset description

### 4.1 Provenance

The data is the **Air Quality** dataset from the UCI Machine Learning Repository
(ID 360), contributed by S. De Vito, E. Massera, M. Piga, L. Martinotto and
G. Di Francia. It was collected by a gas multi-sensor device deployed at a road
site in an Italian city, co-located with a certified reference analyser, and
released as a single Excel workbook. EcoPulse reads that workbook directly and
never writes to it.

Byte-level integrity is enforced in two places. The test suite asserts the
SHA-256 digest of both files in `Data/`, and the dashboard's *Methodology* page
recomputes the digests at runtime and displays whether they match:

| File | SHA-256 |
| --- | --- |
| `AirQualityUCI.xlsx` | `C92D102BB35CD4C0B42C0B6A1336C4065CCDCA7069F80A6890EE1B7E3124BD5D` |
| `Mohd_Anas_2026010138_Project_Synopsis.pdf` | `7E2A7F4D7C6A13EB9C75DF258A5246107F83B3CF14BE696E8E9C6245D8382A57` |

### 4.2 Shape and period

| Property | Value |
| --- | --- |
| Rows | 9,357 |
| Measured columns | 13 |
| Timestamp columns | `Date`, `Time` (combined into one hourly timestamp) |
| First observation | 2004-03-10 18:00 |
| Last observation | 2005-04-04 14:00 |
| Expected hourly rows | 9,357 |
| Missing hours in the timeline | 0 |
| Gaps | 0 |
| Duplicate timestamps | 0 |

The record is therefore a complete hourly series over the interval it claims to
cover. This matters because every later statement about "hours" is well defined:
the number of hours in a date range equals the number of rows in it.

### 4.3 Variables

The 13 measured columns fall into three groups.

**Reference analysers** (certified measurements): `CO(GT)` in mg/m³, `NMHC(GT)`
in µg/m³, `C6H6(GT)` in µg/m³, `NOx(GT)` in µg/m³, `NO2(GT)` in µg/m³.

**Metal-oxide sensors** (raw responses, arbitrary units): `PT08.S1(CO)`,
`PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`.

**Weather**: `T` (°C), `RH` (%), `AH` (absolute humidity).

### 4.4 The missing-value sentinel

Every missing measurement is recorded as `-200`. This is not a physical value and
must never enter a statistic, because it would drag means and correlations
arbitrarily. Across the whole sheet there are **16,701 sentinel markers**, equal
to **13.73 %** of all data cells. The loader converts every occurrence to a true
missing value, and the cleaning step confirms that no value of exactly `-200`
remains anywhere in any numeric column.

### 4.5 Completeness by column

| Column | Valid hours | % of 9,357 |
| --- | ---: | ---: |
| `CO(GT)` | 7,674 | 82.01 |
| `NOx(GT)` | 7,718 | 82.48 |
| `NO2(GT)` | 7,715 | 82.45 |
| `NMHC(GT)` | 914 | 9.77 |
| `C6H6(GT)` | 8,991 | 96.09 |
| `PT08.S1(CO)` … `PT08.S5(O3)` | 8,991 each | 96.09 |
| `T`, `RH`, `AH` | 8,991 each | 96.09 |

Completeness is strongly structured. The nine device/weather columns are missing
in exactly the same 366 hours, which is why they all share the 96.09 % figure.
`NMHC(GT)` is almost entirely absent and is not usable as a general variable.

---

## 5. Methodology

The project follows a five-stage pipeline. Each stage is a separate module with
its own tests, and the dashboard consumes the same modules the tests exercise, so
a page cannot show a number that the tests do not cover.

| Stage | Module | Responsibility |
| --- | --- | --- |
| 1. Load | `ecopulse.loader` | Read the workbook, combine date and time, parse to a tidy frame |
| 2. Clean | `ecopulse.cleaning` | Sentinel handling, deduplication, continuity check, CSV export |
| 3. Assess quality | `ecopulse.quality` | Missingness per column, co-missingness groups, outage blocks |
| 4. Analyse | `ecopulse.analysis`, `ecopulse.features`, `ecopulse.drift` | Profiles, correlations, calibration, episodes, drift |
| 5. Model | `ecopulse.ml` | Leakage-safe benzene regression and evaluation |

### 5.1 Design principles

**Single source of truth.** The workbook in `Data/` is the only input. The cleaned
CSV in `output/` is a derived artefact that can always be rebuilt.

**No hidden state.** Analysis functions take a frame and return plain objects
(DataFrames, dataclasses). The dashboard results are cached for speed but are
computed from the same functions; there is no separate "dashboard version" of an
analysis.

**Claims are testable.** Every quantitative statement in the README and this
report was produced by running the code, and the test suite re-derives the key
ones. If a change to the analysis alters a headline figure, a test fails.

**Honest evaluation over impressive numbers.** Where a modelling shortcut would
raise a score but invalidate it, the shortcut is refused and the cost of refusing
it is reported.

### 5.2 Tooling

Python 3.13 with pandas, NumPy, SciPy, scikit-learn, Plotly and Streamlit, and
pytest for verification. The package is installable (`pip install -e .`) so that
`ecopulse` imports cleanly from both the tests and the app.

---

## 6. Architecture

### 6.1 Layering

```
Data/ (read-only originals)
   │
   ▼
ecopulse.loader ──► ecopulse.cleaning ──► output/airquality_clean.csv
                          │
                          ▼
                   ecopulse.quality ──► quality report
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
 ecopulse.analysis  ecopulse.features  ecopulse.drift
        │                 │                  │
        └─────────────────┴──────────┬───────┘
                                     ▼
                              ecopulse.ml
                                     │
                                     ▼
                          app/ (Streamlit, 7 pages)
```

The library layer has no knowledge of Streamlit. The application layer has no
analysis logic of its own: it calls the library and renders the result. This
separation is what allows the test suite to verify the numbers without launching
a browser, and the dashboard tests to verify rendering without duplicating the
analysis.

### 6.2 Modules

| Module | Contents |
| --- | --- |
| `loader.py` | `load_raw`, source path resolution, timestamp assembly, `DataLoadError` |
| `cleaning.py` | `clean_frame`, `load_clean_dataset`, CSV export, command-line entry point |
| `quality.py` | `build_quality_report`, `summary_metrics`, `before_after_table`, `OutageBlock` |
| `features.py` | Time features, missing indicators, lag and rolling features, `rolling_name` |
| `analysis.py` | `describe_variables`, `pollutant_distribution`, `hourly_profile`, `daily_profile`, `monthly_aggregate`, `seasonal_aggregate`, `missingness_by_period`, `correlation_matrix`, `pairwise_counts`, `sensor_reference_summary`, `absolute_humidity_check`, `find_episodes`, `episode_summary` |
| `drift.py` | `assess_drift`, `assess_all_sensors`, `matched_period_comparison`, pairs bootstrap |
| `ml.py` | `build_ml_frame`, `chronological_split`, `build_model`, `cross_validate_chronologically`, `regression_metrics`, `run_benzene_benchmark`, `run_leakage_diagnostic`, `feature_importance_table` |

### 6.3 Application layer

`app/main.py` is the entry point. It sets the page configuration, draws the
sidebar (navigation via a custom radio control), renders the global filters the
selected page needs, applies them, constructs an `AppContext`, and calls the
page's `render` function.

Page modules live in `app/views/` rather than `app/pages/`. This is deliberate:
Streamlit reserves a `pages/` directory next to the entry script for its own
multipage mechanism, which would conflict with the custom sidebar navigation.

`AppContext` exposes the filtered frame, the unfiltered frame, the selected
pollutant and the date range, plus lazy properties for the quality report, the
model benchmark and the leakage diagnostic. The expensive objects are only
computed when a page actually asks for them, so opening the Overview page does not
fit a Random Forest.

Failure is contained. Import errors, data-load errors and rendering errors are
each caught and shown as an explanation inside the page, with the traceback made
available, rather than taking the whole application down.

### 6.4 State and caching

The cleaned frame and the three expensive analysis results are cached with
`st.cache_data`. The cache is keyed on the arguments, so changing the date filter
recomputes only the filtered frame while the model results (which do not depend
on the filter) are reused. No result is written to disk and no result is committed
to the repository.

---

## 7. Data preprocessing

### 7.1 Steps

1. **Read** the Excel sheet into a DataFrame.
2. **Assemble the timestamp** from the separate `Date` and `Time` columns into a
   single `datetime` column, with hours parsed correctly.
3. **Replace the sentinel.** Every `-200` in a measured column becomes `NaN`.
4. **Remove duplicates.** Exact duplicate rows and duplicate timestamps are
   removed (0 of each were found, but the step runs and is reported).
5. **Validate continuity.** The expected hourly index is generated from the first
   to the last timestamp and compared with the actual one.
6. **Reject non-physical values** where a column defines a valid range.
7. **Export** the cleaned frame to `output/airquality_clean.csv`.

### 7.2 Result of cleaning

| Check | Result |
| --- | --- |
| Rows read | 9,357 |
| Rows retained | 9,357 (100.00 %) |
| Duplicate rows removed | 0 |
| Duplicate timestamps removed | 0 |
| Missing hours | 0 |
| Gaps | 0 |
| Largest gap | 0 hours |
| Non-physical values found | 0 |
| Sentinel markers replaced | 16,701 (13.73 % of cells) |

Cleaning does not delete any row. Every row is a real hour; what changes is that
sentinel values become honest missing values. This is the correct treatment for
this dataset: the hours exist and carry information in the columns that were
recorded, even when some instruments were down.

### 7.3 Missing-value mechanism

Grouping columns by their exact pattern of missingness reveals that the missingness
is not random and not independent. There are exactly **four groups**:

| Group | Columns | Blocks | Total block-hours |
| --- | --- | ---: | ---: |
| Device outage | `PT08.S1(CO)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH` | 16 | 366 |
| Nitrogen oxides | `NOx(GT)`, `NO2(GT)` | 344 | 1,639 |
| Carbon monoxide | `CO(GT)` | 187 | 1,683 |
| Non-methane hydrocarbons | `NMHC(GT)` | 6 | 8,443 |

The four total **553 outage blocks**. The block-hour figures overlap in time
because different groups fail independently, so they must not be summed into a
single "hours missing" total.

Two consequences follow, and both are respected throughout the analysis:

- **The 16 device outages explain the 3.91 % floor** of missingness that affects
  all nine device columns. During those 366 hours the instrument was off, not
  merely recording a bad value.
- **`NMHC(GT)` is a special case.** It is present for only 914 hours across six
  blocks. Any statistic involving it describes those hours only, and it is
  excluded from the modelling features.

### 7.4 Imputation policy

No value is imputed before a descriptive statistic is computed. Means, medians and
correlations are computed over observed values, and correlation tables report the
number of shared hours behind each pair. Line charts break at gaps rather than
joining across them.

Imputation is used in exactly one place: the benzene model's predictor pipeline.
There, missing predictors are filled with the **median of the training data**, and
an indicator column is added for each imputed feature so the model can learn that
a value was filled rather than measured. Both steps happen **inside** the fitted
pipeline, so the imputation statistics are learned from training rows only and
never from the test period. Rows without a benzene target are dropped, because a
missing target cannot be trained on or scored against.

---

## 8. Exploratory analysis

> Everything in this section is descriptive. Patterns are reported as they appear
> in the data; no causal explanation is claimed, and the confound between season,
> weather and emissions is not resolved.

### 8.1 Distributions

| Column | Valid | Mean | Median | Min | Max | Skew |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CO(GT)` (mg/m³) | 7,674 | 2.153 | 1.80 | 0.10 | 11.90 | 1.37 |
| `NO2(GT)` (µg/m³) | 7,715 | 113.08 | 109.0 | 2.0 | 339.7 | 0.62 |
| `NOx(GT)` (µg/m³) | 7,718 | 246.88 | 179.8 | 2.0 | 1,479.0 | 1.72 |
| `C6H6(GT)` (µg/m³) | 8,991 | 10.08 | 8.24 | 0.149 | 63.74 | 1.36 |
| `NMHC(GT)` (µg/m³) | 914 | 218.81 | 150.0 | 7.0 | 1,189.0 | 1.56 |
| `T` (°C) | 8,991 | 18.32 | 17.75 | −1.9 | 44.6 | 0.31 |
| `RH` (%) | 8,991 | 49.23 | 49.55 | 9.18 | 88.73 | −0.04 |
| `AH` | 8,991 | 1.026 | 0.995 | 0.185 | 2.231 | 0.25 |

The pollutants are right-skewed, as expected for concentrations that spend most
hours at a moderate level and occasionally spike. `NOx(GT)` is the most skewed
(1.72); `NO2(GT)` the least among the pollutants (0.62). Temperature and relative
humidity are close to symmetric.

### 8.2 Absolute humidity is not an independent variable

`AH` is supplied as a separate column, but it is a deterministic function of
temperature and relative humidity. Fitting it gives

```
AH ≈ 0.09942 × (Magnus saturation vapour pressure at T) × RH / 100
```

with **r = 0.99993** across all 8,991 hours and a scale-factor coefficient of
variation of **0.80 %**. In other words, `AH` carries essentially no information
beyond `T` and `RH`.

A related check corrects a common assumption: `AH` correlates with **temperature
at r = 0.656** and with **relative humidity at only r = 0.168**. It is not simply
"a humidity measure"; it tracks temperature more strongly. This finding is encoded
as a test (`absolute_humidity_check`) so that any future change to the data or the
formula is caught.

### 8.3 Diurnal cycle

| Pollutant | Peak hour | Peak mean | Trough hour | Trough mean |
| --- | ---: | ---: | ---: | ---: |
| `NO2(GT)` | 19:00 | 150.08 µg/m³ | 04:00 | 58.60 µg/m³ |
| `CO(GT)` | 19:00 | 3.73 mg/m³ | 05:00 | 0.71 mg/m³ |
| `C6H6(GT)` | 19:00 | 17.74 µg/m³ | 04:00 | 2.92 µg/m³ |

All three pollutants share the same shape: a broad morning rise, a plateau through
the middle of the day, a pronounced evening maximum around 19:00, and a deep
minimum in the small hours. This is consistent with traffic emissions accumulating
under a shallow nocturnal boundary layer overnight, but the data alone does not
establish the mechanism, and traffic counts are not part of the record.

### 8.4 Seasonal cycle

| Pollutant | Winter | Spring | Summer | Autumn |
| --- | ---: | ---: | ---: | ---: |
| `NO2(GT)` (µg/m³) | **139.91** | 109.84 | 91.46 | 109.70 |
| `NOx(GT)` (µg/m³) | **359.68** | 189.53 | 111.41 | 358.97 |
| `CO(GT)` (mg/m³) | 2.321 | 2.093 | 1.686 | **2.517** |
| `C6H6(GT)` (µg/m³) | 8.72 | 9.62 | 9.42 | **12.55** |

Nitrogen oxides show the strongest and most orderly seasonal signal, roughly
tripling from summer to winter. This is an exploratory observation: heating,
mixing height and traffic all vary with season, and this dataset cannot separate
them. Note also that "winter" here means two partial winters (March 2004 and
December 2004–February 2005), not two full seasons.

### 8.5 Monthly means

`NO2(GT)` monthly means range from 70.10 µg/m³ (August 2004) to 160.63 µg/m³
(February 2005). The last three months in the record (January–March 2005) are all
high at 134.74, 160.63 and 137.01 µg/m³. `C6H6(GT)` follows a different shape,
peaking in October 2004 (13.52 µg/m³) and reaching its lowest monthly mean in
August 2004 (6.99 µg/m³). The two pollutants are strongly correlated overall but
do not share the same seasonal profile, which is why a single "air quality"
narrative would be misleading.

### 8.6 Correlations between reference pollutants

| Pair | Pearson r |
| --- | ---: |
| `CO(GT)` – `C6H6(GT)` | 0.931 |
| `CO(GT)` – `NOx(GT)` | 0.795 |
| `NOx(GT)` – `NO2(GT)` | 0.763 |
| `C6H6(GT)` – `NOx(GT)` | 0.719 |
| `CO(GT)` – `NO2(GT)` | 0.684 |
| `C6H6(GT)` – `NO2(GT)` | 0.615 |

All correlations are pairwise-complete and the shared-hour count is available for
each pair in the dashboard. The strongest relationship, CO–benzene at 0.931,
reflects a shared source (combustion) and is the reason CO appears prominently in
the benzene model.

### 8.7 Sensor versus reference calibration

| Sensor | Reference | Pairs | Pearson r | Spearman ρ | Slope | R² | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PT08.S1(CO)` | `CO(GT)` | 7,344 | 0.879 | 0.879 | 0.0058 | 0.773 | 0.684 |
| `PT08.S2(NMHC)` | `C6H6(GT)` | 8,991 | **0.982** | **1.000** | 0.0274 | 0.964 | 1.408 |
| `PT08.S3(NOx)` | `NOx(GT)` | 7,396 | −0.656 | −0.790 | −0.527 | 0.430 | 155.76 |
| `PT08.S4(NO2)` | `NO2(GT)` | 7,393 | 0.158 | 0.154 | 0.0212 | 0.025 | 47.02 |
| `PT08.S5(O3)` | `NO2(GT)` | 7,393 | 0.708 | 0.709 | 0.0831 | 0.502 | 33.61 |

Three things stand out.

- **`PT08.S2(NMHC)` is effectively a second benzene instrument** (r = 0.982, and a
  Spearman ρ of exactly 1.000, which indicates a monotone transformation). This is
  the single most important fact for the modelling section.
- **`PT08.S3(NOx)` is negatively correlated with NOx**, because this sensor's
  response falls as NOx rises. The negative sign is a property of the device, not
  an error.
- **`PT08.S4(NO2)` barely tracks NO2** (r = 0.158, R² = 0.025). Whatever this
  channel is measuring, it is not a good proxy for the NO2 reference in this
  record. This is reported rather than hidden because it is directly relevant to
  claims about sensor usefulness.

---

## 9. Dashboard pages

The application has seven pages. Every figure on every page is computed from the
cleaned frame when the page loads.

| # | Page | Purpose | Key content |
| --- | --- | --- | --- |
| 1 | **Overview** | Orientation and headline facts | Period, row count, completeness, headline patterns |
| 2 | **Pollution Explorer** | Time exploration of one pollutant | Hourly and daily series, user-set threshold, exceedance counts, distribution |
| 3 | **Pollutant Relationships** | How variables move together | Correlation matrix with sample counts, sensor-versus-reference panels |
| 4 | **Data Quality** | What is missing and why it matters | Before/after table, missingness by period, co-missingness groups, outage blocks |
| 5 | **Anomaly Detection** | Pollution episodes | Episode cards, threshold overlay, episode table, duration distribution |
| 6 | **Benzene Regression** | The model and its evidence | Sensor-only headline model, model comparison, predicted-versus-observed, residuals, importances, the labelled reference-assisted comparison, leakage cost |
| 7 | **Methodology & About** | Reproducibility | Live hash verification, column dictionary, missing-data and drift policies, limitations, credits |

**Global behaviour.** The sidebar carries navigation and the filters a page needs
(a date range and, where relevant, a pollutant). A historical-data notice is
present on every page. An empty date range, a pollutant with no readings, a
missing column, or a failed page produces an explanation rather than a crash.

**Engineering note.** The dashboard test suite (15 tests) caught a real defect
during development: when the date range was narrowed so that `NO2(GT)` never
reached 200 µg/m³, the default threshold exceeded the widget's maximum and raised
an exception. The threshold control now clamps its default into the observed span,
and the fix is shared by the two pages that use it.

---

## 10. Anomaly detection

An **episode** is a run of consecutive hours in which a pollutant stays strictly
above a user-chosen threshold. Because the record has no gaps, "consecutive hours"
is unambiguous: an episode of *n* hours spans exactly *n* rows.

### 10.1 Method

1. Convert the column to numeric; drop missing values from consideration (an
   unrecorded hour cannot be an exceedance).
2. Find runs where the value is above the threshold.
3. Join runs that are adjacent in time into a single episode.
4. For each episode, record start, end, duration, peak value, mean value and the
   time of the peak.
5. Summarise: count of exceedance hours, percentage of valid hours, number of
   episodes, longest and median episode length, peak value, and exceedance hours
   by calendar year.

The threshold is user-controlled. For `NO2(GT)` it defaults to **200 µg/m³**, the
European hourly limit value; for other pollutants it defaults to the 95th
percentile of the selected range. The default is clamped to the observed data.

### 10.2 Verified result at the NO2 limit value

| Measure | Value |
| --- | ---: |
| Valid NO2 readings | 7,715 |
| Hours above 200 µg/m³ | 388 |
| Share of valid hours | 5.03 % |
| Episodes | 137 |
| Longest episode | 18 hours |
| Median episode length | 2 hours |
| Highest reading | 339.7 µg/m³ |
| Exceedance hours in 2004 | 115 |
| Exceedance hours in 2005 | 273 |

The longest episode runs from **12 February 2005 08:00 to 13 February 2005 01:00**,
peaking at 264.6 µg/m³. The highest single reading, **339.7 µg/m³**, occurs on
**11 February 2005 at 16:00**, inside a separate 15-hour episode.

### 10.3 Interpretation, stated carefully

**70.4 % of the exceedance hours (273 of 388) fall in 2005**, even though 2005
contributes only about three and a half months of data. The high exceedance count
is concentrated in January–March 2005, which also have the highest monthly NO2
means in the record.

This is a verified fact about the data. It is *not* a demonstration of
deteriorating air quality: the 2005 portion of the record is winter-heavy, and
winter NO2 is higher for reasons this dataset cannot isolate. The correct reading
is that exceedances cluster in winter, and that the winter of 2005 is unusually
well represented relative to the rest of the year.

---

## 11. Benzene regression

### 11.1 Task

Estimate `C6H6(GT)`, the reference benzene concentration, from predictors that
would be available to a cheaper station. The model is a *retrospective estimator*:
it uses other instruments' readings from the **same hour** to estimate benzene at
that hour. It is not a forecast.

### 11.2 The two ways this task is usually cheated

**(a) Target leakage.** If the target, or anything derived from it, enters the
features, the model is scored on information it would not have in use. EcoPulse
excludes the target from the feature matrix and verifies the exclusion by
scrambling the target column and confirming the feature matrix does not change.

**(b) Using a proxy for the answer.** `PT08.S2(NMHC)` correlates with benzene at
**r = 0.982**. Adding it raises the score substantially, but the gain comes from
measuring the same quantity with a second instrument, not from estimating it. It
is therefore excluded from the headline model. Crucially, the cost of excluding it
is **measured** (Section 11.6) rather than asserted, so a reader can judge whether
the exclusion was worth it.

`NMHC(GT)` is also excluded: the brief excludes it, and it is present for only
9.77 % of hours, so it could not support a general model in any case.

### 11.3 The chronological split

The model is trained on the **earlier** part of the record and tested on the
**later** part.

| Property | Value |
| --- | --- |
| Rows with a benzene target | 8,991 |
| Rows dropped (no target) | 366 |
| Training rows | 7,192 |
| Test rows | 1,799 |
| Train ends | 2005-01-16 01:00 |
| Test begins | 2005-01-16 02:00 |
| Test ends | 2005-04-04 14:00 |

A random split would place neighbouring hours on both sides of the boundary. Since
air quality is highly autocorrelated, the model could then interpolate between
adjacent training points and earn a flattering score that says nothing about its
usefulness on an unseen future hour. The chronological split is the only split
that resembles actual use, and it is the reason one of the results below is
negative.

### 11.4 Models, features and evaluation

Three models see identical rows:

1. **Mean baseline** — predict the training-set average. This is the reference
   point: any model that cannot beat it has learned nothing.
2. **Linear regression** — ordinary least squares.
3. **Random Forest** — 300 trees, fixed seed, `max_features="sqrt"`.

All predictors pass through a pipeline of median imputation (with missing-value
indicators) and standardisation. Cross-validation uses five **expanding-window,
chronological** folds on the training data only; the test period is never touched
during model selection. Metrics are R², MAE and RMSE on the held-out test period,
with RMSE computed directly from the squared errors.

Three nested predictor sets are fitted to show what each group of instruments
contributes:

| Tier | Contents |
| --- | --- |
| **Weather** | `T`, `RH`, `AH` + calendar features + missing-count over those variables |
| **Sensors** (headline) | Weather + the four non-hydrocarbon metal-oxide sensors |
| **Reference-assisted** | Sensors + `CO(GT)`, `NOx(GT)`, `NO2(GT)` |

### 11.5 Sensor-only and reference-assisted: two different claims

The distinction between the last two tiers is not cosmetic, and it is the
central methodological point of this section.

The **sensor-only** model (tier *Sensors*) is the headline. Its predictors are
the three weather measurements, the four non-hydrocarbon metal-oxide sensor
responses, the calendar features and the missing-count taken over those same
variables. Every one of them is
available to a low-cost monitoring station, so the score it produces describes
the scenario the project is actually about: *can a cheap station estimate
benzene?*

The **reference-assisted** model (tier *Reference-assisted*) additionally reads
`CO(GT)`, `NOx(GT)` and `NO2(GT)` from the co-located certified reference
analysers. It is a perfectly valid model, but it is a different claim: it
describes a station that already owns the expensive instruments and is handed
their answers for CO, NOx and NO2. Its score is an upper bound, not a low-cost
sensor result.

Reporting only the reference-assisted number would overstate what cheap
hardware achieves; reporting only the sensor-only number would hide how much
the analysers contribute. EcoPulse reports both, with the sensor-only model as
the headline and the reference-assisted model explicitly labelled as a
comparison.

### 11.6 Verified results

**Headline — sensor-only predictors** (weather + the four non-hydrocarbon
sensors + calendar + missing-count; held-out test period):

| Model | R² | MAE (µg/m³) | RMSE (µg/m³) | CV R² (mean ± sd) |
| --- | ---: | ---: | ---: | ---: |
| Mean baseline | **−0.1822** | 6.0666 | 7.0836 | −0.0162 ± 0.0287 |
| Linear regression | 0.5146 | 4.1636 | 4.5390 | 0.8311 ± 0.1111 |
| Random Forest | **0.8404** | **2.2366** | **2.6029** | 0.9086 ± 0.0296 |

**Reference-assisted comparison** — the same pipeline, split and test hours,
with the certified analysers added to the predictors:

| Model | R² | MAE (µg/m³) | RMSE (µg/m³) | CV R² (mean ± sd) |
| --- | ---: | ---: | ---: | ---: |
| Mean baseline | **−0.1822** | 6.0666 | 7.0836 | −0.0162 ± 0.0287 |
| Linear regression | 0.5823 | 3.8770 | 4.2105 | 0.8494 ± 0.1001 |
| Random Forest | 0.8580 | 2.1020 | 2.4547 | 0.9277 ± 0.0231 |

**Predictor tiers (Random Forest):**

| Predictors | R² | MAE | RMSE | Role |
| --- | ---: | ---: | ---: | --- |
| Weather only | 0.3078 | 4.1794 | 5.4204 | context |
| Weather + sensors | 0.8404 | 2.2366 | 2.6029 | **headline (sensor-only)** |
| Weather + sensors + analysers | 0.8580 | 2.1020 | 2.4547 | comparison only |

**Leakage diagnostic** (adding the excluded proxy back into the sensor-only
model):

| Model | R² | RMSE |
| --- | ---: | ---: |
| Without `PT08.S2(NMHC)` (headline) | 0.8404 | 2.6029 |
| With `PT08.S2(NMHC)` | 0.9695 | 1.1382 |
| Difference | **+0.1291** | **−56.3 %** |

**Most important features (headline sensor-only Random Forest):**

| Feature | Importance |
| --- | ---: |
| `PT08.S5(O3)` | 0.2445 |
| `PT08.S1(CO)` | 0.2300 |
| `PT08.S3(NOx)` | 0.1893 |
| `PT08.S4(NO2)` | 0.1764 |
| `is_daytime` | 0.0290 |
| `hour_sin` | 0.0271 |
| `hour_cos` | 0.0176 |
| `RH` | 0.0158 |
| `doy_cos` | 0.0155 |
| `doy_sin` | 0.0151 |
| `AH` | 0.0147 |
| `is_weekend` | 0.0127 |
| `T` | 0.0123 |
| `missing_count` | 0.0000 |

The four cheap sensors account for 0.840 of the total importance between them,
and no importance at all is contributed by the target or by the excluded proxy.

### 11.7 Why the mean baseline scores a negative R²

This result deserves explanation because it looks like an error and is not one.

R² is defined as

```
R² = 1 − SSE / SST
```

where `SSE` is the sum of squared errors of the model and `SST` is the sum of
squared deviations of the **test-period** actual values from their **test-period**
mean. The mean baseline, however, predicts the mean of the **training** period.

The test period is 16 January to 4 April 2005. Benzene in the training period
averages higher than in the test period: the test period contains February and
March 2005, when monthly benzene means were 7.50 and 8.73 µg/m³, against 13.52
µg/m³ in October 2004. A model that predicts a constant that is systematically too
high accumulates more squared error than a constant equal to the test mean would.
Because `SST` is measured around the test mean, the baseline ends up **worse than
"predict nothing"**, and R² goes below zero.

Two lessons follow, and both are stated on the dashboard:

- **A negative baseline R² is evidence that the split is honest.** It shows the
  test period genuinely differs from the training period, which is exactly the
  situation a random split would have hidden.
- **It sets the bar the real models must clear.** The sensor-only Random Forest's
  0.840 is meaningful precisely because the trivial approach scores −0.182 on the
  same rows.

### 11.8 How to read the model comparison

The progression in the headline (sensor-only) table, from the baseline (−0.18)
through linear regression (0.52) to the Random Forest (0.84), shows that the
relationship between benzene and the cheap predictors is strongly **non-linear**:
linear regression captures a minority of the variance and leaves a large
residual, while the forest captures substantially more.

The tier comparison adds an operational insight. Weather alone explains 0.308 of
benzene variance; adding the four cheap sensors lifts this to 0.840. Adding the
reference analysers lifts it only to 0.858 — an increment of 0.018, and a
statement about a station that already owns those analysers rather than about
low-cost hardware. **Most of the benzene signal is already present in
inexpensive instruments.** The expensive analysers contribute a real but
comparatively small increment.

The leakage diagnostic quantifies the temptation. Adding back the excluded proxy
raises the sensor-only R² from 0.840 to 0.969 and cuts RMSE by more than half.
That is a large apparent improvement obtained by measuring benzene twice.
Reporting the excluded model, and reporting the size of the exclusion, is what
makes the choice auditable.

Finally, the cross-validation scores are close to the held-out scores (CV 0.909 ±
0.030 against test 0.840). This indicates the test result is not a fluke of one
particular period: the model performs consistently across time-ordered folds of the
training data, with a modest and expected degradation on the final, unseen period.

### 11.9 Scope of the estimate

The model estimates benzene at the **same hour** from other instruments at the
**same site**. It is not a forecast, it is not transferable to another location
without re-fitting, and its error would grow if the predictors were lagged into the
past. These boundaries are stated on the dashboard as well as in this report.

---

## 12. Sensor drift investigation

### 12.1 The problem with naive drift tests

Comparing a sensor's calibration at the start of the record with its calibration
at the end seems obvious and is wrong here. The record runs from March 2004 to
April 2005, so "start" is spring and "end" is spring — but the comparison also
spans the intervening winter, and any seasonal effect that has not fully cancelled
would be mistaken for drift. Worse, 2005 contributes only about three and a half
months, so a simple first-versus-last comparison is not even symmetric.

EcoPulse therefore applies **four checks simultaneously**, and requires all four to
agree before calling a change evidence of temporal change.

### 12.2 The four checks

1. **Per-month re-calibration.** The sensor-to-reference relationship is re-fitted
   independently for each of the 14 months, and the monthly slopes are regressed
   against time. A real change should produce a consistent direction.
2. **Matched same-calendar-month comparison.** Only months that occur in both years
   (March and April) are compared, so the season is held constant by construction.
3. **A pairs bootstrap confidence interval.** Hours that have both a sensor and a
   reference reading are resampled as **pairs**, preserving the sensor-reference
   link within each hour, and the slope trend is recomputed many times. The 95 %
   interval comes from this distribution, not from an assumed formula.
4. **A sensitivity reading.** A change is only interpreted as a loss of response
   when the sensor's sensitivity to the reference gas actually weakens. A shift in
   the intercept alone is not treated as the same phenomenon.

### 12.3 Verified verdicts

| Sensor → Reference | Pairs | Slope trend / month | 95 % interval | Matched months agreeing | Verdict |
| --- | ---: | ---: | --- | ---: | --- |
| `PT08.S1(CO)` → `CO(GT)` | 7,344 | +2.42 × 10⁻⁵ | [−4.90 × 10⁻⁵, +8.71 × 10⁻⁵] | 1 of 2 | No clear drift signal |
| `PT08.S2(NMHC)` → `C6H6(GT)` | 8,991 | −2.15 × 10⁻⁴ | [−4.67 × 10⁻⁴, +8.58 × 10⁻⁵] | 2 of 2 | No clear drift signal |
| `PT08.S3(NOx)` → `NOx(GT)` | 7,396 | **−0.0394** | **[−0.0591, −0.0250]** | 2 of 2 | **Evidence consistent with a temporal change** |
| `PT08.S4(NO2)` → `NO2(GT)` | 7,393 | +2.51 × 10⁻³ | [−1.92 × 10⁻³, +6.48 × 10⁻³] | 1 of 2 | No clear drift signal |
| `PT08.S5(O3)` → `NO2(GT)` | 7,393 | +1.31 × 10⁻³ | [−1.59 × 10⁻⁵, +2.31 × 10⁻³] | 1 of 2 | No clear drift signal |

**Only `PT08.S3(NOx)` clears all four checks.** Its confidence interval excludes
zero, both matched same-month comparisons move in the same direction as the trend,
and the slope becomes more strongly negative over time. In the March comparison,
the slope moves from −0.263 to −0.681 and the sensor's mean reading falls from
1,015 to 704 while the NOx reference mean rises from 143 to 309 µg/m³. The
sensor-reference correlation also weakens, from −0.828 to −0.764.

This is reported as *evidence consistent with a temporal change in the
sensor-reference relationship*. It is not a diagnosis. The data cannot distinguish
a physical sensor change from a change in the local environment, in the inlet, or
in the reference instrument, and the project does not claim otherwise.

For the other four sensors the interval includes zero, so **the direction of any
trend is not established**. That is a deliberately weaker statement than "there is
no drift": the analysis has not proven absence, only failed to find consistent
evidence of change.

### 12.4 A note on `PT08.S4(NO2)`

Although its verdict is "no clear drift signal", the same-month March comparison
shows its correlation with NO2 falling from 0.835 to 0.506 between 2004 and 2005.
This is worth flagging as an **exploratory observation**: the sensor's tracking of
NO2 appears weaker in the later March. However, the slope trend's confidence
interval includes zero and only one of the two matched comparisons agrees with the
trend, so it does not meet the project's evidence threshold. The contrast between
the two signals is exactly why the four-check rule exists: a single correlation
change would have been enough to make a claim, and that claim would have been
under-determined.

---

## 13. Results

This section separates what is **verified** from what is **exploratory** from what
the project explicitly **cannot** support.

### 13.1 Verified findings

| # | Finding | Evidence |
| --- | --- | --- |
| V1 | The timestamp record is complete and unique | 9,357 expected hours, 0 gaps, 0 duplicates, 100 % retention |
| V2 | Missingness is structured into exactly four column groups | Jaccard grouping over 13 columns |
| V3 | The 366-hour device outage affects nine columns identically | 16 blocks, identical missing pattern |
| V4 | Cleaning removes no rows and eliminates every sentinel | 16,701 markers (13.73 % of cells) replaced; 0 non-physical values |
| V5 | `AH` is a deterministic function of `T` and `RH` | r = 0.99993, scale CV = 0.80 %, n = 8,991 |
| V6 | `AH` tracks temperature more strongly than humidity | r = 0.656 with `T` vs 0.168 with `RH` |
| V7 | NO2, CO and benzene share an evening maximum and small-hours minimum | Peak 19:00, trough 04:00–05:00 in all three |
| V8 | Nitrogen oxides are strongly seasonal | NO2 winter 139.91 vs summer 91.46 µg/m³ |
| V9 | `PT08.S2(NMHC)` is a near-perfect benzene proxy | r = 0.982, R² = 0.964 |
| V10 | `PT08.S4(NO2)` barely tracks NO2 | r = 0.158, R² = 0.025 |
| V11 | 388 hours (5.03 % of valid) exceed 200 µg/m³ NO2 | 137 episodes; longest 18 h; peak 339.7 µg/m³ |
| V12 | 70.4 % of NO2 exceedance hours fall in 2005 | 273 of 388, concentrated in Jan–Mar 2005 |
| V13 | Only `PT08.S3(NOx)` shows evidence consistent with a temporal change | Slope trend −0.0394/month, CI [−0.0591, −0.0250], 2/2 matched months agree |
| V14 | The other four sensors show no clear drift signal | All four slope-trend intervals include zero |
| V15 | The **sensor-only** Random Forest estimates benzene with held-out R² = 0.840 | RMSE 2.6029 µg/m³, MAE 2.2366 µg/m³; headline model, cheap sensors and weather only |
| V16 | The mean baseline scores R² = −0.182 | Systematic train/test level difference under a chronological split |
| V17 | Weather alone explains R² = 0.308 of benzene variance | Random Forest, weather tier |
| V18 | Cheap sensors raise that to 0.840; adding the reference analysers reaches 0.858 | Sensors (headline) and reference-assisted tiers; 0.858 needs the certified analysers and is reported as a labelled comparison, not a low-cost sensor result |
| V19 | Excluding the proxy costs 0.129 R² and 56.3 % of RMSE | Leakage diagnostic on the sensor-only model |
| V20 | Source files are byte-identical to the build baseline | SHA-256 match, checked in tests and at runtime |

### 13.2 Exploratory observations

These are patterns visible in the data. They are not claims about cause and are
presented as such in the dashboard.

- The evening pollutant maximum is consistent with traffic emissions accumulating
  under a shallow nocturnal boundary layer. Traffic and mixing-height data are not
  in the record, so this remains a hypothesis.
- The winter NO2 maximum is consistent with heating and reduced mixing. The
  project cannot separate these from seasonal traffic changes.
- Benzene peaks in October rather than sharing the NO2 seasonal shape, suggesting
  that the two pollutants are not driven by an identical seasonal mixture.
- `PT08.S4(NO2)`'s correlation with NO2 falls between the same calendar months of
  the two years, even though its slope trend is not significant.
- The 2005 portion of the record is winter-heavy, so any whole-record average
  under-weights summer conditions relative to a balanced year.

### 13.3 What the results do **not** show

- They do **not** show general sensor drift. Four of five sensors have no
  established direction of change, and the one that does is a single variable at a
  single site.
- They do **not** show that the site's air quality deteriorated in 2005. The high
  2005 exceedance count reflects the seasonal composition of the record.
- They do **not** show that cheap sensors can replace reference analysers. The
  benzene model is retrospective and same-hour, and its target is one pollutant at
  one site. The reference-assisted score (R² = 0.858) describes a station that
  already owns the certified analysers; the low-cost sensor figure is the
  R² = 0.840 one, and it is the one quoted as the headline.
- They do **not** show that `PT08.S4(NO2)` is useless in general. They show that,
  in this record, it does not track the NO2 reference well.

---

## 14. Limitations

### 14.1 Data limitations

1. **One site, four seasons, fourteen months.** Every result describes one
   monitoring location. Nothing here generalises to another street, city, or year
   without re-running the analysis on that data.
2. **Historical data, not live monitoring.** The record ends on 4 April 2005.
   EcoPulse does not read a live feed, does not display current readings, and must
   not be used for health or safety decisions.
3. **Highly structured missingness.** Missingness is not random: it comes in long
   device outages. The missing hours may differ systematically from the observed
   hours, so every statistic describes the *observed* hours, not the whole period.
4. **`NMHC(GT)` is effectively unusable.** With only 9.77 % of hours present, any
   statistic involving it describes a small, non-representative subset.
5. **Partial final year.** The record contains a full year plus about one month, so
   "winter" is two partial winters and "annual" comparisons are not balanced.

### 14.2 Methodological limitations

6. **The drift test can fail to detect real drift.** A non-significant interval is
   not proof of stability; with 14 monthly points and only two matched calendar
   months, the test has limited power. The project therefore reports "no clear
   drift signal", never "no drift".
7. **Seasonal control is partial.** Only March and April occur in both years, so
   the matched comparison uses two months. The per-month trend model is a linear
   approximation over a short series.
8. **The bootstrap assumes exchangeability of hours.** Resampling pairs estimates
   the uncertainty of the slope trend under that assumption; serial correlation in
   the residuals is not modelled explicitly.
9. **The benzene model is retrospective.** It uses same-hour predictors. A genuine
   forecast would require every predictor to be lagged and would score worse.
10. **The test period is unusually hard.** It sits after the sensor sensitivity had
    begun to shift and after the training-period benzene level, which is why the
    mean baseline is negative. This is a property of the split, not a defect, but it
    means the reported R² should be read as performance on a difficult future
    period rather than a typical one.
11. **Feature importance is not causation.** Random Forest importances measure how
    much a variable reduces prediction error when used to split the data. They do
    not show that the variable causes benzene.
12. **Correlation is pairwise-complete.** Each correlation is computed over the
    hours where both variables exist, so different pairs rest on different samples.

### 14.3 Presentation limitations

13. **No screenshots are included in this report.** Every figure is available live
    in the dashboard and is recomputed on load, which is stronger than a static
    image but means the report cannot show a frozen picture.
14. **Numbers can shift in the last digit** across library versions. The models use
    a fixed random seed, so repeated runs on the same versions reproduce the same
    scores.

---

## 15. Conclusion

EcoPulse set out to answer three questions on a real, imperfect dataset and to
answer them in a way a reader could check.

**Can the record be trusted?** Yes, within clear limits. The timeline is complete
and unique, cleaning removes no rows, and all missingness is value-level and
structured into four explainable groups. The cost of that structure is that a large
fraction of hours lacks one or more instruments, so every statistic describes the
observed hours.

**Do the sensors drift?** Not in any general sense that this record can establish.
Four of five sensors show no consistent direction of change under a test that
controls for season and quantifies uncertainty. One `PT08.S3(NOx)` shows evidence
consistent with a temporal change, and the project reports exactly that and no
more.

**Can benzene be estimated cheaply?** Yes, with a held-out R² of 0.840 and an RMSE
of 2.60 µg/m³ from low-cost sensors and weather alone, and with a clear
understanding of why those numbers are credible. The estimate survives a
chronological split, beats a linear model, and is evaluated honestly: the target
and the near-perfect proxy are excluded, and the cost of excluding the proxy is
measured at 0.129 R². The negative baseline R² is not a defect to be hidden but
the evidence that the split is doing its job. The score that also uses the
co-located reference analysers (R² = 0.858) is reported separately and labelled,
because it describes a well-equipped station rather than a cheap one.

The larger contribution is methodological. The project demonstrates, on a real
dataset, that a careful analysis can be more informative than an impressive score:
that controlling for season can turn a dramatic-looking "drift" into an
inconclusive result, that excluding a helpful feature can be the right choice when
that feature measures the answer, and that reporting a negative baseline is more
honest than quietly choosing a split that makes it positive.

---

## 16. Future scope

The following are natural extensions. None is implemented, and none is claimed.

1. **Live ingestion.** Connect the same pipeline to a current feed so the dashboard
   becomes a monitoring tool rather than a historical explorer. This would require
   handling real-time gaps, late arrivals and changing sensor calibration.
2. **Multi-site analysis.** Apply the drift test across several stations to
   distinguish site-specific artefacts from instrument-specific behaviour.
3. **Prospective validation of the benzene model.** Log predictions on an unseen
   future period and compare them with reference readings as they arrive, which is a
   stronger test than a retrospective split.
4. **Lag-based forecasting.** Build a genuinely predictive model using only past
   information, and report its lower but more operationally relevant score.
5. **Calibration correction.** If a sensor's response is shown to change, apply a
   time-varying calibration and measure whether the correction improves downstream
   estimates.
6. **Uncertainty quantification for the regression.** Report prediction intervals
   alongside point estimates so a user can see how confident the model is on a given
   hour.
7. **Deployment.** Host the dashboard behind an authenticated endpoint with a
   reproducible environment and automated rebuild of the cleaned data. The project
   is currently local-only.

---

## 17. References

1. UCI Machine Learning Repository. *Air Quality* dataset (ID 360). Contributed by
   S. De Vito, E. Massera, M. Piga, L. Martinotto and G. Di Francia. Available at
   <https://archive.ics.uci.edu/dataset/360/air+quality>.
2. De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008).
   *On field calibration of an electronic nose for benzene estimation in an urban
   pollution monitoring scenario.* Sensors and Actuators B: Chemical, 129(2),
   750–757. (The publication associated with the dataset.)
3. Directive 2008/50/EC of the European Parliament and of the Council of 21 May 2008
   on ambient air quality and cleaner air for Europe. (Source of the 200 µg/m³
   hourly NO₂ limit value used as the default episode threshold.)
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
   Journal of Machine Learning Research, 12, 2825–2830.
5. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.*
   Proceedings of the 9th Python in Science Conference, 56–61.
6. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.*
   Chapman & Hall. (Basis for the pairs bootstrap used in the drift test.)
7. Streamlit documentation. <https://docs.streamlit.io>
8. Plotly documentation. <https://plotly.com/python/>
9. pandas documentation. <https://pandas.pydata.org/docs/>

---

## Appendix A — Reproducing the results

From the project root on Windows:

```powershell
python -m pip install -e .
python -m ecopulse.cleaning          # rebuild output/airquality_clean.csv
python -m pytest -q                  # 281 tests
python -m streamlit run app/main.py  # dashboard at http://localhost:8501
```

The dashboard's *Methodology & About* page recomputes the two source-file SHA-256
digests and shows whether they match the expected values, so the provenance of the
figures can be checked without reading the test code.

## Appendix B — Test coverage summary

| Test module | Focus |
| --- | --- |
| `test_phase0_scaffold.py` | Project layout and source-hash tamper guard |
| `test_phase1_cleaning.py` | Sentinel handling, deduplication, continuity, CSV export |
| `test_phase2_features.py` | Time, lag, rolling and missing-indicator features |
| `test_phase2_analysis.py` | Profiles, correlations, calibration, absolute humidity, episodes |
| `test_phase2_drift.py` | Monthly trend, matched comparison, pairs bootstrap, verdict logic |
| `test_phase3_ml.py` | Feature exclusion, chronological split, metrics, leakage diagnostic |
| `test_phase4_app.py` | Every dashboard page renders; filters and empty ranges behave |

Total: **281 tests** at the time of writing.
