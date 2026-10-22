# EcoPulse — Demo Script

A step-by-step plan for presenting the dashboard. Two timings are provided: a
**5-minute** version and a **10-minute** version. Every figure below is computed
live by the application; if a number on screen differs from this script, **say what
is on screen** — the dashboard is the source of truth, not this document.

---

## 1. Before the demo (do this 30 minutes early)

### 1.1 Prepare the project

Open PowerShell in the project folder:

```powershell
cd C:\Users\ASUS\Desktop\EcoPulse
```

Install the package once so the library and tests import cleanly:

```powershell
python -m pip install -e .
```

Rebuild the cleaned dataset so `output/airquality_clean.csv` is current:

```powershell
python -m ecopulse.cleaning
```

You should see a report ending with 9,357 rows read, 9,357 rows after cleaning,
0 rows dropped, and a list of the four missingness groups.

### 1.2 Prove the project is healthy

```powershell
python -m pytest -q
```

Expected: **281 passed**. Do not present without running this once beforehand. If
it reports a failure, fix it before the demo rather than hiding it.

### 1.3 Launch the dashboard

Either use the launcher:

```powershell
.\run_app.bat
```

or start Streamlit directly:

```powershell
python -m streamlit run app/main.py
```

### 1.4 Open the URL

**<http://localhost:8501>**

Leave the browser tab open and loaded before you begin. The first page load fits
nothing expensive, but the Benzene Regression page computes three models on first
visit, so **open that page once beforehand** to warm the cache, then return to the
Overview page.

### 1.5 Have these ready

- This script, open on a second screen or printed.
- The **Overview** page showing in the browser.
- A terminal window with the test result visible, in case you are asked.
- `PROJECT_REPORT.md` and `docs/VIVA_PREPARATION.md` open in an editor for
  reference.

### 1.6 Fallback plan

If the dashboard will not start during the demo:

1. Do not restart repeatedly in front of the audience.
2. Switch to `PROJECT_REPORT.md` and walk through the findings table.
3. Show the terminal output of `python -m pytest -q` as evidence the analysis
   works.
4. Explain that the app is a presentation layer over the same tested functions,
   and offer to demonstrate it immediately afterwards.

A confident explanation of a tested analysis is far better than watching someone
fight a terminal.

---

## 2. What not to say

State these boundaries clearly and early; they make the rest of the presentation
credible.

| Do not say | Say instead |
| --- | --- |
| "This monitors air quality in real time." | "This analyses a fixed historical record from 2004–2005." |
| "The sensors drift." | "One sensor channel shows evidence consistent with a temporal change." |
| "The air got worse in 2005." | "Exceedances cluster in winter, and 2005 in this record is winter-heavy." |
| "The model predicts benzene." | "The model estimates benzene at the same hour, retrospectively." |
| "This proves cheap sensors work." | "At this site, most of the benzene signal is in the cheap sensors." |
| "Our R² is 0.86, so it works everywhere." | "R² 0.86 on a held-out future period at this one site." |
| "There is no drift in the other sensors." | "There is no clear drift signal for the other four." |

---

## 3. The 5-minute demo

Target: one clear story, four pages, no scrolling marathons.

### 0:00 — Set the frame (Overview)

**Say:**

> "EcoPulse analyses one urban air-quality monitoring station: 9,357 hourly
> readings from March 2004 to April 2005. It combines five certified reference
> analysers, five cheap metal-oxide sensors, and weather. The important constraint
> is that this is a historical record, not a live feed, and everything you see is
> recomputed when the page loads — there are no stored results."

**Point at** the period, the row count and the completeness figures on the page.

**Transition:** "First question: can we trust the record?"

### 0:45 — Data Quality

**Click** *Data Quality*.

**Say:**

> "Missing values here are written as minus 200, not blanks, so naively they would
> corrupt every average. There are 16,701 of them, about 13.7 % of all cells.
> Cleaning replaces them with true missing values and deletes no rows: 9,357 in,
> 9,357 out, with zero gaps and zero duplicate timestamps."

**Point at** the four missingness groups and the outage table.

> "The missingness is structured. Nine columns fail together in 16 outages totalling
> 366 hours — that's the device being off. NOx and NO2 fail together, CO fails
> alone, and NMHC is almost entirely absent. So missingness is not random, and
> every statistic describes the hours we actually observed."

### 1:45 — Pollution Explorer

**Click** *Pollution Explorer*.

**Say:**

> "Two patterns stand out. There is a strong diurnal cycle: NO2 peaks around seven
> in the evening at about 150 micrograms per cubic metre and bottoms out around four
> in the morning at about 59. And there is a seasonal cycle: winter averages about
> 140, summer about 91."

**Change** the pollutant to carbon monoxide to show the same evening peak.

> "It's not just NO2 — CO and benzene share that evening maximum."

### 2:45 — Anomaly Detection

**Click** *Anomaly Detection*. Leave the threshold at 200.

**Say:**

> "200 micrograms per cubic metre is the European hourly limit for NO2. At that
> threshold there are 388 exceedance hours in 137 separate episodes. The longest
> runs 18 hours, from 12 to 13 February 2005. The highest single reading is 339.7."

**Mention the trap:**

> "273 of those 388 hours fall in 2005 — but 2005 is only three and a half months of
> winter data. So that is a seasonal effect, not evidence that air quality worsened."

### 3:45 — Benzene Regression

**Click** *Benzene Regression*. This is the centrepiece; slow down.

**Say:**

> "Can we estimate benzene, which needs an expensive analyser, from signals a cheap
> station could collect? Two things make this easy to fake, and we avoid both.
>
> First, the target never enters the features, and the test suite proves that by
> scrambling the target and checking the feature matrix is unchanged.
>
> Second, this sensor — PT08.S2 — correlates at 0.982 with benzene. Including it
> would just measure benzene twice. We exclude it, and then we measure the cost of
> excluding it rather than assuming it."

**Point at** the comparison table and then the leakage panel.

> "The headline model is the sensor-only one: weather and the four cheap metal-oxide
> sensors, nothing else. The Random Forest reaches R² 0.840 on the held-out future
> period, with an RMSE of 2.60 micrograms. Adding the proxy back would push R² to
> 0.969 and cut the error by more than half — but that gain is measurement, not
> prediction."

**Point at** the negative baseline.

> "The mean baseline scores minus 0.18. That is not a bug: we split chronologically,
> training on earlier data and testing on later data, and the later period sits at a
> lower benzene level. So predicting the training average is worse than predicting
> nothing. That negative number is the evidence the split is honest."

**Point at** the predictor tiers.

> "Weather alone explains 0.28. Adding the four cheap sensors takes it to 0.84, and
> that is the headline — the low-cost sensor case. The last row adds the reference
> analysers and only reaches 0.86, and we label it as a reference-assisted
> comparison, because it describes a station that already owns those expensive
> instruments. Most of the signal is already in the inexpensive ones."

### 4:45 — Close (Methodology)

**Click** *Methodology & About*.

**Say:**

> "Every number you've seen is recomputed from the source files, and this page
> re-checks their SHA-256 digests live and shows they match. The same digests are
> asserted in the test suite, so the results cannot silently drift from the data.
> 281 tests cover loading, cleaning, analysis, drift, the model and every page."

**Final line:**

> "The project's contribution is as much methodological as statistical: controlling
> for season before claiming drift, excluding a feature that measures the answer,
> and reporting a negative baseline instead of choosing a split that hides it."

---

## 4. The 10-minute demo

Add these segments to the 5-minute track.

### Drift (insert after Data Quality, ~2:30)

**Click** *Pollutant Relationships*.

**Say:**

> "Each sensor is paired with the reference analyser it is meant to represent. CO
> and its sensor correlate at 0.88; the benzene sensor at 0.98. But look at the NO2
> sensor: 0.16. Whatever that channel is measuring, it is not tracking NO2 in this
> record. We report that rather than hiding it."

Then explain the drift test without needing a dedicated page (the verdicts are in
the report and the Relationships page shows the calibration panels):

> "Sensor drift is easy to claim and hard to prove, because the obvious test is
> confounded by season. We apply four checks at once: a per-month trend, a matched
> same-calendar-month comparison, a pairs-bootstrap confidence interval, and a
> sensitivity reading. Four of the five sensors have no established direction of
> change. One — the NOx sensor — shows a slope trend of minus 0.039 per month with a
> confidence interval that excludes zero, and both matched months agree. We report
> that as evidence consistent with a temporal change, not as proof of drift,
> because the data cannot tell us whether the cause is the sensor, the inlet, or the
> environment."

### The tier and importance story (extend Benzene Regression by 2 minutes)

**Say:**

> "The four cheap sensors account for about 71 % of the model's total feature
> importance. The single most important input is the ozone channel, then the CO
> sensor, then the NOx sensor. That is the operational finding: at this site, most
> of the benzene signal is available cheaply."

**Then add the caveat:**

> "Importance is not causation. It measures how much each variable reduces prediction
> error, not whether it causes benzene."

### Reproducibility (extend Methodology by 1 minute)

**Show** the terminal:

```powershell
python -m pytest -q
```

> "281 tests. They re-derive the headline figures, so if the analysis changes a
> claim, a test fails. The source files are hash-locked, and the dashboard re-checks
> those hashes at runtime."

### Limitations (close with 1 minute)

**Say:**

> "What this cannot do: it is one site over fourteen months, so it does not
> generalise. It is historical, so it is not live monitoring. The model is
> retrospective, so it is not a forecast. The drift test has limited power over
> fourteen monthly points, so 'no clear drift signal' is not the same as 'no drift'.
> And the missingness is structured, so every statistic describes observed hours."

---

## 5. Likely questions during the demo

| Question | Short answer |
| --- | --- |
| "Why not use a random train/test split?" | "Air quality is autocorrelated; a random split lets the model interpolate between neighbouring hours. We split chronologically so the score reflects a genuine future period." |
| "Why is the baseline negative?" | "It predicts the training mean, but R² is measured against the test mean, and the test period is lower. A negative baseline shows the split is honest." |
| "Why exclude the best predictor?" | "It correlates at 0.982 with the target; including it measures benzene twice. We report the 0.107 R² cost of excluding it." |
| "Is the model accurate enough to use?" | "For retrospective estimation at this site, yes: R² 0.86, RMSE 2.42 µg/m³. It is not validated elsewhere and is not a forecast." |
| "Do the sensors drift?" | "One channel shows evidence consistent with a temporal change; the other four have no established direction. We don't claim general drift." |
| "Why is NMHC excluded?" | "The brief excludes it, and it is only present for 9.8 % of hours." |
| "How do you know the data wasn't altered?" | "SHA-256 digests are asserted in the tests and re-checked live on the Methodology page." |
| "What happens if the data file is missing?" | "The app catches the load error and explains how to rebuild the CSV instead of crashing." |
| "How many tests?" | "281, including 18 that launch the real dashboard and visit every page." |
| "What would you do next?" | "Prospective validation on unseen data, lag-based forecasting, prediction intervals, and multi-site replication." |

For deeper answers, hand over to `docs/VIVA_PREPARATION.md`.

---

## 6. Post-demo checklist

- [ ] Stop the Streamlit process (`Ctrl+C` in its terminal).
- [ ] Confirm `Data/` still contains only the two original files.
- [ ] Confirm `git status` (if used) shows no unintended changes to the dataset.
- [ ] Note any question you could not answer, and add it to the viva guide.

---

## 7. Exact commands, for reference

```powershell
# From C:\Users\ASUS\Desktop\EcoPulse

# one-time setup
python -m pip install -e .

# rebuild the cleaned dataset (reads Data/, writes output/)
python -m ecopulse.cleaning

# run the full test suite
python -m pytest -q

# launch the dashboard (either of these)
.\run_app.bat
python -m streamlit run app/main.py
```

Open **<http://localhost:8501>**.

To stop the server, press `Ctrl+C` in the terminal running Streamlit.
