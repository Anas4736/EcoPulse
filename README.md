# EcoPulse: Air Quality Data Analysis

EcoPulse is an interactive web application developed using Python and Streamlit to analyse historical air-quality data. The project focuses on data cleaning, exploratory data analysis, pollutant relationships, sensor drift, anomaly detection, and machine-learning-based benzene estimation.

**Live Demo:** https://ecopulse-3zmursrcnshm59k2f5chqn.streamlit.app/

**GitHub Repository:** https://github.com/Anas4736/EcoPulse

**Important:** This project analyses historical measurements recorded between March 2004 and April 2005. It is not a real-time air-quality monitoring system and does not provide live readings or future pollution forecasts.

## 1. Project Overview

Air-quality datasets can help identify pollution patterns, relationships between pollutants, and changes in sensor behaviour. However, missing values, measurement limitations, and temporal variations can affect the reliability of the analysis.

EcoPulse provides an interactive dashboard for examining these issues through data processing, statistical analysis, visualisation, data-quality assessment, and machine-learning experiments.

## 2. Features

- **Overview:** Summarises dataset statistics, measurement coverage, and key findings.
- **Pollution Explorer:** Explores pollutant concentrations and their variation over time.
- **Pollutant Relationships:** Examines correlations between pollutants and sensor responses.
- **Data Quality:** Analyses missing values, incomplete measurements, and data gaps.
- **Anomaly Detection:** Identifies periods that exceed user-selected pollutant thresholds.
- **Benzene Regression:** Compares regression models and displays evaluation metrics and diagnostic plots.
- **Methodology and About:** Documents the dataset, processing methods, modelling approach, and project limitations.

## 3. Dataset

EcoPulse uses the [UCI Air Quality dataset, Dataset ID 360](https://archive.ics.uci.edu/dataset/360/air+quality), available through the UCI Machine Learning Repository.

| Attribute | Description |
|---|---|
| Source | UCI Machine Learning Repository |
| Study period | March 2004 to April 2005 |
| Observations | 9,357 hourly records |
| Measured variables | 13 |
| Monitoring locations | One station in an Italian city |
| Missing-value marker | `-200` |
| Original dataset file | `Data/AirQualityUCI.xlsx` |

The cleaning pipeline treats `-200` as a missing value rather than a valid measurement.

The dataset does not contain PM2.5 or PM10 measurements. Consequently, EcoPulse does not calculate a standard composite Air Quality Index (AQI). Its charts and threshold counts refer to individual pollutants.

### Dataset Citation

De Vito, S. (2008). *Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F

Related publication: De Vito, S. et al. (2008). “On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario.” *Sensors and Actuators B: Chemical*, 129(2).

Please consult the dataset's current licence and usage terms before reusing it.

## 4. Machine Learning

The Benzene Regression module estimates the reference benzene concentration, represented by `C6H6(GT)`, using selected sensor measurements, weather variables, and time-related features.

The project compares:

- Mean baseline
- Linear Regression
- Random Forest Regression
- Sensor-only and reference-assisted feature sets

Model evaluation uses a chronological train/test split and time-series cross-validation.

The sensor response `PT08.S2(NMHC)` is excluded from the headline sensor-only model because it is strongly associated with the benzene reference measurement. Including it could make the model appear more accurate than would be justified for the intended low-cost sensor scenario. A separate diagnostic demonstrates the effect of including this proxy.

The reported results are retrospective estimates on the available dataset. They do not establish future forecasting ability or guarantee performance at other locations.

## 5. Technologies Used

- **Programming language:** Python
- **Web framework:** Streamlit
- **Data processing:** pandas, NumPy
- **Visualisation:** Plotly
- **Machine learning:** scikit-learn
- **Statistical analysis:** SciPy
- **Testing:** pytest
- **Version control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

The application entry point is `app/main.py`. The reusable analysis modules are located in `src/ecopulse/`.

## 6. Project Structure

```text
EcoPulse/
├── Data/
│   └── AirQualityUCI.xlsx
├── output/
│   └── airquality_clean.csv
├── src/
│   └── ecopulse/
│       ├── config.py
│       ├── loader.py
│       ├── cleaning.py
│       ├── quality.py
│       ├── features.py
│       ├── analysis.py
│       ├── drift.py
│       └── ml.py
├── app/
│   ├── main.py
│   ├── data.py
│   ├── theme.py
│   └── views/
├── tests/
├── docs/
│   ├── VIVA_PREPARATION.md
│   └── DEMO_SCRIPT.md
├── requirements.txt
├── pyproject.toml
├── PROJECT_REPORT.md
└── README.md
```

## 7. Installation and Local Setup

### Prerequisites

- Python 3.13, recommended to match the locally verified environment
- Git
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/Anas4736/EcoPulse.git
cd EcoPulse
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Step 5: Run the Application

```bash
python -m streamlit run app/main.py
```

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

## 8. Running Tests

Run the test suite with:

```powershell
python -m pytest -q
```

On Windows, if temporary-directory cleanup causes issues, use:

```powershell
python -m pytest -q --basetemp "$HOME\pytest-ecopulse-temp"
```

The project previously passed 281 tests in the development environment. Tests should be rerun after changes to the source code or dependencies.

## 9. Rebuilding the Cleaned Dataset

After installing the project package, run:

```powershell
python -m ecopulse.cleaning
```

The cleaning pipeline reads the original workbook and generates the cleaned CSV under `output/`. The original dataset should remain unchanged.

## 10. Deployment

EcoPulse is deployed using Streamlit Community Cloud.

| Setting | Value |
|---|---|
| Hosting platform | Streamlit Community Cloud |
| GitHub repository | https://github.com/Anas4736/EcoPulse |
| Branch | `main` |
| Application entry point | `app/main.py` |
| Live application | https://ecopulse-3zmursrcnshm59k2f5chqn.streamlit.app/ |

Deployment dependencies are listed in `requirements.txt`, while `pyproject.toml` defines the Python package configuration.

## 11. Limitations

- The dataset represents one monitoring station over approximately fourteen months in 2004–2005.
- The application does not receive live sensor readings.
- Missing values and sensor limitations may affect the analysis.
- Sensor responses are not equivalent to certified reference measurements.
- Machine-learning performance on this dataset may not generalise to other locations or time periods.
- Benzene regression is a retrospective estimation experiment, not a future pollution forecast.
- A standard composite AQI is not calculated because PM2.5 and PM10 measurements are absent.
- The project is intended for academic analysis and should not be used for health, emergency, or regulatory decisions.

## 12. Project Documentation

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md): Detailed project report.
- [`docs/VIVA_PREPARATION.md`](docs/VIVA_PREPARATION.md): Viva preparation notes.
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md): Project demonstration guide.

## 13. Team Members

| Name | Roll Number |
|---|---|
| Mohd Anas | 2026010138 |
| Sumayah Lateef | 2026010125 |

The team should document each member's actual contributions before final academic submission.

## 14. Acknowledgements

We acknowledge the UCI Machine Learning Repository and the original dataset contributors for making the Air Quality dataset available for research and educational use.

---

EcoPulse is an academic data science project demonstrating historical air-quality analysis, data-quality assessment, visualisation, and machine-learning evaluation.
