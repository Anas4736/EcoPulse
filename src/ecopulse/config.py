"""Central configuration for the EcoPulse project.















Everything that other modules need to agree on lives here:







  * file locations (project folders, source file, generated artifacts)







  * the sentinel value used by the UCI dataset for missing measurements







  * the canonical column registry (name, human label, unit, role, description)















Keeping this in one place means the data loader, the cleaning step, the charts







and the ML models all refer to the *same* column names, so a rename only ever







has to happen once.















Source of truth for the dataset description:







    Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository.







    https://doi.org/10.24432/C59K5F







"""















from pathlib import Path















# --------------------------------------------------------------------------







# Project paths







# --------------------------------------------------------------------------







# src/ecopulse/config.py  ->  src/ecopulse  ->  src  ->  EcoPulse (project root)







PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]















# NOTE ON "Data" vs "output":







# Windows filesystems are case-insensitive, so a folder called "data" would be







# the *same* folder as the original "Data" source folder. To keep the supplied







# source directory untouched, all generated artifacts go to "output" instead.







DATA_DIR: Path = PROJECT_ROOT / "Data"







RAW_DIR: Path = DATA_DIR







PROCESSED_DIR: Path = PROJECT_ROOT / "output"







APP_DIR: Path = PROJECT_ROOT / "app"







TESTS_DIR: Path = PROJECT_ROOT / "tests"















#: The original UCI workbook. This file is treated as READ-ONLY and is never







#: modified by this project. The cleaned CSV is written to PROCESSED_DIR.







SOURCE_XLSX: Path = RAW_DIR / "AirQualityUCI.xlsx"















#: Cleaned dataset produced by the cleaning pipeline (a deliverable).







CLEANED_CSV: Path = PROCESSED_DIR / "airquality_clean.csv"















#: Optional SQLite store. Only created if it is actually enabled (see db.py).







SQLITE_DB: Path = PROCESSED_DIR / "ecopulse.db"















#: Where fitted scikit-learn models are cached between Streamlit sessions.







MODELS_DIR: Path = PROCESSED_DIR / "models"















# --------------------------------------------------------------------------







# Dataset constants







# --------------------------------------------------------------------------







#: The UCI Air Quality dataset tags every missing measurement with -200.







#: This value is NOT a physical reading and must never enter a chart or model.







SENTINEL_VALUE: float = -200















DATE_COLUMN = "Date"







TIME_COLUMN = "Time"







#: Name of the merged timestamp column created during cleaning.







DATETIME_COLUMN = "datetime"















#: The 13 measured variables (everything except Date and Time).







MEASURED_COLUMNS: list[str] = [







    "CO(GT)",







    "PT08.S1(CO)",







    "NMHC(GT)",







    "C6H6(GT)",







    "PT08.S2(NMHC)",







    "NOx(GT)",







    "PT08.S3(NOx)",







    "NO2(GT)",







    "PT08.S4(NO2)",







    "PT08.S5(O3)",







    "T",







    "RH",







    "AH",







]















#: The 5 metal-oxide sensor responses (raw device units, not concentrations).







SENSOR_COLUMNS: list[str] = [







    "PT08.S1(CO)",







    "PT08.S2(NMHC)",







    "PT08.S3(NOx)",







    "PT08.S4(NO2)",







    "PT08.S5(O3)",







]















#: The 5 co-located certified reference analyzer concentrations.







REFERENCE_COLUMNS: list[str] = [







    "CO(GT)",







    "NMHC(GT)",







    "C6H6(GT)",







    "NOx(GT)",







    "NO2(GT)",







]















#: Meteorological measurements recorded alongside the sensors.







WEATHER_COLUMNS: list[str] = ["T", "RH", "AH"]















#: Reference columns that are useful as *pollution* measures. NMHC(GT) is kept







#: out of this group because ~90% of its values are missing.







POLLUTANT_REFERENCE_COLUMNS: list[str] = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]















# --------------------------------------------------------------------------







# Column registry







# --------------------------------------------------------------------------







# For every measured column: a display label, the unit, the role of the







# variable and a short plain-English description.







#







# role:







#   "reference" -> concentration measured by the certified reference analyzer







#   "sensor"    -> raw response of one metal-oxide sensor (device units)







#   "weather"   -> meteorological measurement







COLUMN_METADATA: dict[str, dict[str, str]] = {
    "CO(GT)": {
        "label": "Carbon monoxide (CO)",
        "unit": "mg/m³",
        "role": "reference",
        "description": "True hourly averaged CO concentration from the reference analyzer.",
    },
    "PT08.S1(CO)": {
        "label": "CO sensor response",
        "unit": "device units",
        "role": "sensor",
        "description": "Tin oxide sensor (S1) nominally targeted at CO. Raw response, not a concentration.",
    },
    "NMHC(GT)": {
        "label": "Non-methane hydrocarbons (NMHC)",
        "unit": "µg/m³",
        "role": "reference",
        "description": "Overall non-methane hydrocarbon concentration from the reference analyzer. About 90% of values are missing.",
    },
    "C6H6(GT)": {
        "label": "Benzene (C6H6)",
        "unit": "µg/m³",
        "role": "reference",
        "description": "Benzene concentration from the reference analyzer. Primary modelling target.",
    },
    "PT08.S2(NMHC)": {
        "label": "NMHC sensor response",
        "unit": "device units",
        "role": "sensor",
        "description": "Titania sensor (S2) nominally targeted at NMHC. Raw response.",
    },
    "NOx(GT)": {
        "label": "Nitrogen oxides (NOx)",
        "unit": "ppb",
        "role": "reference",
        "description": "True hourly averaged NOx concentration from the reference analyzer.",
    },
    "PT08.S3(NOx)": {
        "label": "NOx sensor response",
        "unit": "device units",
        "role": "sensor",
        "description": "Tungsten oxide sensor (S3) nominally targeted at NOx. Raw response.",
    },
    "NO2(GT)": {
        "label": "Nitrogen dioxide (NO2)",
        "unit": "µg/m³",
        "role": "reference",
        "description": "True hourly averaged NO2 concentration from the reference analyzer.",
    },
    "PT08.S4(NO2)": {
        "label": "NO2 sensor response",
        "unit": "device units",
        "role": "sensor",
        "description": "Tungsten oxide sensor (S4) nominally targeted at NO2. Raw response.",
    },
    "PT08.S5(O3)": {
        "label": "O3 sensor response",
        "unit": "device units",
        "role": "sensor",
        "description": "Indium oxide sensor (S5) nominally targeted at O3. Raw response.",
    },
    "T": {
        "label": "Temperature",
        "unit": "°C",
        "role": "weather",
        "description": "Ambient temperature.",
    },
    "RH": {
        "label": "Relative humidity",
        "unit": "%",
        "role": "weather",
        "description": "Relative humidity.",
    },
    "AH": {
        "label": "Absolute humidity",
        "unit": "scaled",
        "role": "weather",
        "description": "Absolute humidity. Found to be a scaled function of T and RH, so it adds little independent information.",
    },
}
















#: Reference column that the benzene regression page models.







PRIMARY_TARGET = "C6H6(GT)"















#: Each metal-oxide sensor together with the reference analyser it is







#: compared against. The sensor-drift investigation and the dashboard's







#: sensor panels both work from this list, so the pairing is defined once.







SENSOR_REFERENCE_PAIRS: list[tuple[str, str]] = [







    ("PT08.S1(CO)", "CO(GT)"),







    ("PT08.S2(NMHC)", "C6H6(GT)"),







    ("PT08.S3(NOx)", "NOx(GT)"),







    ("PT08.S4(NO2)", "NO2(GT)"),







    ("PT08.S5(O3)", "NO2(GT)"),







]















#: Reference threshold used for the pollution episode analysis, in the unit of







#: the column being examined. Defaults to the EU hourly limit value for NO2







#: (Directive 2008/50/EC). The UI lets the user change it.







DEFAULT_NO2_THRESHOLD_UG_M3: float = 200.0















#: Expected SHA-256 digests of the two original files in ``Data/``. The







#: dashboard recomputes these at runtime so a visitor can see that the figures







#: on screen come from the unmodified source files.







SOURCE_HASHES: dict[str, str] = {







    "AirQualityUCI.xlsx": "C92D102BB35CD4C0B42C0B6A1336C4065CCDCA7069F80A6890EE1B7E3124BD5D",







    "Mohd_Anas_2026010138_Project_Synopsis.pdf": "7E2A7F4D7C6A13EB9C75DF258A5246107F83B3CF14BE696E8E9C6245D8382A57",







}















__all__ = [







    "PROJECT_ROOT",







    "DATA_DIR",







    "RAW_DIR",







    "PROCESSED_DIR",







    "APP_DIR",







    "TESTS_DIR",







    "SOURCE_XLSX",







    "CLEANED_CSV",







    "SQLITE_DB",







    "MODELS_DIR",







    "SENTINEL_VALUE",







    "DATE_COLUMN",







    "TIME_COLUMN",







    "DATETIME_COLUMN",







    "MEASURED_COLUMNS",







    "SENSOR_COLUMNS",







    "REFERENCE_COLUMNS",







    "WEATHER_COLUMNS",







    "POLLUTANT_REFERENCE_COLUMNS",







    "COLUMN_METADATA",







    "PRIMARY_TARGET",







    "SENSOR_REFERENCE_PAIRS",







    "SOURCE_HASHES",







    "DEFAULT_NO2_THRESHOLD_UG_M3",







]







