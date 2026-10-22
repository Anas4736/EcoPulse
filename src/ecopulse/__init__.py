"""EcoPulse - Air Quality Monitoring and Data Analysis System.

A Streamlit dashboard for exploring and modelling the UCI Air Quality dataset
(UCI Machine Learning Repository, Dataset ID 360).

The package is deliberately split into small, single-purpose modules so each
step of the pipeline can be read (and tested) on its own:

    config.py     shared constants and the column registry
    loader.py     reads the original workbook into a tidy DataFrame
    cleaning.py   converts the -200 missing markers, validates the timeline
    features.py   derives time, lag and rolling features
    analysis.py   exploratory statistics computed on demand
    ml.py         scikit-learn pipelines and evaluation

Team: Mohd Anas (2026010138) and Sumayah Lateef (2026010125).
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
