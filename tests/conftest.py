"""Shared pytest fixtures for the EcoPulse test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecopulse.config import SOURCE_XLSX


@pytest.fixture(scope="session")
def source_path() -> Path:
    """Absolute path to the original, read-only UCI workbook."""
    return SOURCE_XLSX


@pytest.fixture(scope="session")
def raw_frame():
    """The workbook read exactly as pandas returns it, with no cleaning.

    This is the "before" state used by the cleaning tests to prove that
    ``-200`` markers are converted correctly and that no rows are lost.
    """
    import pandas as pd

    return pd.read_excel(SOURCE_XLSX)


@pytest.fixture(scope="session")
def loaded_frame():
    """The workbook loaded through the EcoPulse loader (sentinels intact)."""
    from ecopulse.loader import load_raw

    return load_raw()


@pytest.fixture(scope="session")
def cleaned_result():
    """The fully cleaned dataset plus its data-quality report.

    Session-scoped because reading and cleaning the workbook takes a couple of
    seconds and every test needs the same result.
    """
    from ecopulse.cleaning import load_clean_dataset

    return load_clean_dataset()
