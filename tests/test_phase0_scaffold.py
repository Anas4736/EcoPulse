"""Phase 0 tests: project scaffolding, configuration and source-data integrity.

These tests guard the two things that matter most at the start of the project:

1. the original dataset file is present and **unchanged** (SHA-256 check), and
2. the configuration module describes the dataset correctly, so every later
   module can rely on it.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from ecopulse import config


# SHA-256 of the supplied source files, recorded at project start.
# If these change, the original data has been modified and the project must
# stop and investigate before continuing.
EXPECTED_SOURCE_HASHES = {
    "AirQualityUCI.xlsx": "C92D102BB35CD4C0B42C0B6A1336C4065CCDCA7069F80A6890EE1B7E3124BD5D",
    "Mohd_Anas_2026010138_Project_Synopsis.pdf": (
        "7E2A7F4D7C6A13EB9C75DF258A5246107F83B3CF14BE696E8E9C6245D8382A57"
    ),
}


def _sha256(path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class TestSourceDataIntegrity:
    """The original files must never be modified by this project."""

    def test_source_workbook_exists(self, source_path):
        assert source_path.exists(), f"Missing source dataset: {source_path}"

    def test_source_workbook_hash_unchanged(self, source_path):
        actual = _sha256(source_path)
        expected = EXPECTED_SOURCE_HASHES["AirQualityUCI.xlsx"]
        assert actual == expected, (
            "The original AirQualityUCI.xlsx has been modified. "
            "EcoPulse must treat this file as read-only."
        )

    def test_synopsis_pdf_hash_unchanged(self):
        pdf = config.RAW_DIR / "Mohd_Anas_2026010138_Project_Synopsis.pdf"
        assert pdf.exists()
        assert _sha256(pdf) == EXPECTED_SOURCE_HASHES[
            "Mohd_Anas_2026010138_Project_Synopsis.pdf"
        ]

    def test_source_workbook_is_readable(self, source_path):
        frame = pd.read_excel(source_path)
        assert not frame.empty
        assert list(frame.columns)[:2] == ["Date", "Time"]


class TestConfiguration:
    """The configuration must describe the real dataset."""

    def test_paths_point_inside_project_root(self):
        assert config.SOURCE_XLSX.is_relative_to(config.PROJECT_ROOT)
        assert config.PROCESSED_DIR.is_relative_to(config.PROJECT_ROOT)
        assert config.APP_DIR.is_relative_to(config.PROJECT_ROOT)

    def test_measured_columns_unique_and_complete(self):
        assert len(config.MEASURED_COLUMNS) == 13
        assert len(set(config.MEASURED_COLUMNS)) == 13

    def test_column_groups_partition_measured_columns(self):
        groups = (
            config.SENSOR_COLUMNS
            + config.REFERENCE_COLUMNS
            + config.WEATHER_COLUMNS
        )
        assert sorted(groups) == sorted(config.MEASURED_COLUMNS)

    def test_every_measured_column_has_metadata(self):
        for column in config.MEASURED_COLUMNS:
            assert column in config.COLUMN_METADATA, f"No metadata for {column}"
            meta = config.COLUMN_METADATA[column]
            assert meta["label"] and meta["unit"] and meta["role"]
            assert meta["role"] in {"reference", "sensor", "weather"}

    def test_primary_target_is_a_reference_column(self):
        assert config.PRIMARY_TARGET in config.REFERENCE_COLUMNS

    def test_nmhc_flagged_as_descriptive_only(self):
        """NMHC(GT) stays in the dataset but is not a pollutant reference column."""
        assert "NMHC(GT)" in config.MEASURED_COLUMNS
        assert "NMHC(GT)" not in config.POLLUTANT_REFERENCE_COLUMNS


class TestWorkspaceLayout:
    """The folder layout required by the project plan must exist."""

    @pytest.mark.parametrize(
        "folder",
        ["Data", "src", "app", "tests", "output"],
    )
    def test_required_folders_exist(self, folder):
        assert (config.PROJECT_ROOT / folder).is_dir()

    def test_data_folder_contains_only_the_original_files(self):
        """The supplied Data folder must not gain generated sub-folders."""
        names = {p.name for p in config.RAW_DIR.iterdir()}
        assert names == {
            "AirQualityUCI.xlsx",
            "Mohd_Anas_2026010138_Project_Synopsis.pdf",
        }, f"Unexpected contents in the source Data folder: {names}"

    def test_requirements_file_exists(self):
        assert (config.PROJECT_ROOT / "requirements.txt").is_file()

    def test_readme_exists(self):
        assert (config.PROJECT_ROOT / "README.md").is_file()
