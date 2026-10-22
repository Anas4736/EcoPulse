"""Pytest configuration for the EcoPulse project.

The analysis code lives in ``src/ecopulse`` (a "src layout"). Adding ``src`` to
``sys.path`` here means the tests can ``import ecopulse`` without having to
install the package first, which keeps the setup simple for anyone cloning the
repository.
"""

import sys
from pathlib import Path

# Repository root = the folder that contains this conftest.py
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
