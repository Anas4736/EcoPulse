#!/usr/bin/env bash
# ===========================================================================
# EcoPulse - Air Quality Monitoring and Data Analysis System
# Starts the Streamlit dashboard on macOS / Linux.
# ===========================================================================
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[EcoPulse] Creating virtual environment in .venv ..."
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "[EcoPulse] Starting dashboard ..."
echo "[EcoPulse] Open http://localhost:8501 in your browser."
python -m streamlit run app/main.py
