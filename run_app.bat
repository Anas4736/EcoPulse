@echo off
REM ===========================================================================
REM EcoPulse - Air Quality Monitoring and Data Analysis System
REM Starts the Streamlit dashboard on Windows.
REM
REM The first run creates a virtual environment (.venv) and installs the
REM packages listed in requirements.txt. Later runs reuse it.
REM ===========================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [EcoPulse] Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [EcoPulse] ERROR: could not create the virtual environment.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo [EcoPulse] Installing dependencies from requirements.txt ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [EcoPulse] ERROR: dependency installation failed.
        pause
        exit /b 1
    )
) else (
    call .venv\Scripts\activate.bat
)

echo [EcoPulse] Starting dashboard ...
echo [EcoPulse] Open http://localhost:8501 in your browser.
python -m streamlit run app\main.py

endlocal
