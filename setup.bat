@echo off
REM Setup script for Windows

echo ==========================================
echo Intelli-ODM Environment Setup
echo Platform: Windows
echo ==========================================

REM Check Python version
echo.
echo [1/5] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/downloads/
    exit /b 1
)
python --version

REM Create virtual environment
echo.
echo [2/5] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    exit /b 1
)
echo Virtual environment created at .venv\

REM Activate virtual environment
echo.
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/5] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo.
echo [5/5] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate
echo.
echo To deactivate, run:
echo   deactivate
echo.
echo Next steps:
echo   1. Install Ollama: https://ollama.com/download/windows
echo   2. Pull Llama3 model: ollama pull llama3:8b
echo   3. Run the system: python orchestrator.py
echo.
pause

