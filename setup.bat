@echo off
REM Healthcare Prediction System - Quick Setup Script for Windows
REM Run this file to set up and start the project

echo ========================================
echo Healthcare Prediction System Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/4] Setup complete!
echo.
echo ========================================
echo Ready to run!
echo ========================================
echo.
echo Choose an option:
echo   1. Full setup (Generate data + Train models + Start web app)
echo   2. Quick start (Start web app only - requires trained models)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Running full setup...
    python run_project.py
) else if "%choice%"=="2" (
    echo.
    echo Starting web application...
    cd web
    python app.py
) else (
    echo Invalid choice. Please run the script again.
)

pause
