@echo off
title SilentNote - Offline Minutes Generator

:: Change to the directory containing this batch file
cd /d "%~dp0"

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found. Using system Python 3.10.
    echo Run SETUP.md instructions first for a clean environment.
    echo.
)

:: Launch the application
echo Starting SilentNote...
py -3.10 main.py

:: If py -3.10 fails, try python
if errorlevel 1 (
    python main.py
)

pause
