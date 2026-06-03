@echo off
title SilentNote - Starting...
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo  Starting SilentNote...
echo.
python main.py
pause
