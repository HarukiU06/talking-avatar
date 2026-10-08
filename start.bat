@echo off
rem Talking Avatar: double-click to start the app.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Talking Avatar is not installed yet. Run setup.sh first: see README section 3.
    pause
    exit /b 1
)
rem UTF-8 console, so names and text in any language print correctly
chcp 65001 >nul
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
