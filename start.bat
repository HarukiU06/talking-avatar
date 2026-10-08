@echo off
rem Talking Avatar: double-click to install (first run only) and open the app.
rem It runs start.sh with Git Bash, which comes with Git for Windows.
cd /d "%~dp0"
rem UTF-8 console, so names and text in any language print correctly
chcp 65001 >nul

set "GITBASH="
for %%B in ("%ProgramFiles%\Git\bin\bash.exe" "%ProgramFiles(x86)%\Git\bin\bash.exe" "%LocalAppData%\Programs\Git\bin\bash.exe") do (
    if not defined GITBASH if exist "%%~B" set "GITBASH=%%~B"
)

if defined GITBASH (
    "%GITBASH%" start.sh %*
    if errorlevel 1 pause
    exit /b
)

rem No Git Bash: the app can still open if it was installed some other way.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py %*
    if errorlevel 1 pause
    exit /b
)

echo Talking Avatar installs itself with Git Bash, which comes with Git for Windows:
echo   https://git-scm.com/download/win
echo Install it, then double-click start.bat again.
pause
exit /b 1
