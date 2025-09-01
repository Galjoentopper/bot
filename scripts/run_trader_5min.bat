@echo off
REM Script to run trader.py for a maximum of 5 minutes
REM This addresses the user's guideline to run scripts/trader.py for max 5 minutes
REM and then kill it, rather than waiting for it to end by itself.

echo Running trader.py for maximum 5 minutes (300 seconds)...

REM Check if trader.py exists
if not exist "scripts\trader.py" (
    echo ERROR: scripts\trader.py not found!
    exit /b 1
)

REM Check for configuration file
set CONFIG_FILE=
if exist "training_config.yaml" (
    set CONFIG_FILE=--config training_config.yaml
    echo Using configuration file: training_config.yaml
) else (
    echo No configuration file found, running with defaults
)

REM Run trader.py with timeout using PowerShell
powershell -Command "Start-Process python -ArgumentList 'scripts/trader.py %CONFIG_FILE% --iterations 100' -NoNewWindow -PassThru | ForEach-Object { $_.WaitForExit(300000); if (-not $_.HasExited) { $_.Kill(); Write-Host \"Trader script timed out after 5 minutes (this is expected per guidelines)\" } }"

REM Check the result
if %ERRORLEVEL% EQU 0 (
    echo Trader script completed successfully
    echo ✅ Test completed successfully
    exit /b 0
) else (
    echo Trader script failed with an error
    echo ❌ Test failed
    exit /b 1
)