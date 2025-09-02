@echo off
REM Test script to run enhanced_trader.py for maximum 5 minutes then kill it
REM This addresses the guideline: "do run scripts/trader.py for max 5 minutes than kill it"

echo 🧪 Enhanced Trader Test Script (Windows Batch)
echo This will run enhanced_trader.py for a maximum of 5 minutes
echo ==========================================

set SCRIPT_DIR=%~dp0
set TRADER_PATH=%SCRIPT_DIR%enhanced_trader.py

if not exist "%TRADER_PATH%" (
    echo ❌ Error: enhanced_trader.py not found at %TRADER_PATH%
    exit /b 1
)

echo 🚀 Starting enhanced_trader.py with 5 minute timeout...
echo 📍 Command: python "%TRADER_PATH%"
echo ⏰ Will automatically kill after 300 seconds

REM Start the enhanced trader in background
start /B python "%TRADER_PATH%" > enhanced_trader_output.log 2>&1
set TRADER_PID=%ERRORLEVEL%

echo ✅ Enhanced trader process started
echo ⏳ Waiting for enhanced trader to complete or timeout...

REM Monitor for 5 minutes (300 seconds)
set SECONDS_ELAPSED=0
set TIMEOUT_SECONDS=300

:monitor_loop
if %SECONDS_ELAPSED% geq %TIMEOUT_SECONDS% goto timeout_reached

REM Check if python process is still running (simplified check)
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe" >NUL
if errorlevel 1 (
    echo ✅ Enhanced trader process completed on its own
    goto cleanup
)

REM Show progress every 30 seconds
set /a MODULO=%SECONDS_ELAPSED% %% 30
if %MODULO%==0 if %SECONDS_ELAPSED% gtr 0 (
    set /a REMAINING=%TIMEOUT_SECONDS% - %SECONDS_ELAPSED%
    echo ⏱️  Elapsed: %SECONDS_ELAPSED%s ^| Remaining: %REMAINING%s
)

timeout /t 1 /nobreak > NUL
set /a SECONDS_ELAPSED+=1
goto monitor_loop

:timeout_reached
echo ⏰ 5 minute timeout reached. Attempting to kill enhanced trader process...

REM Try to kill python processes gracefully
taskkill /IM python.exe /T /F >NUL 2>&1
if %ERRORLEVEL%==0 (
    echo 💀 Enhanced trader process force killed
) else (
    echo ⚠️  Could not find enhanced trader process to kill
)

:cleanup
echo 📊 Test completed
echo 🧹 Cleaning up...

REM Clean up log file if it exists
if exist enhanced_trader_output.log (
    echo 📄 Enhanced trader output saved to enhanced_trader_output.log
)

echo ✅ Cleanup completed
echo.
echo Test finished. Press any key to exit...
pause > NUL