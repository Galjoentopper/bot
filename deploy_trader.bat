@echo off
REM Simple Trading Bot Deployment Script
REM Reads config file, loads models for symbols, and starts trading

setlocal enabledelayedexpansion

echo.
echo ================================
echo   Trading Bot Deployment
echo ================================
echo.

REM Configuration
set "CONFIG_FILE=training_config.yaml"
set "TRADER_SCRIPT=scripts\enhanced_trader.py"

echo [INFO] Starting trading bot deployment...

REM Check if we're in the correct directory
if not exist "scripts" (
    echo [ERROR] Please run this script from the Bot root directory
    echo Current directory: %CD%
    echo Expected to find 'scripts' folder here
    pause
    exit /b 1
)

REM Check for required files
if not exist "%CONFIG_FILE%" (
    echo [ERROR] Configuration file not found: %CONFIG_FILE%
    echo Please ensure training_config.yaml exists in the root directory
    pause
    exit /b 1
)

if not exist "%TRADER_SCRIPT%" (
    echo [ERROR] Trading script not found: %TRADER_SCRIPT%
    if exist "scripts\trader.py" (
        set "TRADER_SCRIPT=scripts\trader.py"
        echo [WARNING] Falling back to standard trader script
    ) else (
        echo Please ensure the trading script exists
        pause
        exit /b 1
    )
)

REM Extract symbols from configuration file
echo [INFO] Extracting trading symbols from configuration...
(
echo import yaml
echo with open('%CONFIG_FILE%', 'r'^) as f:
echo     config = yaml.safe_load(f^)
echo symbols = config.get('data_acquisition', {}^).get('symbols', []^)
echo if not symbols:
echo     symbols = config.get('trading', {}^).get('symbols', []^)
echo if not symbols:
echo     symbols = config.get('symbols', []^)
echo if symbols:
echo     print('SYMBOLS:' + ','.join(symbols^)^)
echo else:
echo     print('NO_SYMBOLS_FOUND'^)
) > temp_extract_symbols.py

python temp_extract_symbols.py > temp_symbols.txt 2>nul
set "python_result=!errorlevel!"
del temp_extract_symbols.py 2>nul

if not !python_result! == 0 (
    echo [ERROR] Failed to extract symbols from configuration
    del temp_symbols.txt 2>nul
    pause
    exit /b 1
)

REM Read extracted symbols
set "TRADING_SYMBOLS="
for /f "tokens=2 delims=:" %%i in ('type temp_symbols.txt ^| findstr "SYMBOLS"') do (
    set "TRADING_SYMBOLS=%%i"
)
del temp_symbols.txt 2>nul

if "!TRADING_SYMBOLS!"=="" (
    echo [ERROR] No trading symbols found in configuration file
    echo Please ensure symbols are defined in the configuration file under:
    echo   data_acquisition.symbols or trading.symbols or symbols
    pause
    exit /b 1
)

echo [SUCCESS] Trading symbols found: !TRADING_SYMBOLS!

REM Convert comma-separated symbols to space-separated for processing
set "SYMBOLS_SPACED=!TRADING_SYMBOLS:,= !"

REM Check if models exist for the symbols
echo [INFO] Checking for trained models...
set "MISSING_MODELS=0"
set "MODEL_TYPES=gru lightgbm ppo"

for %%s in (!SYMBOLS_SPACED!) do (
    echo [INFO] Checking models for symbol: %%s
    set "symbol_has_models=0"
    
    for %%m in (!MODEL_TYPES!) do (
        REM Check multiple possible model locations
        if exist "models\%%m\%%s\" (
            for %%f in ("models\%%m\%%s\*") do (
                if exist "%%f" (
                    set "symbol_has_models=1"
                    echo [INFO]   Found %%m model for %%s
                    goto :next_model_%%s_%%m
                )
            )
        )
        
        REM Check flat structure
        for %%e in (pkl pt joblib zip) do (
            if exist "models\%%m_%%s.%%e" (
                set "symbol_has_models=1"
                echo [INFO]   Found %%m model for %%s (flat structure)
                goto :next_model_%%s_%%m
            )
            if exist "models\best_wf_%%m_%%s.%%e" (
                set "symbol_has_models=1"
                echo [INFO]   Found %%m model for %%s (best walkforward)
                goto :next_model_%%s_%%m
            )
        )
        
        :next_model_%%s_%%m
    )
    
    if !symbol_has_models! == 0 (
        echo [WARNING] No models found for symbol: %%s
        set /a MISSING_MODELS=!MISSING_MODELS!+1
    )
)

if !MISSING_MODELS! GTR 0 (
    echo [WARNING] Some symbols are missing models, but continuing...
    echo [INFO] You may want to train models for missing symbols first
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"

REM Start trading
echo.
echo ================================
echo   Starting Trading Bot
echo ================================
echo.
echo Configuration: %CONFIG_FILE%
echo Trading Symbols: !TRADING_SYMBOLS!
echo Model Types: %MODEL_TYPES%
echo.
echo The trading bot will now start with the configured symbols.
echo Check logs\trading.log for detailed activity.
echo Press Ctrl+C to stop the trading bot.
echo.

REM Start the trading script
python "%TRADER_SCRIPT%" --config "%CONFIG_FILE%"

if errorlevel 1 (
    echo.
    echo [ERROR] Trading bot encountered an error
    echo Check logs\trading.log for details
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Trading bot completed successfully
pause
exit /b 0