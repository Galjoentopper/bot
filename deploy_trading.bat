@echo off
REM Deployment script starting
setlocal enabledelayedexpansion

echo ========================================
echo  Automated Paper Trading Deployment
echo ========================================
echo.
echo This script will automatically set up and run the paper trading bot
echo with automatic model import and feature generation.
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

echo Python is installed.
echo.

REM Check if we're in the correct directory
if not exist "scripts" (
    echo ERROR: Please run this script from the Bot_kilo root directory
    echo Current directory: %CD%
    echo Expected to find 'scripts' folder here.
    exit /b 1
)

echo Directory check passed.
echo.

REM Check if models directory exists and has content (label-based, avoids complex parentheses)
if not exist models goto :NO_MODELS
set "MODELS_HAVE_CONTENT="
for /f "delims=" %%A in ('dir /b models 2^>nul') do set MODELS_HAVE_CONTENT=1
if not defined MODELS_HAVE_CONTENT goto :EMPTY_MODELS
goto :MODELS_OK

:NO_MODELS
echo Models directory not found. Checking for model packages to import...
echo.

REM Look for model packages in current directory
set "found_package=0"
for %%f in (*.zip) do (
    echo Found model package: %%f
    echo Attempting automatic import...
    call import_models.bat
    if not errorlevel 1 (
        set "found_package=1"
        echo Model import completed successfully.
        goto :MODELS_OK
    ) else (
        echo Warning: Failed to import %%f
    )
)

if !found_package! equ 0 (
    echo ERROR: No models found and no model packages available for import!
    echo Please place a model transfer package (*.zip) in this directory.
    exit /b 1
)

:EMPTY_MODELS
echo Models directory is empty. Checking for model packages to import...
echo.

REM Look for model packages in current directory
set "found_package=0"
for %%f in (*.zip) do (
    echo Found model package: %%f
    echo Attempting automatic import...
    call import_models.bat
    if not errorlevel 1 (
        set "found_package=1"
        echo Model import completed successfully.
        goto :MODELS_OK
    ) else (
        echo Warning: Failed to import %%f
    )
)

if !found_package! equ 0 (
    echo ERROR: Models directory is empty and no model packages available for import!
    echo Please place a model transfer package (*.zip) in this directory.
    exit /b 1
)

:MODELS_OK

echo Models directory found with content.
echo.

REM Check if requirements.txt exists and install from it
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo WARNING: Some dependencies from requirements.txt failed to install
        echo Continuing with manual dependency check...
    ) else (
        echo Dependencies installed from requirements.txt
        goto :skip_manual_deps
    )
)

REM Manual dependency check if requirements.txt failed or doesn't exist
echo Checking Python dependencies manually...

REM Essential packages for trading
set "IMPORT_NAMES=numpy pandas yaml sklearn requests telegram ccxt"
for %%p in (%IMPORT_NAMES%) do (
    if not "%%p"=="" (
        echo Checking %%p...
        python -c "import %%p" 2>nul
        if errorlevel 1 (
            echo Installing %%p...
            pip install %%p
            if errorlevel 1 (
                echo WARNING: Failed to install %%p
                echo You may need to install it manually: pip install %%p
            )
        )
    )
)

:skip_manual_deps
echo Dependencies check completed.
echo.

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "config" mkdir config

echo Created necessary directories.
echo.

REM Check for trading configuration
if not exist "src\config\config_trading.yaml" (
    if exist "config_trading.yaml" (
        echo Moving config_trading.yaml to src\config directory...
        move config_trading.yaml src\config\
    ) else (
        echo WARNING: config_trading.yaml not found!
        echo The bot will use default settings.
        echo You may want to create a configuration file for optimal performance.
    )
)

REM Check if enhanced trader script exists, fallback to regular trader
set "TRADER_SCRIPT="
if exist "scripts\enhanced_trader.py" (
    set "TRADER_SCRIPT=scripts\enhanced_trader.py"
    echo Using enhanced trader script.
) else (
    if exist "scripts\trader.py" (
        set "TRADER_SCRIPT=scripts\trader.py"
        echo Using standard trader script.
    ) else (
        echo ERROR: No trader script found!
        echo Expected: scripts\enhanced_trader.py or scripts\trader.py
        exit /b 1
    )
)

echo.

REM Run automated model validation
echo Running automated model validation...
if exist "validate_models.bat" (
    call validate_models.bat
    if errorlevel 1 (
        echo Model validation failed. Continuing with warnings...
        echo Check logs for validation details.
    ) else (
        echo Model validation completed successfully.
    )
) else (
    echo validate_models.bat not found, skipping validation...
)

REM Generate features based on model metadata
echo.
echo Analyzing model metadata for feature generation...
if exist "scripts\generate_features_from_metadata.py" (
    echo Running feature generation from model metadata...
    python "scripts\generate_features_from_metadata.py" --models-dir "models" --output-dir "." --verbose
    if errorlevel 1 (
        echo Warning: Feature generation from metadata failed. Using default features.
        echo The bot will attempt to use existing feature configurations.
    ) else (
        echo Feature generation completed successfully.
        echo Generated files: feature_config.json, feature_mapping.json, feature_config.yaml
        if exist "feature_config.json" (
            echo Feature configuration file created successfully.
        )
        if exist "feature_mapping.json" (
            echo Feature mapping file created successfully.
        )
    )
) else (
    echo ERROR: Feature generation script not found!
    echo Expected: scripts\generate_features_from_metadata.py
    echo This script is required for proper feature alignment with models.
    exit /b 1
)

echo.
echo ========================================
echo     Starting Paper Trading Bot
echo ========================================
echo.
echo Configuration:
echo - Trader Script: !TRADER_SCRIPT!
echo - Models Directory: models
echo - Logs Directory: logs
echo - Mode: Paper Trading
echo.
echo The bot will start in paper trading mode.
echo Press Ctrl+C to stop the bot.
echo.
echo Logs will be saved to the 'logs' directory.
echo Monitor the logs for trading activity and performance.
echo.
echo Starting trader automatically...
echo.

REM Start the trading bot with appropriate arguments
echo Starting trader...
if "!TRADER_SCRIPT!"=="scripts\enhanced_trader.py" (
    python !TRADER_SCRIPT!
) else (
    python !TRADER_SCRIPT!
)

if errorlevel 1 (
    echo.
    echo ERROR: Trading bot encountered an error!
    echo Check the logs directory for more information.
    echo.
    echo Common issues:
    echo - Missing or corrupted model files
    echo - Network connectivity problems
    echo - Configuration errors
    echo.
    echo Try running 'validate_models.bat' to check your models.
    exit /b 1
)

echo.
echo ========================================
echo Trading bot stopped successfully.
echo ========================================
echo.
echo Check the following for results:
echo - logs\ directory for trading logs
echo - data\ directory for market data
echo - Any generated reports or performance files
echo.