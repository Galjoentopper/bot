@echo off
REM Enhanced Trading System Deployment Script for Windows
REM =====================================================
REM This script initializes and starts the live trading system with comprehensive
REM error handling, progress indicators, and automatic dependency management

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   Enhanced Trading System Deployment
echo   Live Trading Initialization
echo ========================================
echo.

REM Configuration
set "LOG_FILE=logs\deployment.log"
set "CONFIG_FILE=config_trading.yaml"
set "TRADER_SCRIPT=scripts\enhanced_trader.py"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"

REM Initialize log file
echo [%date% %time%] Starting trading system deployment > "%LOG_FILE%"

call :log_info "Enhanced trading system deployment started"

REM Execute deployment pipeline
call :check_system_requirements
if errorlevel 1 exit /b 1

call :validate_environment
if errorlevel 1 exit /b 1

call :setup_dependencies
if errorlevel 1 exit /b 1

call :import_models_if_needed
if errorlevel 1 exit /b 1

call :validate_models
if errorlevel 1 exit /b 1

call :configure_trading_system
if errorlevel 1 exit /b 1

call :start_trading_system
if errorlevel 1 exit /b 1

call :log_success "Trading system deployment completed successfully!"
echo.
echo ========================================
echo   🚀 TRADING SYSTEM DEPLOYED! 🚀
echo   Monitor logs\trading.log for activity
echo ========================================
echo.
pause
exit /b 0

REM ==========================================
REM FUNCTION DEFINITIONS
REM ==========================================

:log_info
echo [INFO] %~1
echo [%date% %time%] [INFO] %~1 >> "%LOG_FILE%"
exit /b 0

:log_success
echo [SUCCESS] %~1
echo [%date% %time%] [SUCCESS] %~1 >> "%LOG_FILE%"
exit /b 0

:log_warning
echo [WARNING] %~1
echo [%date% %time%] [WARNING] %~1 >> "%LOG_FILE%"
exit /b 0

:log_error
echo [ERROR] %~1
echo [%date% %time%] [ERROR] %~1 >> "%LOG_FILE%"
exit /b 0

:check_system_requirements
call :log_info "Checking system requirements..."

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed or not in PATH"
    echo   Please install Python 3.8+ from https://python.org
    echo   Make sure to add Python to PATH during installation
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :log_info "Python version: %PYTHON_VERSION%"

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    call :log_error "pip is not available"
    echo   pip should be included with Python 3.8+
    exit /b 1
)

REM Check memory availability (basic check)
for /f "tokens=2 delims=:" %%i in ('systeminfo ^| findstr "Total Physical Memory"') do (
    call :log_info "System memory: %%i"
)

call :log_success "System requirements validated"
exit /b 0

:validate_environment
call :log_info "Validating environment structure..."

REM Check if we're in the correct directory
if not exist "scripts" (
    call :log_error "Please run this script from the Bot_kilo root directory"
    echo   Current directory: %CD%
    echo   Expected to find 'scripts' folder here
    exit /b 1
)

REM Check for required scripts
if not exist "%TRADER_SCRIPT%" (
    call :log_error "Trading script not found: %TRADER_SCRIPT%"
    echo   Please ensure all required files are present
    exit /b 1
)

REM Create necessary directories
call :log_info "Creating necessary directories..."
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"
if not exist "data" mkdir "data"
if not exist "backups" mkdir "backups"

call :log_success "Environment structure validated"
exit /b 0

:setup_dependencies
call :log_info "Setting up Python dependencies..."

REM Upgrade pip first
pip install --quiet --upgrade pip

REM Install core requirements
if exist "requirements.txt" (
    call :log_info "Installing requirements from requirements.txt..."
    pip install --quiet -r requirements.txt
    if errorlevel 1 (
        call :log_warning "Some packages failed to install, continuing..."
    )
) else (
    call :log_info "Installing essential packages manually..."
    pip install --quiet pandas numpy pyyaml python-binance ccxt python-telegram-bot mlflow
)

call :log_success "Dependencies setup completed"
exit /b 0

:import_models_if_needed
call :log_info "Checking for trained models..."

REM Check if models directory exists and has content
if not exist "models" goto :no_models_found
set "models_exist=0"
for /f "delims=" %%A in ('dir /b models 2^>nul') do set models_exist=1
if !models_exist! == 1 goto :models_found

:no_models_found
call :log_warning "No models found, checking for import packages..."

REM Look for model packages to import
set "found_packages=0"
for %%f in (*.zip) do (
    set "found_packages=1"
    call :log_info "Found model package: %%f"
)

if !found_packages! == 1 (
    call :log_info "Attempting automatic model import..."
    call import_models.bat
    if errorlevel 1 (
        call :log_error "Automatic model import failed"
        echo   Please manually run import_models.bat first
        exit /b 1
    )
    call :log_success "Models imported successfully"
) else (
    call :log_error "No trained models or import packages found"
    echo   Please either:
    echo     1. Copy a model transfer package (*.zip) to this directory
    echo     2. Run import_models.bat manually
    echo     3. Train models using train_models_linux.sh on a Linux machine
    exit /b 1
)
goto :import_complete

:models_found
call :log_success "Trained models found in models directory"

:import_complete
exit /b 0

:validate_models
call :log_info "Validating imported models..."

set "model_count=0"
set "gru_models=0"
set "lightgbm_models=0"
set "ppo_models=0"

REM Count model files
for /r "models" %%f in (*.pkl *.pt *.joblib *.zip) do (
    set /a model_count+=1
)

REM Check for specific model types
if exist "models\gru" set "gru_models=1"
if exist "models\lightgbm" set "lightgbm_models=1"
if exist "models\ppo" set "ppo_models=1"

if !model_count! == 0 (
    call :log_error "No model files found after import validation"
    exit /b 1
)

call :log_info "Model validation results:"
call :log_info "  Total model files: %model_count%"
if !gru_models! == 1 (
    call :log_info "  ✓ GRU models available"
) else (
    call :log_warning "  ✗ GRU models not found"
)
if !lightgbm_models! == 1 (
    call :log_info "  ✓ LightGBM models available"
) else (
    call :log_warning "  ✗ LightGBM models not found"
)
if !ppo_models! == 1 (
    call :log_info "  ✓ PPO models available"
) else (
    call :log_warning "  ✗ PPO models not found"
)

call :log_success "Model validation completed"
exit /b 0

:configure_trading_system
call :log_info "Configuring trading system..."

REM Check for trading configuration
if not exist "%CONFIG_FILE%" (
    if exist "training_config.yaml" (
        call :log_info "Creating trading configuration from training config..."
        REM Create a simplified trading config
        (
            echo # Trading Configuration - Auto-generated
            echo trading:
            echo   initial_balance: 10000
            echo   max_position_size: 0.1
            echo   transaction_fee: 0.001
            echo   slippage: 0.0005
            echo   model_weights:
            echo     gru: 0.45
            echo     lightgbm: 0.45
            echo     ppo: 0.1
            echo models:
            echo   lightgbm:
            echo     enabled: true
            echo   gru:
            echo     enabled: true
            echo   ppo:
            echo     enabled: true
            echo notifications:
            echo   telegram:
            echo     enabled: true
            echo     bot_token: '7733436451:AAH6Sls8uL4fEgd6Ty7VEKSBIMauhaVkN4c'
            echo     chat_id: '7988790407'
        ) > "%CONFIG_FILE%"
        call :log_info "Trading configuration created"
    ) else (
        call :log_warning "No trading configuration found, using defaults"
    )
)

REM Validate Python environment for trading
call :log_info "Validating Python trading environment..."
python -c "
try:
    import pandas, numpy, yaml
    import ccxt, binance
    print('Core trading dependencies verified')
except ImportError as e:
    print(f'Missing dependency: {e}')
    exit(1)
" 2>nul
if errorlevel 1 (
    call :log_warning "Some trading dependencies missing, attempting to install..."
    pip install --quiet python-binance ccxt pandas numpy pyyaml
)

call :log_success "Trading system configured"
exit /b 0

:start_trading_system
call :log_info "Starting live trading system..."

REM Final pre-flight checks
if not exist "%TRADER_SCRIPT%" (
    call :log_error "Trader script not found: %TRADER_SCRIPT%"
    exit /b 1
)

REM Create startup command
set "startup_cmd=python %TRADER_SCRIPT%"
if exist "%CONFIG_FILE%" (
    set "startup_cmd=!startup_cmd! --config %CONFIG_FILE%"
)

call :log_info "Executing trading system startup..."
call :log_info "Command: %startup_cmd%"

echo.
echo ========================================
echo   🚀 LAUNCHING TRADING SYSTEM 🚀
echo ========================================
echo.
echo The trading system is now starting...
echo Monitor the console for real-time updates
echo Log files are available in the logs/ directory
echo.
echo Press Ctrl+C to stop the trading system
echo.

REM Execute the trading system
%startup_cmd%
set "trader_exit_code=!errorlevel!"

if !trader_exit_code! == 0 (
    call :log_success "Trading system exited normally"
) else (
    call :log_warning "Trading system exited with code: !trader_exit_code!"
)

exit /b !trader_exit_code!
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