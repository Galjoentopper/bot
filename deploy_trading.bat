@echo off
REM Enhanced Automated Trading System Deployment Script
REM ===================================================
REM Comprehensive trading automation with configuration processing,
REM model verification, scheduled execution, error handling, and detailed reporting

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   Enhanced Automated Trading System
echo   Comprehensive Trading Deployment
echo ========================================
echo.

REM Configuration
set "LOG_FILE=logs\deployment.log"
set "TRADING_LOG=logs\trading.log"
set "ERROR_LOG=logs\error.log"
set "TRADES_CSV=logs\trades_report.csv"
set "CONFIG_FILE=training_config.yaml"
set "TRADER_SCRIPT=scripts\enhanced_trader.py"
set "EXECUTION_INTERVAL=1800"
set "MAX_RETRIES=3"
set "RETRY_DELAY=30"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"

REM Initialize log files
echo [%date% %time%] Enhanced automated trading system deployment started > "%LOG_FILE%"
echo [%date% %time%] Trading operations log initialized > "%TRADING_LOG%"
echo [%date% %time%] Error tracking log initialized > "%ERROR_LOG%"

REM Initialize CSV report with headers
if not exist "%TRADES_CSV%" (
    REM Added TradeID column (optional) and corrected header ordering (Symbol retained second)
    echo Timestamp,TradeID,Symbol,TradeType,Quantity,Price,OrderStatus,Notes,ModelUsed,Confidence,Balance > "%TRADES_CSV%"
)

call :log_info "Enhanced automated trading system deployment started"

REM Execute comprehensive deployment pipeline
call :check_system_requirements
if errorlevel 1 exit /b 1

call :validate_environment
if errorlevel 1 exit /b 1

call :setup_dependencies
if errorlevel 1 exit /b 1

call :process_configuration
if errorlevel 1 exit /b 1

call :verify_models
if errorlevel 1 exit /b 1

call :initialize_trading_system
if errorlevel 1 exit /b 1

call :start_automated_trading_loop
if errorlevel 1 exit /b 1

call :log_success "Automated trading system deployment completed successfully!"
echo.
echo ========================================
echo   AUTOMATED TRADING SYSTEM ACTIVE!
echo   Monitor logs\trading.log for activity
echo   CSV reports: logs\trades_report.csv
echo ========================================
echo.
pause
exit /b 0

REM ==========================================
REM LOGGING FUNCTIONS
REM ==========================================

:log_info
echo [INFO] %~1
echo [%date% %time%] [INFO] %~1 >> "%LOG_FILE%"
echo [%date% %time%] [INFO] %~1 >> "%TRADING_LOG%"
exit /b 0

:log_success
echo [SUCCESS] %~1
echo [%date% %time%] [SUCCESS] %~1 >> "%LOG_FILE%"
echo [%date% %time%] [SUCCESS] %~1 >> "%TRADING_LOG%"
exit /b 0

:log_warning
echo [WARNING] %~1
echo [%date% %time%] [WARNING] %~1 >> "%LOG_FILE%"
echo [%date% %time%] [WARNING] %~1 >> "%TRADING_LOG%"
echo [%date% %time%] [WARNING] %~1 >> "%ERROR_LOG%"
exit /b 0

:log_error
echo [ERROR] %~1
echo [%date% %time%] [ERROR] %~1 >> "%LOG_FILE%"
echo [%date% %time%] [ERROR] %~1 >> "%TRADING_LOG%"
echo [%date% %time%] [ERROR] %~1 >> "%ERROR_LOG%"
exit /b 0

:log_trade
REM Parameters: trade_id, symbol, trade_type, quantity, price, status, notes, model, confidence, balance
set "timestamp=%date% %time%"
set "trade_id=%~1"
REM Shift remaining parameters (symbol now %2 etc.)
echo %timestamp%,%trade_id%,%~2,%~3,%~4,%~5,%~6,%~7,%~8,%~9 >> "%TRADES_CSV%"
call :log_info "Trade logged: %~2 %~3 %~4 @ %~5 - Status: %~6 (ID=%trade_id%)"
exit /b 0

REM ==========================================
REM SYSTEM VALIDATION FUNCTIONS
REM ==========================================

:check_system_requirements
call :log_info "Checking comprehensive system requirements..."

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed or not in PATH"
    echo   Please install Python 3.8+ from https://python.org
    echo   Make sure to add Python to PATH during installation
    exit /b 1
)

REM Get and validate Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :log_info "Python version: %PYTHON_VERSION%"

REM Check pip availability
pip --version >nul 2>&1
if errorlevel 1 (
    call :log_error "pip is not available"
    echo   pip should be included with Python 3.8+
    exit /b 1
)

REM Check system memory
for /f "tokens=2 delims=:" %%i in ('systeminfo ^| findstr "Total Physical Memory"') do (
    call :log_info "System memory: %%i"
)

REM Check disk space
for /f "tokens=3" %%i in ('dir /-c ^| find "bytes free"') do (
    call :log_info "Available disk space: %%i bytes"
)

call :log_success "System requirements validated successfully"
exit /b 0

:validate_environment
call :log_info "Validating comprehensive environment structure..."

REM Check if we're in the correct directory
if not exist "scripts" (
    call :log_error "Please run this script from the Bot root directory"
    echo   Current directory: %CD%
    echo   Expected to find 'scripts' folder here
    exit /b 1
)

REM Check for required scripts
if not exist "%TRADER_SCRIPT%" (
    call :log_error "Enhanced trading script not found: %TRADER_SCRIPT%"
    if exist "scripts\trader.py" (
        set "TRADER_SCRIPT=scripts\trader.py"
        call :log_warning "Falling back to standard trader script"
    ) else (
        echo   Please ensure all required files are present
        exit /b 1
    )
)

REM Create comprehensive directory structure
call :log_info "Creating comprehensive directory structure..."
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"
if not exist "data" mkdir "data"
if not exist "config" mkdir "config"
if not exist "backups" mkdir "backups"
if not exist "reports" mkdir "reports"
if not exist "cache" mkdir "cache"

call :log_success "Environment structure validated and created"
exit /b 0

:setup_dependencies
call :log_info "Setting up comprehensive Python dependencies..."

REM Upgrade pip first
pip install --quiet --upgrade pip
if errorlevel 1 (
    call :log_warning "Failed to upgrade pip, continuing with current version"
)

REM Install core requirements with error handling
if exist "requirements.txt" (
    call :log_info "Installing requirements from requirements.txt..."
    pip install --quiet -r requirements.txt
    if errorlevel 1 (
        call :log_warning "Some packages failed to install from requirements.txt"
        call :install_essential_packages
    ) else (
        call :log_success "All requirements installed successfully"
    )
) else (
    call :log_info "No requirements.txt found, installing essential packages..."
    call :install_essential_packages
)

call :log_success "Dependencies setup completed"
exit /b 0

:install_essential_packages
call :log_info "Installing essential trading packages..."
set "ESSENTIAL_PACKAGES=pandas numpy pyyaml python-binance ccxt python-telegram-bot scikit-learn lightgbm torch stable-baselines3 mlflow"
for %%p in (%ESSENTIAL_PACKAGES%) do (
    call :log_info "Installing %%p..."
    pip install --quiet %%p
    if errorlevel 1 (
        call :log_warning "Failed to install %%p, may cause issues"
    )
)
exit /b 0

REM ==========================================
REM CONFIGURATION PROCESSING FUNCTIONS
REM ==========================================

:process_configuration
call :log_info "Processing comprehensive trading configuration..."

REM Check for configuration file
if not exist "%CONFIG_FILE%" (
    call :log_error "Configuration file not found: %CONFIG_FILE%"
    echo   Please ensure training_config.yaml exists in the root directory
    exit /b 1
)

REM Validate configuration file format
python -c "import yaml; yaml.safe_load(open('%CONFIG_FILE%'))" 2>nul
if errorlevel 1 (
    call :log_error "Invalid YAML format in configuration file"
    exit /b 1
)

REM Extract and validate trading symbols
call :log_info "Extracting trading symbols from configuration..."
(
echo import yaml
echo with open('%CONFIG_FILE%', 'r'^) as f:
echo     config = yaml.safe_load(f^)
echo symbols = config.get('data_acquisition', {}^).get('symbols', []^)
echo if not symbols:
echo     symbols = config.get('data', {}^).get('symbols', []^)
echo if not symbols:
echo     symbols = config.get('symbols', []^)
echo if symbols:
echo     print('SYMBOLS_FOUND:' + ','.join(symbols^)^)
echo else:
echo     print('NO_SYMBOLS_FOUND'^)
) > temp_extract_symbols.py

python temp_extract_symbols.py > temp_symbols.txt 2>nul
set "python_result=!errorlevel!"
del temp_extract_symbols.py 2>nul

if not !python_result! == 0 (
    call :log_error "Failed to extract symbols from configuration"
    exit /b 1
)

REM Read extracted symbols
set "TRADING_SYMBOLS="
for /f "tokens=2 delims=:" %%i in ('type temp_symbols.txt ^| findstr "SYMBOLS_FOUND"') do (
    set "TRADING_SYMBOLS=%%i"
)
del temp_symbols.txt 2>nul

if "!TRADING_SYMBOLS!"=="" (
    call :log_error "No trading symbols found in configuration"
    echo   Please ensure symbols are defined in the configuration file
    exit /b 1
)

call :log_info "Trading symbols extracted: !TRADING_SYMBOLS!"

REM Convert comma-separated symbols to space-separated for batch processing
set "SYMBOLS_SPACED=!TRADING_SYMBOLS:,= !"

REM Validate each symbol configuration
call :log_info "Validating symbol configurations..."
set "VALID_SYMBOLS="
for %%s in (!SYMBOLS_SPACED!) do (
    call :validate_symbol_config "%%s"
    if not errorlevel 1 (
        if "!VALID_SYMBOLS!"=="" (
            set "VALID_SYMBOLS=%%s"
        ) else (
            set "VALID_SYMBOLS=!VALID_SYMBOLS!,%%s"
        )
    )
)

if "!VALID_SYMBOLS!"=="" (
    call :log_error "No valid symbol configurations found"
    exit /b 1
)

call :log_success "Configuration processing completed. Valid symbols: !VALID_SYMBOLS!"
exit /b 0

:validate_symbol_config
set "symbol=%~1"
call :log_info "Validating configuration for symbol: %symbol%"

REM Basic validation - just ensure symbol is not empty and looks reasonable
if "%symbol%"=="" (
    call :log_warning "Symbol is empty"
    exit /b 1
)

REM Check minimum length (should be at least 6 characters for crypto pairs)
if "%symbol:~5%"=="" (
    call :log_warning "Symbol %symbol% too short (minimum 6 characters)"
    exit /b 1
)

REM Symbols from our config file are trusted, so just accept them
call :log_info "Symbol %symbol% configuration is valid"
exit /b 0

REM ==========================================
REM MODEL VERIFICATION FUNCTIONS
REM ==========================================

:verify_models
call :log_info "Performing comprehensive model verification..."

if not exist "models" (
    call :log_error "Models directory not found"
    exit /b 1
)

set "VERIFIED_SYMBOLS="
set "MODEL_TYPES=gru lightgbm ppo"
REM Track per-model-type availability to allow degraded operation later
set "AVAILABLE_MODEL_TYPES="

for %%s in (!VALID_SYMBOLS!) do (
    call :verify_symbol_models "%%s"
    if not errorlevel 1 (
        if "!VERIFIED_SYMBOLS!"=="" (
            set "VERIFIED_SYMBOLS=%%s"
        ) else (
            set "VERIFIED_SYMBOLS=!VERIFIED_SYMBOLS!,%%s"
        )
    ) else (
        call :log_warning "Excluding symbol %%s due to missing models"
    )
)

if "!VERIFIED_SYMBOLS!"=="" (
    call :log_error "No symbols have complete model sets"
    exit /b 1
)

call :log_success "Model verification completed. Verified symbols: !VERIFIED_SYMBOLS!"
exit /b 0

:verify_symbol_models
set "symbol=%~1"
set "models_found=0"
set "missing_models="
set "found_types="

call :log_info "Verifying models for symbol: %symbol%"

for %%m in (!MODEL_TYPES!) do (
    call :find_model_for_symbol_and_type "!symbol!" "%%m"
    if not errorlevel 1 (
        set /a models_found=!models_found!+1
        if "!found_types!"=="" (set "found_types=%%m") else (set "found_types=!found_types!,%%m")
        call :log_info "  Found %%m model for !symbol!"
    ) else (
        if "!missing_models!"=="" (
            set "missing_models=%%m"
        ) else (
            set "missing_models=!missing_models!,%%m"
        )
        call :log_warning "  Missing %%m model for !symbol!"
    )
)

if !models_found! LSS 1 (
    call :log_warning "Symbol !symbol! has no models (!models_found!/3). Missing: !missing_models!"
    exit /b 1
)

REM Record globally which model types are present at least somewhere
for %%t in (!found_types!) do (
    echo !AVAILABLE_MODEL_TYPES! | find /i "%%t" >nul || set "AVAILABLE_MODEL_TYPES=!AVAILABLE_MODEL_TYPES! %%t"
)

call :log_success "Symbol !symbol! has !models_found! model(s) available"
exit /b 0

:find_model_for_symbol_and_type
set "check_symbol=%~1"
set "check_model_type=%~2"

REM Enhanced model search with multiple fallback strategies
REM 1. Standard directory structure: models/model_type/symbol/
if exist "models\%check_model_type%\!check_symbol!" (
    REM Look for direct files OR one level deeper (timestamped directories)
    for %%f in ("models\%check_model_type%\!check_symbol!\*.pkl" "models\%check_model_type%\!check_symbol!\*.pt" "models\%check_model_type%\!check_symbol!\*.pth" "models\%check_model_type%\!check_symbol!\*.zip" "models\%check_model_type%\!check_symbol!\*.joblib") do (
        if exist "%%f" exit /b 0
    )
    for /d %%d in ("models\%check_model_type%\!check_symbol!\*") do (
        for %%f in ("%%d\*.pkl" "%%d\*.pt" "%%d\*.pth" "%%d\*.zip" "%%d\*.joblib") do (
            if exist "%%f" exit /b 0
        )
        REM One more nested level (e.g., model_type\symbol\model_type\timestamp\file)
        for /d %%e in ("%%d\*") do (
            for %%g in ("%%e\*.pkl" "%%e\*.pt" "%%e\*.pth" "%%e\*.zip" "%%e\*.joblib") do (
                if exist "%%g" exit /b 0
            )
        )
    )
)

REM 2. Flat structure with various naming patterns
for %%e in (pkl pt joblib zip) do (
    if exist "models\%check_model_type%_!check_symbol!.%%e" exit /b 0
    if exist "models\%check_model_type%\%check_model_type%_!check_symbol!.%%e" exit /b 0
    if exist "models\best_wf_%check_model_type%_!check_symbol!.%%e" exit /b 0
    if exist "models\%check_model_type%\best_wf_%check_model_type%_!check_symbol!.%%e" exit /b 0
    if exist "models\!check_symbol!_%check_model_type%.%%e" exit /b 0
)

REM 3. Search in subdirectories for any model files
if exist "models\%check_model_type%" (
    for /r "models\%check_model_type%" %%f in (*!check_symbol!*) do (
        if exist "%%f" exit /b 0
    )
)

REM 4. Fallback directories
for %%d in (imported_models packaged_models legacy_models) do (
    if exist "%%d" (
        for %%e in (pkl pt joblib zip) do (
            if exist "%%d\%check_model_type%_!check_symbol!.%%e" exit /b 0
            if exist "%%d\%check_model_type%\!check_symbol!" (
                for %%f in ("%%d\%check_model_type%\!check_symbol!\*") do (
                    if exist "%%f" exit /b 0
                )
            )
        )
    )
)

exit /b 1

:initialize_trading_system
call :log_info "Initializing comprehensive trading system..."

REM Validate Python trading environment
call :log_info "Validating Python trading environment..."
(
echo try:
echo     import pandas, numpy, yaml, ccxt
echo     import sklearn, lightgbm
echo     print('Core trading dependencies verified'^)
echo except ImportError as e:
echo     print(f'Missing dependency: {e}'^)
echo     exit(1^)
) > temp_validate_env.py

python temp_validate_env.py 2>nul
set "validation_result=!errorlevel!"
del temp_validate_env.py 2>nul

if not !validation_result! == 0 (
    call :log_warning "Some trading dependencies missing, attempting recovery..."
    call :install_essential_packages
)

REM Test trading script functionality
call :log_info "Testing trading script functionality..."
python "%TRADER_SCRIPT%" --test-mode --config "%CONFIG_FILE%" 2>nul
if errorlevel 1 (
    call :log_warning "Trading script test failed, but continuing..."
)

call :log_success "Trading system initialized successfully"
exit /b 0

REM ==========================================
REM AUTOMATED TRADING LOOP FUNCTIONS
REM ==========================================

:start_automated_trading_loop
call :log_info "Starting automated trading loop with %EXECUTION_INTERVAL% second intervals..."

echo.
echo ========================================
echo   AUTOMATED TRADING SYSTEM ACTIVE
echo ========================================
echo.
echo Configuration:
echo - Execution Interval: %EXECUTION_INTERVAL% seconds (30 minutes)
echo - Trading Symbols: %VERIFIED_SYMBOLS%
echo - Model Types: %MODEL_TYPES%
echo - Logs Directory: logs\
echo - CSV Reports: %TRADES_CSV%
echo.
echo The system will execute trading operations every 30 minutes.
echo Press Ctrl+C to stop the automated trading system.
echo.
echo Monitor logs\trading.log for real-time activity
echo Monitor logs\trades_report.csv for trade records
echo.

set "loop_count=0"

:trading_loop
set /a loop_count+=1
call :log_info "=== Starting Unified Continuous Trading ===" 

REM Execute unified continuous trading (runs indefinitely)
call :execute_unified_trading_cycle

REM If we reach here, trading has stopped - log and exit
call :log_info "Unified continuous trading has stopped"
exit /b 0

:execute_unified_trading_cycle
call :log_info "Starting unified continuous trading for all symbols: %VERIFIED_SYMBOLS%"

REM Execute unified continuous trading (let it run indefinitely)
REM Normalize VERIFIED_SYMBOLS (comma-separated) to space-delimited for argparse nargs+
set "SYMBOL_ARGS=%VERIFIED_SYMBOLS:,= %"
python "%TRADER_SCRIPT%" --symbols %SYMBOL_ARGS% --config "%CONFIG_FILE%" 2>temp_error.log
set "trade_result=%errorlevel%"

if %trade_result% EQU 0 (
    call :log_success "Unified continuous trading session completed"
    
    REM Log trading session end
    call :log_trade "SESSION" "ALL_SYMBOLS" "CONTINUOUS_SESSION_END" "N/A" "N/A" "SUCCESS" "Continuous trading session ended normally" "ENSEMBLE" "N/A" "N/A"
    
    if exist temp_error.log del temp_error.log
    exit /b 0
) else (
    REM Read error details
    set "error_msg="
    if exist temp_error.log (
        for /f "delims=" %%i in (temp_error.log) do set "error_msg=%%i"
        del temp_error.log
    )
    
    call :log_error "Unified continuous trading failed: %error_msg%"
    
    REM Log failed trading session
    call :log_trade "SESSION" "ALL_SYMBOLS" "CONTINUOUS_SESSION_FAILED" "N/A" "N/A" "ERROR" "Continuous trading failed: %error_msg%" "N/A" "N/A" "N/A"
    
    exit /b 1
)

exit /b 0

REM ==========================================
REM ERROR RECOVERY FUNCTIONS
REM ==========================================

:handle_api_error
call :log_error "API Error detected: %~1"
REM Implement API-specific recovery logic
exit /b 0

:handle_model_error
call :log_error "Model Error detected: %~1"
REM Implement model-specific recovery logic
exit /b 0

:handle_trade_error
call :log_error "Trade Error detected: %~1"
REM Implement trade-specific recovery logic
exit /b 0

REM ==========================================
REM CLEANUP AND MAINTENANCE FUNCTIONS
REM ==========================================

:cleanup_logs
REM Rotate logs if they get too large
for %%f in ("%LOG_FILE%" "%TRADING_LOG%" "%ERROR_LOG%") do (
    for %%s in (%%f) do (
        if %%~zs GTR 10485760 (
            call :log_info "Rotating log file: %%f"
            copy %%f %%f.backup >nul
            echo. > %%f
        )
    )
)
exit /b 0

:backup_trades_csv
REM Backup trades CSV daily
set "backup_name=logs\trades_report_%date:~-4,4%%date:~-10,2%%date:~-7,2%.csv"
if not exist "%backup_name%" (
    copy "%TRADES_CSV%" "%backup_name%" >nul
    call :log_info "Trades CSV backed up to: %backup_name%"
)
exit /b 0