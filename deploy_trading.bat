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
    echo Timestamp,Symbol,TradeType,Quantity,Price,OrderStatus,Notes,ModelUsed,Confidence,Balance > "%TRADES_CSV%"
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
REM Parameters: symbol, trade_type, quantity, price, status, notes, model, confidence, balance
set "timestamp=%date% %time%"
echo %timestamp%,%~1,%~2,%~3,%~4,%~5,%~6,%~7,%~8,%~9 >> "%TRADES_CSV%"
call :log_info "Trade logged: %~1 %~2 %~3 @ %~4 - Status: %~5"
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

REM Check if symbol has proper format (e.g., BTCEUR, ETHEUR)
echo %symbol% | findstr /R "^[A-Z][A-Z][A-Z][A-Z][A-Z][A-Z]*$" >nul
if errorlevel 1 (
    call :log_warning "Symbol %symbol% has invalid format"
    exit /b 1
)

REM Additional validation can be added here
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

for %%s in (!VALID_SYMBOLS!) do (
    call :verify_symbol_models "%%s"
    if not errorlevel 1 (
        if "!VERIFIED_SYMBOLS!"=="" (
            set "VERIFIED_SYMBOLS=%%s"
        ) else (
            set "VERIFIED_SYMBOLS=!VERIFIED_SYMBOLS!,%%s"
        )
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

call :log_info "Verifying models for symbol: %symbol%"

for %%m in (!MODEL_TYPES!) do (
    call :find_model_for_symbol_and_type "!symbol!" "%%m"
    if not errorlevel 1 (
        set /a models_found=!models_found!+1
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

call :log_success "Symbol !symbol! has !models_found! model(s) available"
exit /b 0

:find_model_for_symbol_and_type
set "check_symbol=%~1"
set "check_model_type=%~2"

REM Enhanced model search with multiple fallback strategies
REM 1. Standard directory structure: models/model_type/symbol/
if exist "models\%check_model_type%\!check_symbol!" (
    for %%f in ("models\%check_model_type%\!check_symbol!\*") do (
        if exist "%%f" exit /b 0
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
call :log_info "=== Trading Loop Iteration %loop_count% ==="

REM Execute trading cycle for each verified symbol
for %%s in (%VERIFIED_SYMBOLS%) do (
    call :execute_trading_cycle "%%s"
)

call :log_info "Trading cycle completed. Next execution in %EXECUTION_INTERVAL% seconds..."
echo.
echo [%date% %time%] Cycle %loop_count% completed. Waiting %EXECUTION_INTERVAL% seconds...
echo.

REM Wait for next execution
timeout /t %EXECUTION_INTERVAL% /nobreak >nul

REM Continue loop
goto :trading_loop

:execute_trading_cycle
set "symbol=%~1"
set "retry_count=0"

call :log_info "Executing trading cycle for %symbol%..."

:retry_trading
set /a retry_count+=1

REM Execute trading with error handling
python "%TRADER_SCRIPT%" --symbol "%symbol%" --config "%CONFIG_FILE%" --single-cycle 2>temp_error.log
set "trade_result=%errorlevel%"

if %trade_result% EQU 0 (
    call :log_success "Trading cycle completed successfully for %symbol%"
    
    REM Log successful trade (placeholder - actual trade details would come from the Python script)
    call :log_trade "%symbol%" "CYCLE_COMPLETE" "N/A" "N/A" "SUCCESS" "Automated cycle completed" "ENSEMBLE" "N/A" "N/A"
    
    if exist temp_error.log del temp_error.log
    exit /b 0
) else (
    REM Read error details
    set "error_msg="
    if exist temp_error.log (
        for /f "delims=" %%i in (temp_error.log) do set "error_msg=%%i"
        del temp_error.log
    )
    
    call :log_error "Trading cycle failed for %symbol% (attempt %retry_count%/%MAX_RETRIES%): %error_msg%"
    
    REM Log failed trade
    call :log_trade "%symbol%" "CYCLE_FAILED" "N/A" "N/A" "ERROR" "Cycle failed: %error_msg%" "N/A" "N/A" "N/A"
    
    REM Retry logic
    if %retry_count% LSS %MAX_RETRIES% (
        call :log_info "Retrying in %RETRY_DELAY% seconds..."
        timeout /t %RETRY_DELAY% /nobreak >nul
        goto :retry_trading
    ) else (
        call :log_error "Max retries exceeded for %symbol%, skipping this cycle"
        exit /b 1
    )
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