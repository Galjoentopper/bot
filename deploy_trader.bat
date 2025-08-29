@echo off
REM Deploy Trader Script - Flexible symbol and model deployment
REM ============================================================
REM This script reads the configuration file, extracts symbols,
REM verifies corresponding models exist, and starts the trader
REM with the appropriate symbols and models.

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   Enhanced Trading Bot Deployment
echo   Flexible Symbol and Model Setup
echo ========================================
echo.

REM Configuration
set "CONFIG_FILE=training_config.yaml"
set "TRADER_SCRIPT=scripts\enhanced_trader.py"
set "LOG_FILE=logs\deployment.log"
set "ERROR_LOG=logs\error.log"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"

REM Initialize logging
echo [%date% %time%] Deploy trader started > "%LOG_FILE%"

call :log_info "Starting flexible trader deployment..."

REM Check if configuration file exists
if not exist "%CONFIG_FILE%" (
    call :log_error "Configuration file not found: %CONFIG_FILE%"
    echo   Please ensure training_config.yaml exists in the root directory
    pause
    exit /b 1
)

REM Check if trader script exists
if not exist "%TRADER_SCRIPT%" (
    call :log_error "Trader script not found: %TRADER_SCRIPT%"
    echo   Please ensure enhanced_trader.py exists in the scripts directory
    pause
    exit /b 1
)

call :log_info "Extracting symbols from configuration..."

REM Extract symbols from configuration file
python -c "
import yaml
try:
    with open('%CONFIG_FILE%', 'r') as f:
        config = yaml.safe_load(f)
    symbols = config.get('data_acquisition', {}).get('symbols', [])
    if not symbols:
        symbols = config.get('data', {}).get('symbols', [])
    if not symbols:
        symbols = config.get('symbols', [])
    if symbols:
        print('SYMBOLS_FOUND:' + ','.join(symbols))
    else:
        print('NO_SYMBOLS_FOUND')
except Exception as e:
    print(f'ERROR:{e}')
" > temp_symbols.txt 2>&1

if errorlevel 1 (
    call :log_error "Failed to extract symbols from configuration"
    type temp_symbols.txt
    del temp_symbols.txt 2>nul
    pause
    exit /b 1
)

REM Read extracted symbols
set "TRADING_SYMBOLS="
for /f "tokens=2 delims=:" %%i in ('type temp_symbols.txt ^| findstr "SYMBOLS_FOUND"') do (
    set "TRADING_SYMBOLS=%%i"
)
del temp_symbols.txt 2>nul

if "%TRADING_SYMBOLS%"=="" (
    call :log_error "No trading symbols found in configuration"
    echo   Please ensure symbols are defined in the configuration file
    echo   Example configuration:
    echo   data_acquisition:
    echo     symbols: ['BTCEUR', 'ETHEUR', 'ADAEUR']
    pause
    exit /b 1
)

call :log_info "Trading symbols found: %TRADING_SYMBOLS%"

REM Extract model types from configuration
call :log_info "Extracting model types from configuration..."

python -c "
import yaml
try:
    with open('%CONFIG_FILE%', 'r') as f:
        config = yaml.safe_load(f)
    models = config.get('training', {}).get('models', [])
    if not models:
        models = ['gru', 'lightgbm', 'ppo']  # Default models
    if models:
        print('MODELS_FOUND:' + ','.join(models))
    else:
        print('NO_MODELS_FOUND')
except Exception as e:
    print(f'ERROR:{e}')
" > temp_models.txt 2>&1

set "TRADING_MODELS="
for /f "tokens=2 delims=:" %%i in ('type temp_models.txt ^| findstr "MODELS_FOUND"') do (
    set "TRADING_MODELS=%%i"
)
del temp_models.txt 2>nul

if "%TRADING_MODELS%"=="" (
    set "TRADING_MODELS=gru,lightgbm,ppo"
    call :log_warning "No models specified in config, using defaults: %TRADING_MODELS%"
) else (
    call :log_info "Model types found: %TRADING_MODELS%"
)

REM Verify models directory exists
if not exist "models" (
    call :log_error "Models directory not found"
    echo   Please ensure models have been trained and are available in the models directory
    echo   Run training first using: python scripts\enhanced_trainer.py
    pause
    exit /b 1
)

REM Verify models exist for symbols
call :log_info "Verifying models exist for symbols..."
set "VERIFIED_SYMBOLS="
set "VERIFIED_MODELS="

for %%s in (%TRADING_SYMBOLS%) do (
    set "symbol_models_found=0"
    set "available_models="
    
    for %%m in (%TRADING_MODELS%) do (
        if exist "models\%%m\%%s" (
            set /a symbol_models_found+=1
            if "!available_models!"=="" (
                set "available_models=%%m"
            ) else (
                set "available_models=!available_models!,%%m"
            )
            call :log_info "  Found %%m model for %%s"
        ) else (
            call :log_warning "  Missing %%m model for %%s"
        )
    )
    
    if !symbol_models_found! GEQ 1 (
        call :log_info "Symbol %%s has !symbol_models_found! model(s): !available_models!"
        if "%VERIFIED_SYMBOLS%"=="" (
            set "VERIFIED_SYMBOLS=%%s"
        ) else (
            set "VERIFIED_SYMBOLS=!VERIFIED_SYMBOLS!,%%s"
        )
        
        REM Build verified models list
        for %%vm in (!available_models!) do (
            echo !VERIFIED_MODELS! | findstr "%%vm" >nul
            if errorlevel 1 (
                if "%VERIFIED_MODELS%"=="" (
                    set "VERIFIED_MODELS=%%vm"
                ) else (
                    set "VERIFIED_MODELS=!VERIFIED_MODELS!,%%vm"
                )
            )
        )
    ) else (
        call :log_warning "Symbol %%s has no available models"
    )
)

if "%VERIFIED_SYMBOLS%"=="" (
    call :log_error "No symbols have any available models"
    echo   Please train models for your symbols first
    echo   Run: python scripts\enhanced_trainer.py --symbols %TRADING_SYMBOLS%
    pause
    exit /b 1
)

if "%VERIFIED_MODELS%"=="" (
    set "VERIFIED_MODELS=%TRADING_MODELS%"
)

call :log_success "Model verification completed!"
call :log_info "Verified symbols: %VERIFIED_SYMBOLS%"
call :log_info "Available models: %VERIFIED_MODELS%"

REM Test the trader configuration
call :log_info "Testing trader configuration..."
python "%TRADER_SCRIPT%" --config "%CONFIG_FILE%" --symbols %VERIFIED_SYMBOLS% --models %VERIFIED_MODELS% --test-mode

if errorlevel 1 (
    call :log_error "Trader configuration test failed"
    echo   Please check the error messages above
    pause
    exit /b 1
)

call :log_success "Trader configuration test passed!"

REM Ask user for deployment mode
echo.
echo Select deployment mode:
echo 1. Single cycle (run once for each symbol)
echo 2. Continuous trading (run indefinitely)
echo 3. Test mode only (validate and exit)
echo.
set /p "MODE=Enter your choice (1-3): "

if "%MODE%"=="1" (
    call :log_info "Starting single cycle mode..."
    echo.
    echo Starting single trading cycle for symbols: %VERIFIED_SYMBOLS%
    echo Using models: %VERIFIED_MODELS%
    echo.
    python "%TRADER_SCRIPT%" --config "%CONFIG_FILE%" --symbols %VERIFIED_SYMBOLS% --models %VERIFIED_MODELS% --single-cycle
    call :log_info "Single cycle completed"
) else if "%MODE%"=="2" (
    call :log_info "Starting continuous trading mode..."
    echo.
    echo ========================================
    echo   CONTINUOUS TRADING MODE ACTIVE
    echo ========================================
    echo.
    echo Trading symbols: %VERIFIED_SYMBOLS%
    echo Model types: %VERIFIED_MODELS%
    echo Configuration: %CONFIG_FILE%
    echo.
    echo Press Ctrl+C to stop trading
    echo.
    python "%TRADER_SCRIPT%" --config "%CONFIG_FILE%" --symbols %VERIFIED_SYMBOLS% --models %VERIFIED_MODELS%
    call :log_info "Continuous trading stopped"
) else if "%MODE%"=="3" (
    call :log_info "Test mode completed - configuration is valid"
    echo.
    echo Configuration validation completed successfully!
    echo - Symbols: %VERIFIED_SYMBOLS%
    echo - Models: %VERIFIED_MODELS%
    echo - All models are properly accessible
) else (
    call :log_warning "Invalid selection, defaulting to test mode"
    echo Configuration validation completed successfully!
)

echo.
echo ========================================
echo   Deployment completed
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
exit /b 0

:log_success
echo [SUCCESS] %~1
echo [%date% %time%] [SUCCESS] %~1 >> "%LOG_FILE%"
exit /b 0

:log_warning
echo [WARNING] %~1
echo [%date% %time%] [WARNING] %~1 >> "%LOG_FILE%"
echo [%date% %time%] [WARNING] %~1 >> "%ERROR_LOG%"
exit /b 0

:log_error
echo [ERROR] %~1
echo [%date% %time%] [ERROR] %~1 >> "%LOG_FILE%"
echo [%date% %time%] [ERROR] %~1 >> "%ERROR_LOG%"
exit /b 0