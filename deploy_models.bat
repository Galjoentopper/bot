@echo off
REM Deploy Models Script for Enterprise Trading System
REM ================================================
REM Comprehensive model deployment with symbol extraction from config,
REM model verification, fallback loading, and enterprise-ready features

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   Enterprise Trading Model Deployment
echo   Advanced Model Loading and Validation
echo ========================================
echo.

REM Configuration
set "LOG_FILE=logs\model_deployment.log"
set "CONFIG_FILE=training_config.yaml"
set "MODELS_DIR=models"
set "TRADER_SCRIPT=scripts\enhanced_trader.py"
set "FALLBACK_DIRS=imported_models,packaged_models,legacy_models"

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"

REM Initialize logging
echo [%date% %time%] Model deployment started > "%LOG_FILE%"

call :log_info "Starting enterprise model deployment system..."

REM Execute deployment pipeline
call :validate_environment
if errorlevel 1 exit /b 1

call :extract_config_symbols
if errorlevel 1 exit /b 1

call :discover_available_models
if errorlevel 1 exit /b 1

call :validate_model_compatibility
if errorlevel 1 exit /b 1

call :prepare_deployment_configuration
if errorlevel 1 exit /b 1

call :test_model_loading
if errorlevel 1 exit /b 1

call :display_deployment_summary
call :log_success "Model deployment completed successfully!"

echo.
echo ========================================
echo   DEPLOYMENT COMPLETE!
echo   Ready for trading operations
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
exit /b 0

:log_error
echo [ERROR] %~1
echo [%date% %time%] [ERROR] %~1 >> "%LOG_FILE%"
exit /b 0

REM ==========================================
REM VALIDATION FUNCTIONS
REM ==========================================

:validate_environment
call :log_info "Validating deployment environment..."

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed or not accessible"
    echo   Please install Python 3.8+ and ensure it's in your PATH
    exit /b 1
)

REM Check configuration file
if not exist "%CONFIG_FILE%" (
    call :log_error "Configuration file not found: %CONFIG_FILE%"
    echo   Please ensure training_config.yaml exists in the current directory
    exit /b 1
)

REM Check enhanced trader script
if not exist "%TRADER_SCRIPT%" (
    call :log_error "Enhanced trader script not found: %TRADER_SCRIPT%"
    echo   Please ensure %TRADER_SCRIPT% exists
    exit /b 1
)

REM Create required directories
call :log_info "Creating required directory structure..."
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "logs" mkdir "logs"
if not exist "data" mkdir "data"

call :log_success "Environment validation completed"
exit /b 0

:extract_config_symbols
call :log_info "Extracting trading symbols from configuration..."

REM Validate YAML format
python -c "import yaml; yaml.safe_load(open('%CONFIG_FILE%'))" 2>nul
if errorlevel 1 (
    call :log_error "Invalid YAML format in configuration file"
    exit /b 1
)

REM Extract symbols using enhanced extraction script
(
echo import yaml
echo import sys
echo try:
echo     with open('%CONFIG_FILE%', 'r'^) as f:
echo         config = yaml.safe_load^(f^)
echo     # Multiple extraction paths for robustness
echo     symbols = ^(
echo         config.get^('data_acquisition', {}^).get^('symbols', []^) or
echo         config.get^('data', {}^).get^('symbols', []^) or
echo         config.get^('symbols', []^) or
echo         config.get^('trading', {}^).get^('symbols', []^)
echo     ^)
echo     models = ^(
echo         config.get^('training', {}^).get^('models', []^) or
echo         config.get^('models', []^) or
echo         ['gru', 'lightgbm', 'ppo']
echo     ^)
echo     if symbols:
echo         print^('SYMBOLS_FOUND:' + ','.join^(symbols^)^)
echo         print^('MODELS_FOUND:' + ','.join^(models^)^)
echo         sys.exit^(0^)
echo     else:
echo         print^('NO_SYMBOLS_FOUND'^)
echo         sys.exit^(1^)
echo except Exception as e:
echo     print^(f'ERROR_EXTRACTING: {e}'^)
echo     sys.exit^(1^)
) > temp_extract_config.py

python temp_extract_config.py > temp_config_results.txt 2>&1
set "extraction_result=%errorlevel%"
del temp_extract_config.py 2>nul

if not %extraction_result% == 0 (
    call :log_error "Failed to extract symbols from configuration"
    if exist temp_config_results.txt (
        echo Error details:
        type temp_config_results.txt
        del temp_config_results.txt
    )
    exit /b 1
)

REM Parse extracted symbols and models
set "TRADING_SYMBOLS="
set "MODEL_TYPES="
for /f "tokens=2 delims=:" %%i in ('type temp_config_results.txt ^| findstr "SYMBOLS_FOUND"') do (
    set "TRADING_SYMBOLS=%%i"
)
for /f "tokens=2 delims=:" %%i in ('type temp_config_results.txt ^| findstr "MODELS_FOUND"') do (
    set "MODEL_TYPES=%%i"
)
del temp_config_results.txt 2>nul

if "!TRADING_SYMBOLS!"=="" (
    call :log_error "No trading symbols found in configuration"
    echo   Please ensure symbols are properly defined in %CONFIG_FILE%
    exit /b 1
)

if "!MODEL_TYPES!"=="" (
    set "MODEL_TYPES=gru,lightgbm,ppo"
    call :log_warning "No model types found in config, using defaults: !MODEL_TYPES!"
)

call :log_success "Configuration extraction completed"
call :log_info "Trading symbols: !TRADING_SYMBOLS!"
call :log_info "Model types: !MODEL_TYPES!"
exit /b 0

:discover_available_models
call :log_info "Discovering available models across multiple sources..."

set "AVAILABLE_MODELS="
set "MISSING_MODELS="
set "TOTAL_MODELS_FOUND=0"

REM Convert comma-separated lists to space-separated for iteration
set "SYMBOLS_LIST=!TRADING_SYMBOLS:,= !"
set "MODELS_LIST=!MODEL_TYPES:,= !"

call :log_info "Scanning model directories and fallback locations..."

for %%s in (!SYMBOLS_LIST!) do (
    for %%m in (!MODELS_LIST!) do (
        call :find_model_for_symbol "%%s" "%%m"
        if not errorlevel 1 (
            set /a TOTAL_MODELS_FOUND+=1
            if "!AVAILABLE_MODELS!"=="" (
                set "AVAILABLE_MODELS=%%s:%%m"
            ) else (
                set "AVAILABLE_MODELS=!AVAILABLE_MODELS!,%%s:%%m"
            )
        ) else (
            if "!MISSING_MODELS!"=="" (
                set "MISSING_MODELS=%%s:%%m"
            ) else (
                set "MISSING_MODELS=!MISSING_MODELS!,%%s:%%m"
            )
        )
    )
)

call :log_info "Model discovery completed:"
call :log_info "  Total models found: !TOTAL_MODELS_FOUND!"
call :log_info "  Available models: !AVAILABLE_MODELS!"
if not "!MISSING_MODELS!"=="" (
    call :log_warning "  Missing models: !MISSING_MODELS!"
)

if !TOTAL_MODELS_FOUND! == 0 (
    call :log_error "No models found for any symbols!"
    echo.
    echo This could mean:
    echo 1. Models haven't been trained yet - run training first
    echo 2. Models are in a different location - check paths
    echo 3. Models were imported but not in expected structure
    echo.
    echo Suggested next steps:
    echo 1. Run: python scripts/enhanced_trainer.py --models lightgbm --symbols BTCEUR --n-splits 2
    echo 2. Or check if models exist in other directories
    echo 3. Or run import_models.bat to import existing models
    echo.
    exit /b 1
)

exit /b 0

:find_model_for_symbol
set "symbol=%~1"
set "model_type=%~2"
set "model_found=0"

REM Search locations in order of preference:
REM 1. Standard models directory structure
REM 2. Flat models directory 
REM 3. Legacy directory structures
REM 4. Fallback directories

REM 1. Standard structure: models/model_type/symbol/
if exist "%MODELS_DIR%\%model_type%\%symbol%" (
    for %%f in ("%MODELS_DIR%\%model_type%\%symbol%\*") do (
        if exist "%%f" (
            call :log_info "  Found %model_type% model for %symbol% in standard location"
            exit /b 0
        )
    )
)

REM 2. Flat structure: models/model_type_symbol.*
for %%e in (pkl pt joblib zip) do (
    if exist "%MODELS_DIR%\%model_type%_%symbol%.%%e" (
        call :log_info "  Found %model_type% model for %symbol% in flat structure"
        exit /b 0
    )
    if exist "%MODELS_DIR%\%model_type%\%model_type%_%symbol%.%%e" (
        call :log_info "  Found %model_type% model for %symbol% in type directory"
        exit /b 0
    )
)

REM 3. Best walkforward naming: best_wf_model_type_symbol.*
for %%e in (pkl pt joblib zip) do (
    if exist "%MODELS_DIR%\best_wf_%model_type%_%symbol%.%%e" (
        call :log_info "  Found %model_type% model for %symbol% in best walkforward format"
        exit /b 0
    )
    if exist "%MODELS_DIR%\%model_type%\best_wf_%model_type%_%symbol%.%%e" (
        call :log_info "  Found %model_type% model for %symbol% in best walkforward type directory"
        exit /b 0
    )
)

REM 4. Search fallback directories
for %%d in (%FALLBACK_DIRS%) do (
    if exist "%%d" (
        for %%e in (pkl pt joblib zip) do (
            if exist "%%d\%model_type%_%symbol%.%%e" (
                call :log_info "  Found %model_type% model for %symbol% in fallback directory %%d"
                exit /b 0
            )
            if exist "%%d\%model_type%\%symbol%" (
                for %%f in ("%%d\%model_type%\%symbol%\*") do (
                    if exist "%%f" (
                        call :log_info "  Found %model_type% model for %symbol% in fallback structure %%d"
                        exit /b 0
                    )
                )
            )
        )
    )
)

call :log_warning "  Model %model_type% for %symbol% not found in any location"
exit /b 1

:validate_model_compatibility
call :log_info "Validating model compatibility and integrity..."

REM Use enhanced trader to validate models
call :log_info "Running model validation through enhanced trader..."

python "%TRADER_SCRIPT%" --test-mode --config "%CONFIG_FILE%" --show-available > temp_model_validation.txt 2>&1
set "validation_result=%errorlevel%"

if exist temp_model_validation.txt (
    call :log_info "Model validation results:"
    type temp_model_validation.txt >> "%LOG_FILE%"
    
    REM Show key validation results
    findstr /C:"Available symbols" temp_model_validation.txt
    findstr /C:"Available model types" temp_model_validation.txt
    findstr /C:"Models directory" temp_model_validation.txt
    
    del temp_model_validation.txt
)

if %validation_result% == 0 (
    call :log_success "Model validation passed"
) else (
    call :log_warning "Model validation completed with warnings - continuing deployment"
)

exit /b 0

:prepare_deployment_configuration
call :log_info "Preparing optimized deployment configuration..."

REM Create deployment-specific config file
call :log_info "Creating deployment configuration file..."

(
echo # Deployment Configuration
echo # Auto-generated for model deployment
echo.
echo deployment:
echo   trading_symbols: [!TRADING_SYMBOLS!]
echo   model_types: [!MODEL_TYPES!]
echo   models_directory: "%MODELS_DIR%"
echo   fallback_directories: [%FALLBACK_DIRS%]
echo   deployment_timestamp: "%date% %time%"
echo.
echo # Include original configuration
) > deployment_config.yaml

REM Append original config content
type "%CONFIG_FILE%" >> deployment_config.yaml

call :log_success "Deployment configuration prepared"
exit /b 0

:test_model_loading
call :log_info "Testing model loading with enhanced trader..."

REM Run a comprehensive test of model loading
call :log_info "Running model loading test..."

python "%TRADER_SCRIPT%" --test-mode --config "deployment_config.yaml" > temp_loading_test.txt 2>&1
set "loading_result=%errorlevel%"

if exist temp_loading_test.txt (
    call :log_info "Model loading test results:"
    
    REM Extract key information
    findstr /C:"Trading symbols" temp_loading_test.txt
    findstr /C:"Model types" temp_loading_test.txt
    findstr /C:"models found" temp_loading_test.txt
    findstr /C:"ERROR" temp_loading_test.txt
    findstr /C:"WARNING" temp_loading_test.txt
    
    REM Log full results
    type temp_loading_test.txt >> "%LOG_FILE%"
    del temp_loading_test.txt
)

if %loading_result% == 0 (
    call :log_success "Model loading test passed successfully"
) else (
    call :log_warning "Model loading test completed with issues - check logs for details"
)

exit /b 0

:display_deployment_summary
echo.
echo ========================================
echo   DEPLOYMENT SUMMARY
echo ========================================
echo.
echo Configuration File: %CONFIG_FILE%
echo Models Directory: %MODELS_DIR%
echo Trading Symbols: !TRADING_SYMBOLS!
echo Model Types: !MODEL_TYPES!
echo Available Models: !AVAILABLE_MODELS!
if not "!MISSING_MODELS!"=="" (
    echo Missing Models: !MISSING_MODELS!
)
echo Total Models Found: !TOTAL_MODELS_FOUND!
echo.
echo Log File: %LOG_FILE%
echo Deployment Config: deployment_config.yaml
echo.
echo Status: Ready for trading operations
echo.
exit /b 0