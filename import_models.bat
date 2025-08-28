@echo off
setlocal enabledelayedexpansion

REM Enhanced Windows Model Import Script
REM ====================================
REM This script extracts and configures imported models into the proper directory structure
REM with robust error handling, progress indicators, and automatic dependency checking

echo.
echo ========================================
echo   Enhanced Model Import System
echo   Windows Deployment Configuration
echo ========================================
echo.

REM Colors for Windows console (if supported)
set "INFO_COLOR=36"
set "SUCCESS_COLOR=32"
set "WARNING_COLOR=33"
set "ERROR_COLOR=31"

REM Logging function simulation
call :log_info "Starting Windows model import process..."

REM Check system requirements
call :check_dependencies
if errorlevel 1 exit /b 1

REM Validate environment
call :validate_environment
if errorlevel 1 exit /b 1

REM Process model packages
call :process_model_packages
if errorlevel 1 exit /b 1

REM Validate imported models
call :validate_imported_models
if errorlevel 1 exit /b 1

REM Configure for Windows deployment
call :configure_for_windows
if errorlevel 1 exit /b 1

call :log_success "Model import completed successfully!"
echo.
echo ========================================
echo   Import Process Complete!
echo   Ready for trading deployment
echo ========================================
echo.
echo Next step: Run deploy_trading.bat
pause
exit /b 0

REM ==========================================
REM FUNCTION DEFINITIONS
REM ==========================================

:log_info
echo [INFO] %~1
exit /b 0

:log_success
echo [SUCCESS] %~1
exit /b 0

:log_warning
echo [WARNING] %~1
exit /b 0

:log_error
echo [ERROR] %~1
exit /b 0

:check_dependencies
call :log_info "Checking system dependencies..."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed or not in PATH"
    echo   Please install Python 3.8+ from https://python.org
    echo   Make sure to add Python to PATH during installation
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :log_info "Python version: %PYTHON_VERSION%"

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    call :log_error "pip is not available"
    echo   pip should be included with Python 3.8+
    exit /b 1
)

REM Install/check required packages
call :log_info "Checking/installing required packages..."
pip install --quiet --upgrade pyyaml pathlib zipfile36 2>nul
if errorlevel 1 (
    call :log_warning "Failed to install some packages, continuing anyway..."
)

call :log_success "Dependencies validated"
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

REM Create necessary directories
if not exist "models" mkdir "models"
if not exist "models\packages" mkdir "models\packages"
if not exist "models\imported" mkdir "models\imported"
if not exist "models\metadata" mkdir "models\metadata"
if not exist "processed_packages" mkdir "processed_packages"
if not exist "logs" mkdir "logs"

call :log_success "Environment structure validated"
exit /b 0

:process_model_packages
call :log_info "Scanning for model transfer packages..."

set "found_packages=0"
set "successful_imports=0"

REM Process all ZIP files in current directory
for %%f in (*.zip) do (
    set "found_packages=1"
    call :log_info "Found package: %%f"
    
    call :process_single_package "%%f"
    if not errorlevel 1 (
        set /a successful_imports+=1
    )
)

if !found_packages! == 0 (
    call :log_error "No transfer packages found in current directory"
    echo.
    echo   Expected files:
    echo     - trading_models_*.zip  ^(from train_models_linux.sh^)
    echo     - model_transfer_*.zip
    echo     - Any .zip file containing trained models
    echo.
    echo   Please copy your model transfer package to this directory and try again.
    exit /b 1
)

if !successful_imports! == 0 (
    call :log_error "No packages were successfully imported"
    exit /b 1
)

call :log_success "Processed %successful_imports% package(s) successfully"
exit /b 0

:process_single_package
set "package_file=%~1"
call :log_info "Processing package: %package_file%"

REM Create temporary extraction directory
set "temp_dir=temp_import_%RANDOM%"
if exist "%temp_dir%" rmdir /s /q "%temp_dir%"
mkdir "%temp_dir%"

REM Extract package
call :log_info "Extracting package..."
powershell -Command "try { Expand-Archive -Path '%package_file%' -DestinationPath '%temp_dir%' -Force; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    call :log_error "Failed to extract package: %package_file%"
    rmdir /s /q "%temp_dir%"
    exit /b 1
)

REM Check package contents and import
if exist "%temp_dir%\models" (
    call :import_models_from_package "%temp_dir%"
    set "import_result=!errorlevel!"
) else if exist "%temp_dir%\bundle_info.json" (
    call :import_bundle_package "%temp_dir%"
    set "import_result=!errorlevel!"
) else (
    call :log_error "Unknown package format: %package_file%"
    set "import_result=1"
)

REM Cleanup and move processed package
rmdir /s /q "%temp_dir%"

if !import_result! == 0 (
    call :log_success "Successfully imported: %package_file%"
    move "%package_file%" "processed_packages\" >nul 2>&1
    if errorlevel 1 (
        call :log_warning "Could not move processed package to processed_packages folder"
    )
) else (
    call :log_error "Failed to import: %package_file%"
)

exit /b !import_result!

:import_models_from_package
set "source_dir=%~1"
call :log_info "Importing standard model package..."

REM Copy models to appropriate directories
if exist "%source_dir%\models" (
    xcopy "%source_dir%\models\*" "models\" /E /Y /I >nul 2>&1
    if errorlevel 1 (
        call :log_error "Failed to copy models"
        exit /b 1
    )
)

REM Copy configuration if present
if exist "%source_dir%\training_config.yaml" (
    copy "%source_dir%\training_config.yaml" "." >nul 2>&1
    call :log_info "Configuration file imported"
)

exit /b 0

:import_bundle_package
set "source_dir=%~1"
call :log_info "Importing bundle package format..."

REM Check for Python import script
if exist "%source_dir%\import_models.py" (
    pushd "%source_dir%"
    python import_models.py
    set "bundle_result=!errorlevel!"
    popd
    
    if !bundle_result! == 0 (
        REM Copy results to main directory
        if exist "%source_dir%\models" (
            xcopy "%source_dir%\models\*" "models\" /E /Y /I >nul 2>&1
        )
        exit /b 0
    ) else (
        call :log_error "Bundle import script failed"
        exit /b 1
    )
) else (
    call :log_error "Bundle package missing import script"
    exit /b 1
)

:validate_imported_models
call :log_info "Validating imported models..."

set "model_count=0"

REM Count imported model files
for /r "models" %%f in (*.pkl *.pt *.joblib *.zip) do (
    set /a model_count+=1
)

if !model_count! == 0 (
    call :log_error "No model files found after import"
    echo   Expected file types: .pkl, .pt, .joblib, .zip
    exit /b 1
)

call :log_success "Found %model_count% model files"

REM Check for required model types
set "has_gru=0"
set "has_lightgbm=0"
set "has_ppo=0"

if exist "models\gru" set "has_gru=1"
if exist "models\lightgbm" set "has_lightgbm=1"
if exist "models\ppo" set "has_ppo=1"

call :log_info "Model availability check:"
if !has_gru! == 1 (
    call :log_info "  ✓ GRU models found"
) else (
    call :log_warning "  ✗ GRU models not found"
)

if !has_lightgbm! == 1 (
    call :log_info "  ✓ LightGBM models found"
) else (
    call :log_warning "  ✗ LightGBM models not found"
)

if !has_ppo! == 1 (
    call :log_info "  ✓ PPO models found"
) else (
    call :log_warning "  ✗ PPO models not found"
)

exit /b 0

:configure_for_windows
call :log_info "Configuring models for Windows deployment..."

REM Create Windows-specific configuration
if exist "training_config.yaml" (
    call :log_info "Creating Windows deployment configuration..."
    
    REM Create simplified trading config for Windows
    (
        echo # Windows Trading Configuration
        echo # Generated automatically by import_models.bat
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
    ) > "config_trading.yaml"
    
    call :log_info "Windows trading configuration created"
)

REM Set proper file permissions (if needed)
call :log_info "Setting file permissions for Windows..."

call :log_success "Windows configuration complete"
exit /b 0