@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    Automated Model Validation Script
echo ========================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and ensure it's in your PATH
    exit /b 1
)

REM Check if we're in the correct directory
echo Checking current directory...
if not exist "scripts" (
    echo ERROR: This script must be run from the project root directory
    echo Current directory: %CD%
    echo Expected to find 'scripts' folder here
    exit /b 1
)

REM Check if models directory exists and has content
echo Checking models directory...
if not exist "models" (
    echo ERROR: Models directory not found
    echo Please run import_models.bat first to import your models
    exit /b 1
)

REM Check if models directory is empty
dir /b "models" 2>nul | findstr . >nul
if errorlevel 1 (
    echo WARNING: Models directory is empty
    echo Please run import_models.bat first to import your models
    exit /b 1
)

echo Validating imported models...
echo.

REM Check if validate_models.py exists
if exist "scripts\validate_models.py" (
    echo Running model validation...
    python scripts\validate_models.py --models-dir models --verbose
    if errorlevel 1 (
        echo ERROR: Model validation script failed
        echo Check the output above for details
        exit /b 1
    ) else (
        echo Model validation completed successfully!
        echo.
        echo You can now proceed with trading
        exit /b 0
    )
) else (
    echo WARNING: validate_models.py not found in scripts directory
    echo Performing basic validation instead...
    echo.
    
    REM Basic validation - check for common model file types
    set "found_models=0"
    for /r "models" %%f in (*.pkl *.pt *.joblib) do (
        set "found_models=1"
        echo Found model file: %%f
    )
    
    if !found_models! equ 0 (
        echo ERROR: No model files found in models directory
        echo Expected file types: .pkl, .pt, .joblib
        exit /b 1
    ) else (
        echo Basic validation passed - found model files
        echo NOTE: This is a basic check only. For full validation, ensure validate_models.py exists
        echo.
        echo Continuing with trading despite validation warnings...
        exit /b 0
    )
)

echo Model validation completed!
echo.