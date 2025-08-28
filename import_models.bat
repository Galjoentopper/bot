@echo off
setlocal enabledelayedexpansion

echo ========================================
echo     Automated Model Import Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    echo Exiting automatically...
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "scripts" (
    echo ERROR: Please run this script from the Bot_kilo root directory
    echo Current directory: %CD%
    echo Exiting automatically...
    exit /b 1
)

echo Checking for transfer packages...
echo.

REM Look for zip files and process them directly
set "found_any=0"
for %%f in (*.zip) do (
    set "found_any=1"
    echo Found package: %%f
    echo.
    
    echo Processing package: %%f
    echo.
    
    REM Check if this is a bundle format (contains bundle_info.json)
    echo Checking package format...
    powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $zip = [System.IO.Compression.ZipFile]::OpenRead('%%f'); $hasBundle = $zip.Entries | Where-Object { $_.Name -eq 'bundle_info.json' }; $zip.Dispose(); if ($hasBundle) { exit 0 } else { exit 1 }"
    if not errorlevel 1 (
        echo Detected bundle format with bundle_info.json
        echo Extracting bundle to temporary directory...
        
        REM Create temp directory and extract
        if exist "temp_import" rmdir /s /q "temp_import"
        mkdir "temp_import"
        powershell -Command "Expand-Archive -Path '%%f' -DestinationPath 'temp_import' -Force"
        
        REM Check if extraction was successful
        if exist "temp_import\bundle_info.json" (
            echo Bundle extracted successfully
            echo Importing models using bundle import script...
            
            REM Change to temp directory and run import
            pushd "temp_import"
            python "import_models.py"
            set "import_result=!errorlevel!"
            popd
            
            if !import_result! equ 0 (
                echo Models imported successfully from bundle!
                
                REM Move models from temp_import to main directory
                if exist "temp_import\models" (
                    echo Moving models to main directory...
                    if exist "models" (
                        echo Merging with existing models directory...
                        xcopy "temp_import\models\*" "models\" /E /Y /I >nul 2>&1
                    ) else (
                        echo Creating models directory...
                        move "temp_import\models" "." >nul 2>&1
                    )
                    
                    if exist "models" (
                        echo Models successfully moved to main directory
                    ) else (
                        echo WARNING: Failed to move models to main directory
                    )
                ) else (
                    echo WARNING: No models directory found in temp_import
                )
                
                REM Clean up temp directory
                rmdir /s /q "temp_import"
                
                REM Move processed package to processed folder
                if not exist "processed_packages" mkdir "processed_packages"
                move "%%f" "processed_packages\" >nul 2>&1
                if errorlevel 1 (
                    echo WARNING: Could not move processed package to processed_packages folder
                ) else (
                    echo Package moved to processed_packages folder
                )
            ) else (
                echo ERROR: Failed to import models from bundle
                rmdir /s /q "temp_import"
            )
        ) else (
            echo ERROR: Failed to extract bundle
            rmdir /s /q "temp_import"
        )
    ) else (
        REM Try legacy cross_platform_transfer format
        echo Trying legacy transfer format...
        python "scripts\cross_platform_transfer.py" validate --package "%%f"
        if errorlevel 1 (
            echo ERROR: Package validation failed for %%f
            echo Skipping this package...
            echo Continuing with next package...
            echo.
        ) else (
            echo Package validation successful!
            echo.
            
            REM Import the package
            echo Importing models from package...
            python "scripts\cross_platform_transfer.py" import --package "%%f" --destination "models"
            if errorlevel 1 (
                echo ERROR: Failed to import models from %%f
                echo.
            ) else (
                echo Models imported successfully!
                echo.
                
                REM Move processed package to processed folder
                if not exist "processed_packages" mkdir "processed_packages"
                move "%%f" "processed_packages\" >nul 2>&1
                if errorlevel 1 (
                    echo WARNING: Could not move processed package to processed_packages folder
                ) else (
                    echo Package moved to processed_packages folder
                )
                echo.
            )
        )
    )
    echo.
)

if "!found_any!" == "0" (
    echo No transfer packages found in current directory.
    echo.
    echo Expected files:
    echo   - model_transfer_bundle_*.zip
    echo   - transfer_package.zip
    echo   - Any .zip file containing model data
    echo.
    echo Please place your model transfer package in this directory and try again.
    echo Exiting automatically...
    exit /b 1
)

echo ========================================
echo Model import process completed!
echo ========================================
echo.
echo Check the models directory for imported files.
echo Processed packages have been moved to processed_packages folder.
echo.
echo Import completed automatically - no user input required.