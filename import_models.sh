#!/bin/bash

# Enhanced Linux Model Import Script
# ==================================
# This script extracts and configures imported models into the proper directory structure
# with robust error handling, progress indicators, and automatic dependency checking

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo
echo "========================================"
echo "   Enhanced Model Import System"
echo "   Linux Deployment Configuration"
echo "========================================"
echo

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            log_error "Python is not installed or not in PATH"
            echo "  Please install Python 3.8+ from your distribution's package manager"
            echo "  Example: sudo apt-get install python3 python3-pip"
            exit 1
        else
            PYTHON_CMD="python"
        fi
    else
        PYTHON_CMD="python3"
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
        log_error "pip is not available"
        echo "  pip should be included with Python 3.8+"
        echo "  Try: sudo apt-get install python3-pip"
        exit 1
    fi
    
    # Determine pip command
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    else
        PIP_CMD="pip"
    fi
    
    # Install/check required packages
    log_info "Checking/installing required packages..."
    $PIP_CMD install --quiet --upgrade pyyaml pathlib2 > /dev/null 2>&1 || {
        log_warning "Failed to install some packages, continuing anyway..."
    }
    
    # Check for unzip utility
    if ! command -v unzip &> /dev/null; then
        log_error "unzip utility is not installed"
        echo "  Please install unzip: sudo apt-get install unzip"
        exit 1
    fi
    
    log_success "Dependencies validated"
}

# Validate environment
validate_environment() {
    log_info "Validating environment structure..."
    
    # Check if we're in the correct directory
    if [ ! -d "scripts" ]; then
        log_error "Please run this script from the Bot_kilo root directory"
        echo "  Current directory: $(pwd)"
        echo "  Expected to find 'scripts' folder here"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p models/packages
    mkdir -p models/imported
    mkdir -p models/metadata
    mkdir -p processed_packages
    mkdir -p logs
    
    log_success "Environment structure validated"
}

# Process model packages
process_model_packages() {
    log_info "Scanning for model transfer packages..."
    
    found_packages=0
    successful_imports=0
    
    # Process all ZIP files in current directory
    for zip_file in *.zip; do
        if [ -f "$zip_file" ]; then
            found_packages=1
            log_info "Found package: $zip_file"
            
            if process_single_package "$zip_file"; then
                ((successful_imports++))
            fi
        fi
    done
    
    if [ $found_packages -eq 0 ]; then
        log_error "No transfer packages found in current directory"
        echo
        echo "  Expected files:"
        echo "    - trading_models_*.zip  (from train_models_linux.sh)"
        echo "    - model_transfer_*.zip"
        echo "    - Any .zip file containing trained models"
        echo
        echo "  Please copy your model transfer package to this directory and try again."
        exit 1
    fi
    
    if [ $successful_imports -eq 0 ]; then
        log_error "No packages were successfully imported"
        exit 1
    fi
    
    log_success "Processed $successful_imports package(s) successfully"
}

# Process single package
process_single_package() {
    local package_file="$1"
    log_info "Processing package: $package_file"
    
    # Create temporary extraction directory
    local temp_dir="temp_import_$$"
    if [ -d "$temp_dir" ]; then
        rm -rf "$temp_dir"
    fi
    mkdir "$temp_dir"
    
    # Extract package
    log_info "Extracting package..."
    if ! unzip -q "$package_file" -d "$temp_dir"; then
        log_error "Failed to extract package: $package_file"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Check package contents and import
    local import_result=1
    if [ -d "$temp_dir/models" ]; then
        import_models_from_package "$temp_dir"
        import_result=$?
    elif [ -f "$temp_dir/bundle_info.json" ]; then
        import_bundle_package "$temp_dir"
        import_result=$?
    else
        log_error "Unknown package format: $package_file"
        import_result=1
    fi
    
    # Cleanup and move processed package
    rm -rf "$temp_dir"
    
    if [ $import_result -eq 0 ]; then
        log_success "Successfully imported: $package_file"
        mv "$package_file" "processed_packages/" 2>/dev/null || {
            log_warning "Could not move processed package to processed_packages folder"
        }
    else
        log_error "Failed to import: $package_file"
    fi
    
    return $import_result
}

# Import models from package
import_models_from_package() {
    local source_dir="$1"
    log_info "Importing standard model package..."
    
    # Copy models to appropriate directories
    if [ -d "$source_dir/models" ]; then
        cp -r "$source_dir/models/"* "models/" 2>/dev/null || {
            log_error "Failed to copy models"
            return 1
        }
    fi
    
    # Copy configuration if present
    if [ -f "$source_dir/training_config.yaml" ]; then
        cp "$source_dir/training_config.yaml" "." 2>/dev/null
        log_info "Configuration file imported"
    fi
    
    return 0
}

# Import bundle package
import_bundle_package() {
    local source_dir="$1"
    log_info "Importing bundle package format..."
    
    # Check for Python import script
    if [ -f "$source_dir/import_models.py" ]; then
        pushd "$source_dir" > /dev/null
        if $PYTHON_CMD import_models.py; then
            popd > /dev/null
            
            # Copy results to main directory
            if [ -d "$source_dir/models" ]; then
                cp -r "$source_dir/models/"* "models/" 2>/dev/null
            fi
            return 0
        else
            popd > /dev/null
            log_error "Bundle import script failed"
            return 1
        fi
    else
        log_error "Bundle package missing import script"
        return 1
    fi
}

# Validate imported models
validate_imported_models() {
    log_info "Validating imported models..."
    
    # Count imported model files
    model_count=$(find models -name "*.pkl" -o -name "*.pt" -o -name "*.joblib" -o -name "*.zip" 2>/dev/null | wc -l)
    
    if [ $model_count -eq 0 ]; then
        log_error "No model files found after import"
        echo "  Expected file types: .pkl, .pt, .joblib, .zip"
        exit 1
    fi
    
    log_success "Found $model_count model files"
    
    # Check for required model types
    has_gru=0
    has_lightgbm=0
    has_ppo=0
    
    if [ -d "models/gru" ]; then has_gru=1; fi
    if [ -d "models/lightgbm" ]; then has_lightgbm=1; fi
    if [ -d "models/ppo" ]; then has_ppo=1; fi
    
    log_info "Model availability check:"
    if [ $has_gru -eq 1 ]; then
        log_info "  ✓ GRU models found"
    else
        log_warning "  ✗ GRU models not found"
    fi
    
    if [ $has_lightgbm -eq 1 ]; then
        log_info "  ✓ LightGBM models found"
    else
        log_warning "  ✗ LightGBM models not found"
    fi
    
    if [ $has_ppo -eq 1 ]; then
        log_info "  ✓ PPO models found"
    else
        log_warning "  ✗ PPO models not found"
    fi
}

# Configure for Linux deployment
configure_for_linux() {
    log_info "Configuring models for Linux deployment..."
    
    # Create Linux-specific configuration
    if [ -f "training_config.yaml" ]; then
        log_info "Creating Linux deployment configuration..."
        
        # Create simplified trading config for Linux
        cat > "config_trading.yaml" << EOF
# Linux Trading Configuration
# Generated automatically by import_models.sh
trading:
  initial_balance: 10000
  max_position_size: 0.1
  transaction_fee: 0.001
  slippage: 0.0005
  model_weights:
    gru: 0.45
    lightgbm: 0.45
    ppo: 0.1
models:
  lightgbm:
    enabled: true
  gru:
    enabled: true
  ppo:
    enabled: true
notifications:
  telegram:
    enabled: true
    bot_token: '7733436451:AAH6Sls8uL4fEgd6Ty7VEKSBIMauhaVkN4c'
    chat_id: '7988790407'
EOF
        
        log_info "Linux trading configuration created"
    fi
    
    # Set proper file permissions
    log_info "Setting file permissions for Linux..."
    chmod +x *.sh 2>/dev/null || true
    
    log_success "Linux configuration complete"
}

# Main execution
main() {
    log_info "Starting Linux model import process..."
    
    # Check system requirements
    check_dependencies
    
    # Validate environment
    validate_environment
    
    # Process model packages
    process_model_packages
    
    # Validate imported models
    validate_imported_models
    
    # Configure for Linux deployment
    configure_for_linux
    
    log_success "Model import completed successfully!"
    echo
    echo "========================================"
    echo "   Import Process Complete!"
    echo "   Ready for trading deployment"
    echo "========================================"
    echo
    echo "Next step: Run ./deploy_trading.sh"
    echo "Press Enter to continue..."
    read -r
}

# Handle script interruption
trap 'log_warning "Import process interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"