#!/bin/bash

# Enhanced Trading System Deployment Script for Linux
# ===================================================
# This script initializes and starts the live trading system with comprehensive
# error handling, progress indicators, and automatic dependency management

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo
echo "========================================"
echo "   Enhanced Trading System Deployment"
echo "   Live Trading Initialization"
echo "========================================"
echo

# Configuration
LOG_FILE="logs/deployment.log"
CONFIG_FILE="config_trading.yaml"
TRADER_SCRIPT="scripts/enhanced_trader.py"

# Create logs directory if it doesn't exist
mkdir -p logs

# Initialize log file
echo "[$(date)] Starting trading system deployment" > "$LOG_FILE"

# Logging functions with file output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[$(date)] [INFO] $1" >> "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[$(date)] [SUCCESS] $1" >> "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[$(date)] [WARNING] $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date)] [ERROR] $1" >> "$LOG_FILE"
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check if Python is installed
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
    
    # Get Python version
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
    
    # Check memory availability (basic check)
    if command -v free &> /dev/null; then
        MEMORY_INFO=$(free -h | grep '^Mem:' | awk '{print $2}')
        log_info "System memory: $MEMORY_INFO"
    fi
    
    log_success "System requirements validated"
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
    
    # Check for required scripts
    if [ ! -f "$TRADER_SCRIPT" ]; then
        log_error "Trading script not found: $TRADER_SCRIPT"
        echo "  Please ensure all required files are present"
        exit 1
    fi
    
    # Create necessary directories
    log_info "Creating necessary directories..."
    mkdir -p models
    mkdir -p logs
    mkdir -p data
    mkdir -p backups
    
    log_success "Environment structure validated"
}

# Setup dependencies
setup_dependencies() {
    log_info "Setting up Python dependencies..."
    
    # Upgrade pip first
    $PIP_CMD install --quiet --upgrade pip
    
    # Install core requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing requirements from requirements.txt..."
        $PIP_CMD install --quiet -r requirements.txt || {
            log_warning "Some packages failed to install, continuing..."
        }
    else
        log_info "Installing essential packages manually..."
        $PIP_CMD install --quiet pandas numpy pyyaml python-binance ccxt python-telegram-bot mlflow
    fi
    
    log_success "Dependencies setup completed"
}

# Import models if needed
import_models_if_needed() {
    log_info "Checking for trained models..."
    
    # Check if models directory exists and has content
    if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
        log_warning "No models found, checking for import packages..."
        
        # Look for model packages to import
        found_packages=0
        for zip_file in *.zip; do
            if [ -f "$zip_file" ]; then
                found_packages=1
                log_info "Found model package: $zip_file"
            fi
        done
        
        if [ $found_packages -eq 1 ]; then
            log_info "Attempting automatic model import..."
            if [ -f "./import_models.sh" ]; then
                chmod +x ./import_models.sh
                if ./import_models.sh; then
                    log_success "Models imported successfully"
                else
                    log_error "Automatic model import failed"
                    echo "  Please manually run ./import_models.sh first"
                    exit 1
                fi
            else
                log_error "import_models.sh not found for automatic import"
                echo "  Please manually import models first"
                exit 1
            fi
        else
            log_error "No trained models or import packages found"
            echo "  Please either:"
            echo "    1. Copy a model transfer package (*.zip) to this directory"
            echo "    2. Run ./import_models.sh manually"
            echo "    3. Train models using ./train_models_linux.sh"
            exit 1
        fi
    else
        log_success "Trained models found in models directory"
    fi
}

# Validate models
validate_models() {
    log_info "Validating imported models..."
    
    # Count model files
    model_count=$(find models -name "*.pkl" -o -name "*.pt" -o -name "*.joblib" -o -name "*.zip" 2>/dev/null | wc -l)
    
    # Check for specific model types
    gru_models=0
    lightgbm_models=0
    ppo_models=0
    
    if [ -d "models/gru" ]; then gru_models=1; fi
    if [ -d "models/lightgbm" ]; then lightgbm_models=1; fi
    if [ -d "models/ppo" ]; then ppo_models=1; fi
    
    if [ $model_count -eq 0 ]; then
        log_error "No model files found after import validation"
        exit 1
    fi
    
    log_info "Model validation results:"
    log_info "  Total model files: $model_count"
    if [ $gru_models -eq 1 ]; then
        log_info "  ✓ GRU models available"
    else
        log_warning "  ✗ GRU models not found"
    fi
    
    if [ $lightgbm_models -eq 1 ]; then
        log_info "  ✓ LightGBM models available"
    else
        log_warning "  ✗ LightGBM models not found"
    fi
    
    if [ $ppo_models -eq 1 ]; then
        log_info "  ✓ PPO models available"
    else
        log_warning "  ✗ PPO models not found"
    fi
    
    log_success "Model validation completed"
}

# Configure trading system
configure_trading_system() {
    log_info "Configuring trading system..."
    
    # Check for trading configuration
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ -f "training_config.yaml" ]; then
            log_info "Creating trading configuration from training config..."
            # Create a simplified trading config
            cat > "$CONFIG_FILE" << EOF
# Trading Configuration - Auto-generated
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
            log_info "Trading configuration created"
        else
            log_warning "No trading configuration found, using defaults"
        fi
    fi
    
    # Validate Python environment for trading
    log_info "Validating Python trading environment..."
    $PYTHON_CMD -c "
try:
    import pandas, numpy, yaml
    import ccxt
    print('Core trading dependencies verified')
except ImportError as e:
    print(f'Missing dependency: {e}')
    exit(1)
" 2>/dev/null || {
        log_warning "Some trading dependencies missing, attempting to install..."
        $PIP_CMD install --quiet python-binance ccxt pandas numpy pyyaml
    }
    
    log_success "Trading system configured"
}

# Start trading system
start_trading_system() {
    log_info "Starting live trading system..."
    
    # Final pre-flight checks
    if [ ! -f "$TRADER_SCRIPT" ]; then
        log_error "Trader script not found: $TRADER_SCRIPT"
        exit 1
    fi
    
    # Create startup command
    startup_cmd="$PYTHON_CMD $TRADER_SCRIPT"
    if [ -f "$CONFIG_FILE" ]; then
        startup_cmd="$startup_cmd --config $CONFIG_FILE"
    fi
    
    log_info "Executing trading system startup..."
    log_info "Command: $startup_cmd"
    
    echo
    echo "========================================"
    echo "   🚀 LAUNCHING TRADING SYSTEM 🚀"
    echo "========================================"
    echo
    echo "The trading system is now starting..."
    echo "Monitor the console for real-time updates"
    echo "Log files are available in the logs/ directory"
    echo
    echo "Press Ctrl+C to stop the trading system"
    echo
    
    # Execute the trading system
    $startup_cmd
    trader_exit_code=$?
    
    if [ $trader_exit_code -eq 0 ]; then
        log_success "Trading system exited normally"
    else
        log_warning "Trading system exited with code: $trader_exit_code"
    fi
    
    exit $trader_exit_code
}

# Enhanced model processing and validation
process_enhanced_validation() {
    log_info "Running enhanced model validation..."
    
    # Check for validation script
    if [ -f "validate_models.sh" ]; then
        chmod +x validate_models.sh
        if ./validate_models.sh; then
            log_success "Model validation completed successfully"
        else
            log_warning "Model validation failed. Continuing with warnings..."
            echo "Check logs for validation details."
        fi
    else
        log_info "validate_models.sh not found, skipping validation..."
    fi
    
    # Generate features based on model metadata
    echo
    log_info "Analyzing model metadata for feature generation..."
    if [ -f "scripts/generate_features_from_metadata.py" ]; then
        log_info "Running feature generation from model metadata..."
        if $PYTHON_CMD "scripts/generate_features_from_metadata.py" --models-dir "models" --output-dir "." --verbose; then
            log_success "Feature generation completed successfully"
            log_info "Generated files: feature_config.json, feature_mapping.json, feature_config.yaml"
            
            if [ -f "feature_config.json" ]; then
                log_info "Feature configuration file created successfully"
            fi
            if [ -f "feature_mapping.json" ]; then
                log_info "Feature mapping file created successfully"
            fi
        else
            log_warning "Feature generation from metadata failed. Using default features."
            echo "The bot will attempt to use existing feature configurations."
        fi
    else
        log_error "Feature generation script not found!"
        echo "Expected: scripts/generate_features_from_metadata.py"
        echo "This script is required for proper feature alignment with models."
        exit 1
    fi
}

# Handle configuration migration
handle_config_migration() {
    log_info "Handling configuration migration..."
    
    # Check for trading configuration in different locations
    if [ ! -f "src/config/config_trading.yaml" ]; then
        if [ -f "config_trading.yaml" ]; then
            log_info "Moving config_trading.yaml to src/config directory..."
            mkdir -p src/config
            mv config_trading.yaml src/config/
        else
            log_warning "config_trading.yaml not found!"
            echo "The bot will use default settings."
            echo "You may want to create a configuration file for optimal performance."
        fi
    fi
    
    # Check if enhanced trader script exists, fallback to regular trader
    if [ -f "scripts/enhanced_trader.py" ]; then
        TRADER_SCRIPT="scripts/enhanced_trader.py"
        log_info "Using enhanced trader script"
    elif [ -f "scripts/trader.py" ]; then
        TRADER_SCRIPT="scripts/trader.py"
        log_info "Using standard trader script"
    else
        log_error "No trader script found!"
        echo "Expected: scripts/enhanced_trader.py or scripts/trader.py"
        exit 1
    fi
}

# Enhanced dependency management
enhanced_dependency_check() {
    log_info "Running enhanced dependency check..."
    
    # Check if requirements.txt exists and install from it
    if [ -f "requirements.txt" ]; then
        log_info "Installing dependencies from requirements.txt..."
        if $PIP_CMD install -r requirements.txt; then
            log_info "Dependencies installed from requirements.txt"
        else
            log_warning "Some dependencies from requirements.txt failed to install"
            echo "Continuing with manual dependency check..."
        fi
    fi
    
    # Manual dependency check
    log_info "Checking Python dependencies manually..."
    
    # Essential packages for trading
    IMPORT_NAMES="numpy pandas yaml sklearn requests telegram ccxt"
    for package in $IMPORT_NAMES; do
        log_info "Checking $package..."
        if ! $PYTHON_CMD -c "import $package" 2>/dev/null; then
            log_info "Installing $package..."
            if ! $PIP_CMD install "$package"; then
                log_warning "Failed to install $package"
                echo "You may need to install it manually: $PIP_CMD install $package"
            fi
        fi
    done
    
    log_info "Dependencies check completed"
}

# Main execution function
main() {
    log_info "Enhanced trading system deployment started"
    
    # Execute deployment pipeline
    check_system_requirements
    validate_environment
    setup_dependencies
    enhanced_dependency_check
    import_models_if_needed
    validate_models
    process_enhanced_validation
    handle_config_migration
    configure_trading_system
    
    echo
    echo "========================================"
    echo "     Starting Paper Trading Bot"
    echo "========================================"
    echo
    echo "Configuration:"
    echo "- Trader Script: $TRADER_SCRIPT"
    echo "- Models Directory: models"
    echo "- Logs Directory: logs"
    echo "- Mode: Paper Trading"
    echo
    echo "The bot will start in paper trading mode."
    echo "Press Ctrl+C to stop the bot."
    echo
    echo "Logs will be saved to the 'logs' directory."
    echo "Monitor the logs for trading activity and performance."
    echo
    echo "Starting trader automatically..."
    echo
    
    start_trading_system
}

# Handle script interruption
trap 'log_warning "Trading deployment interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"

log_success "Trading system deployment completed successfully!"
echo
echo "========================================"
echo "   🚀 TRADING SYSTEM DEPLOYED! 🚀"
echo "   Monitor logs/trading.log for activity"
echo "========================================"
echo