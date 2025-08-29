#!/bin/bash

# Enhanced Training Script for Linux Environments
# ===============================================
# This script reads the centralized configuration file to train machine learning models 
# with specified parameters and hyperparameters, including automated zip archive generation

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/training_config.yaml"
LOG_FILE="$SCRIPT_DIR/logs/training.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Progress indicator
show_progress() {
    local current=$1
    local total=$2
    local step_name="$3"
    local percent=$((current * 100 / total))
    local bar_length=50
    local filled_length=$((percent * bar_length / 100))
    
    printf "\r${BLUE}[%3d%%]${NC} " "$percent"
    printf "["
    for ((i=0; i<filled_length; i++)); do printf "#"; done
    for ((i=filled_length; i<bar_length; i++)); do printf " "; done
    printf "] %s" "$step_name"
}

echo -e "${GREEN}"
echo "=============================================="
echo "   Enhanced Linux Training System"
echo "   Automated ML Model Training Pipeline"
echo "=============================================="
echo -e "${NC}"

# Check dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check for Python 3
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check for pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        log_error "pip is required but not installed"
        exit 1
    fi
    
    # Check project structure
    if [ ! -d "src" ] || [ ! -d "scripts" ]; then
        log_error "This script must be run from the project root directory"
        log_error "Current directory: $(pwd)"
        exit 1
    fi
    
    # Check centralized config file
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Centralized configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    log_success "Dependencies validated"
}

# Initialize ML runtime environment
initialize_ml_runtime() {
    log_info "Initializing ML runtime environment..."
    
    # Create necessary directories
    mkdir -p logs models/exports models/packages models/metadata checkpoints
    
    # Skip the problematic startup_init import (known issue)
    # The enhanced_trainer.py handles initialization gracefully with warnings
    log_success "ML runtime directories created"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing/updating Python dependencies..."
    
    # Use pip3 if available, otherwise pip
    local pip_cmd="pip3"
    if ! command -v pip3 &> /dev/null; then
        pip_cmd="pip"
    fi
    
    # Install core dependencies
    $pip_cmd install --quiet --upgrade pip setuptools wheel
    $pip_cmd install --quiet -r requirements.txt
    
    log_success "Python dependencies installed"
}

# Read symbols and models from centralized config
read_config() {
    log_info "Reading centralized configuration..."
    
    # Use Python to extract config values
    SYMBOLS=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
symbols = config.get('data_acquisition', {}).get('symbols', ['BTCEUR', 'ETHEUR', 'ADAEUR'])
print(' '.join(symbols))")
    
    MODELS=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
models = config.get('training', {}).get('models', ['gru', 'lightgbm', 'ppo'])
print(' '.join(models))")
    
    log_info "Training symbols: $SYMBOLS"
    log_info "Training models: $MODELS"
    log_success "Configuration loaded successfully"
}

# Train models
train_models() {
    log_info "Starting model training process..."
    
    local total_steps=4
    local current_step=0
    
    # Step 1: Data preparation
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps "Preparing training data"
    echo ""
    
    # Check if data exists
    if [ ! -d "data" ] || [ ! -f "data/btceur_30m.db" ]; then
        log_warning "Training data not found. Run fetch_training_data.sh first"
        log_info "Attempting to fetch data automatically..."
        if [ -f "fetch_training_data.sh" ]; then
            bash fetch_training_data.sh
        else
            log_error "Cannot fetch data automatically. Please run fetch_training_data.sh first"
            exit 1
        fi
    fi
    
    # Step 2: Start training
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps "Training machine learning models"
    echo ""
    
    # Run enhanced trainer with centralized config
    if python3 scripts/enhanced_trainer.py \
        --config "$CONFIG_FILE" \
        --symbols $SYMBOLS \
        --models $MODELS \
        --create-transfer-bundle \
        --export-dir "models/exports" \
        "$@"; then
        log_success "Model training completed successfully"
    else
        log_error "Model training failed"
        exit 1
    fi
    
    # Step 3: Create zip archive
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps "Creating deployment archive"
    echo ""
    
    create_deployment_archive
    
    # Step 4: Finalize
    current_step=$((current_step + 1))
    show_progress $current_step $total_steps "Training pipeline complete"
    echo ""
}

# Create deployment archive
create_deployment_archive() {
    log_info "Creating automated zip archive for deployment..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local archive_name="trading_models_${timestamp}.zip"
    local temp_dir="/tmp/trading_models_export"
    
    # Create temporary directory
    mkdir -p "$temp_dir"
    
    # Copy models and necessary files
    if [ -d "models" ]; then
        cp -r models "$temp_dir/"
    fi
    
    # Copy configuration files
    cp "$CONFIG_FILE" "$temp_dir/training_config.yaml"
    
    # Copy deployment scripts
    cp import_models.bat "$temp_dir/" 2>/dev/null || true
    cp deploy_trading.bat "$temp_dir/" 2>/dev/null || true
    
    # Copy requirements
    cp requirements.txt "$temp_dir/" 2>/dev/null || true
    
    # Create archive
    cd "$temp_dir"
    zip -r "$SCRIPT_DIR/$archive_name" . -q
    cd "$SCRIPT_DIR"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_success "Deployment archive created: $archive_name"
    echo ""
    echo -e "${GREEN}🎉 DEPLOYMENT PACKAGE READY 🎉${NC}"
    echo -e "${YELLOW}Archive: $archive_name${NC}"
    echo -e "${YELLOW}Transfer this file to your Windows trading computer${NC}"
    echo ""
}

# Show completion summary
show_completion_summary() {
    echo ""
    echo -e "${GREEN}"
    echo "=============================================="
    echo "   🎯 TRAINING PIPELINE COMPLETED! 🎯"
    echo "=============================================="
    echo -e "${NC}"
    
    log_success "All models trained successfully"
    log_success "Deployment archive created and ready for transfer"
    
    echo ""
    echo -e "${YELLOW}📋 NEXT STEPS:${NC}"
    echo "1. Transfer the deployment archive to your Windows trading computer"
    echo "2. Run import_models.bat on the Windows computer"
    echo "3. Run deploy_trading.bat to start live trading"
    echo ""
    
    # Show archive details
    if ls trading_models_*.zip &>/dev/null; then
        local archive_file=$(ls -t trading_models_*.zip | head -1)
        local archive_size=$(du -h "$archive_file" | cut -f1)
        echo -e "${BLUE}📦 Archive Details:${NC}"
        echo "   File: $archive_file"
        echo "   Size: $archive_size"
        echo ""
    fi
}

# Main execution
main() {
    # Parse command line arguments
    local symbols_override=""
    local models_override=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --symbols)
                symbols_override="$2"
                shift 2
                ;;
            --models)
                models_override="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                # Pass through other arguments to the trainer
                break
                ;;
        esac
    done
    
    log_info "Starting enhanced Linux training pipeline..."
    
    # Execute pipeline steps
    check_dependencies
    initialize_ml_runtime
    install_dependencies
    read_config
    
    # Override config values if provided
    if [ -n "$symbols_override" ]; then
        SYMBOLS="$symbols_override"
        log_info "Symbols overridden: $SYMBOLS"
    fi
    
    if [ -n "$models_override" ]; then
        MODELS="$models_override"
        log_info "Models overridden: $MODELS"
    fi
    
    train_models "$@"
    show_completion_summary
}

# Usage information
show_usage() {
    echo "Usage: $0 [OPTIONS] [TRAINER_OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --symbols SYMBOLS   Override training symbols (e.g., 'BTCEUR ETHEUR')"
    echo "  --models MODELS     Override training models (e.g., 'gru lightgbm')"
    echo "  --help              Show this help message"
    echo ""
    echo "Trainer Options:"
    echo "  All additional options are passed to the enhanced trainer"
    echo ""
    echo "Examples:"
    echo "  $0                           # Train all models for all symbols from config"
    echo "  $0 --symbols BTCEUR          # Train only BTCEUR models"
    echo "  $0 --models gru               # Train only GRU models"
    echo "  $0 --resume                   # Resume from checkpoint"
}

# Handle script interruption
trap 'log_warning "Training interrupted by user"; exit 1' INT TERM

# Ensure logs directory exists
mkdir -p logs

# Run main function
main "$@"

echo -e "${GREEN}Training pipeline completed successfully!${NC}"