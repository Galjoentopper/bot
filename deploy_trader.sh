#!/bin/bash

# Deploy Trader Script - Flexible symbol and model deployment
# ============================================================
# This script reads the configuration file, extracts symbols,
# verifies corresponding models exist, and starts the trader
# with the appropriate symbols and models.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo
echo "========================================"
echo "   Enhanced Trading Bot Deployment"
echo "   Flexible Symbol and Model Setup"
echo "========================================"
echo

# Configuration
CONFIG_FILE="training_config.yaml"
TRADER_SCRIPT="scripts/enhanced_trader.py"
LOG_FILE="logs/deployment.log"
ERROR_LOG="logs/error.log"

# Create logs directory if it doesn't exist
mkdir -p logs

# Initialize logging
echo "[$(date)] Deploy trader started" > "$LOG_FILE"

# Logging functions
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
    echo "[$(date)] [WARNING] $1" >> "$ERROR_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date)] [ERROR] $1" >> "$LOG_FILE"
    echo "[$(date)] [ERROR] $1" >> "$ERROR_LOG"
}

log_info "Starting flexible trader deployment..."

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Configuration file not found: $CONFIG_FILE"
    echo "  Please ensure training_config.yaml exists in the root directory"
    read -p "Press Enter to continue..."
    exit 1
fi

# Check if trader script exists
if [ ! -f "$TRADER_SCRIPT" ]; then
    log_error "Trader script not found: $TRADER_SCRIPT"
    echo "  Please ensure enhanced_trader.py exists in the scripts directory"
    read -p "Press Enter to continue..."
    exit 1
fi

log_info "Extracting symbols from configuration..."

# Extract symbols from configuration file
python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
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

if [ $? -ne 0 ]; then
    log_error "Failed to extract symbols from configuration"
    cat temp_symbols.txt
    rm -f temp_symbols.txt
    read -p "Press Enter to continue..."
    exit 1
fi

# Read extracted symbols
TRADING_SYMBOLS=""
if grep -q "SYMBOLS_FOUND:" temp_symbols.txt; then
    TRADING_SYMBOLS=$(grep "SYMBOLS_FOUND:" temp_symbols.txt | cut -d: -f2)
fi
rm -f temp_symbols.txt

if [ -z "$TRADING_SYMBOLS" ]; then
    log_error "No trading symbols found in configuration"
    echo "  Please ensure symbols are defined in the configuration file"
    echo "  Example configuration:"
    echo "  data_acquisition:"
    echo "    symbols: ['BTCEUR', 'ETHEUR', 'ADAEUR']"
    read -p "Press Enter to continue..."
    exit 1
fi

log_info "Trading symbols found: $TRADING_SYMBOLS"

# Extract model types from configuration
log_info "Extracting model types from configuration..."

python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
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

TRADING_MODELS=""
if grep -q "MODELS_FOUND:" temp_models.txt; then
    TRADING_MODELS=$(grep "MODELS_FOUND:" temp_models.txt | cut -d: -f2)
fi
rm -f temp_models.txt

if [ -z "$TRADING_MODELS" ]; then
    TRADING_MODELS="gru,lightgbm,ppo"
    log_warning "No models specified in config, using defaults: $TRADING_MODELS"
else
    log_info "Model types found: $TRADING_MODELS"
fi

# Verify models directory exists
if [ ! -d "models" ]; then
    log_error "Models directory not found"
    echo "  Please ensure models have been trained and are available in the models directory"
    echo "  Run training first using: python3 scripts/enhanced_trainer.py"
    read -p "Press Enter to continue..."
    exit 1
fi

# Verify models exist for symbols
log_info "Verifying models exist for symbols..."
VERIFIED_SYMBOLS=""
VERIFIED_MODELS=""

# Convert comma-separated strings to arrays
IFS=',' read -ra SYMBOL_ARRAY <<< "$TRADING_SYMBOLS"
IFS=',' read -ra MODEL_ARRAY <<< "$TRADING_MODELS"

for symbol in "${SYMBOL_ARRAY[@]}"; do
    symbol_models_found=0
    available_models=""
    
    for model in "${MODEL_ARRAY[@]}"; do
        if [ -d "models/$model/$symbol" ]; then
            ((symbol_models_found++))
            if [ -z "$available_models" ]; then
                available_models="$model"
            else
                available_models="$available_models,$model"
            fi
            log_info "  Found $model model for $symbol"
        else
            log_warning "  Missing $model model for $symbol"
        fi
    done
    
    if [ $symbol_models_found -ge 1 ]; then
        log_info "Symbol $symbol has $symbol_models_found model(s): $available_models"
        if [ -z "$VERIFIED_SYMBOLS" ]; then
            VERIFIED_SYMBOLS="$symbol"
        else
            VERIFIED_SYMBOLS="$VERIFIED_SYMBOLS,$symbol"
        fi
        
        # Build verified models list
        for vm in $(echo "$available_models" | tr ',' ' '); do
            if [[ "$VERIFIED_MODELS" != *"$vm"* ]]; then
                if [ -z "$VERIFIED_MODELS" ]; then
                    VERIFIED_MODELS="$vm"
                else
                    VERIFIED_MODELS="$VERIFIED_MODELS,$vm"
                fi
            fi
        done
    else
        log_warning "Symbol $symbol has no available models"
    fi
done

if [ -z "$VERIFIED_SYMBOLS" ]; then
    log_error "No symbols have any available models"
    echo "  Please train models for your symbols first"
    echo "  Run: python3 scripts/enhanced_trainer.py --symbols $TRADING_SYMBOLS"
    read -p "Press Enter to continue..."
    exit 1
fi

if [ -z "$VERIFIED_MODELS" ]; then
    VERIFIED_MODELS="$TRADING_MODELS"
fi

log_success "Model verification completed!"
log_info "Verified symbols: $VERIFIED_SYMBOLS"
log_info "Available models: $VERIFIED_MODELS"

# Test the trader configuration
log_info "Testing trader configuration..."
if ! python3 "$TRADER_SCRIPT" --config "$CONFIG_FILE" --symbols "$VERIFIED_SYMBOLS" --models "$VERIFIED_MODELS" --test-mode; then
    log_error "Trader configuration test failed"
    echo "  Please check the error messages above"
    read -p "Press Enter to continue..."
    exit 1
fi

log_success "Trader configuration test passed!"

# Ask user for deployment mode
echo
echo "Select deployment mode:"
echo "1. Single cycle (run once for each symbol)"
echo "2. Continuous trading (run indefinitely)"
echo "3. Test mode only (validate and exit)"
echo
read -p "Enter your choice (1-3): " MODE

case "$MODE" in
    1)
        log_info "Starting single cycle mode..."
        echo
        echo "Starting single trading cycle for symbols: $VERIFIED_SYMBOLS"
        echo "Using models: $VERIFIED_MODELS"
        echo
        python3 "$TRADER_SCRIPT" --config "$CONFIG_FILE" --symbols "$VERIFIED_SYMBOLS" --models "$VERIFIED_MODELS" --single-cycle
        log_info "Single cycle completed"
        ;;
    2)
        log_info "Starting continuous trading mode..."
        echo
        echo "========================================"
        echo "   CONTINUOUS TRADING MODE ACTIVE"
        echo "========================================"
        echo
        echo "Trading symbols: $VERIFIED_SYMBOLS"
        echo "Model types: $VERIFIED_MODELS"
        echo "Configuration: $CONFIG_FILE"
        echo
        echo "Press Ctrl+C to stop trading"
        echo
        python3 "$TRADER_SCRIPT" --config "$CONFIG_FILE" --symbols "$VERIFIED_SYMBOLS" --models "$VERIFIED_MODELS"
        log_info "Continuous trading stopped"
        ;;
    3)
        log_info "Test mode completed - configuration is valid"
        echo
        echo "Configuration validation completed successfully!"
        echo "- Symbols: $VERIFIED_SYMBOLS"
        echo "- Models: $VERIFIED_MODELS"
        echo "- All models are properly accessible"
        ;;
    *)
        log_warning "Invalid selection, defaulting to test mode"
        echo "Configuration validation completed successfully!"
        ;;
esac

echo
echo "========================================"
echo "   Deployment completed"
echo "========================================"
echo
read -p "Press Enter to continue..."
exit 0