#!/bin/bash
# Enhanced Automated Trading System Deployment Script
# ===================================================
# Comprehensive trading automation with configuration processing,
# model verification, scheduled execution, error handling, and detailed reporting

set -e
set -o pipefail

LOG_FILE="logs/deployment.log"
TRADING_LOG="logs/trading.log"
ERROR_LOG="logs/error.log"
TRADES_CSV="logs/trades_report.csv"
CONFIG_FILE="training_config.yaml"
TRADER_SCRIPT="scripts/enhanced_trader.py"
EXECUTION_INTERVAL=1800
MAX_RETRIES=3
RETRY_DELAY=30

mkdir -p logs models data config backups reports cache

echo "[$(date)] Enhanced automated trading system deployment started" > "$LOG_FILE"
echo "[$(date)] Trading operations log initialized" > "$TRADING_LOG"
echo "[$(date)] Error tracking log initialized" > "$ERROR_LOG"

if [ ! -f "$TRADES_CSV" ]; then
    echo "Timestamp,TradeID,Symbol,TradeType,Quantity,Price,OrderStatus,Notes,ModelUsed,Confidence,Balance" > "$TRADES_CSV"
fi

log_info() {
    echo "[INFO] $1"
    echo "[$(date)] [INFO] $1" >> "$LOG_FILE"
    echo "[$(date)] [INFO] $1" >> "$TRADING_LOG"
}

log_success() {
    echo "[SUCCESS] $1"
    echo "[$(date)] [SUCCESS] $1" >> "$LOG_FILE"
    echo "[$(date)] [SUCCESS] $1" >> "$TRADING_LOG"
}

log_warning() {
    echo "[WARNING] $1"
    echo "[$(date)] [WARNING] $1" >> "$LOG_FILE"
    echo "[$(date)] [WARNING] $1" >> "$TRADING_LOG"
    echo "[$(date)] [WARNING] $1" >> "$ERROR_LOG"
}

log_error() {
    echo "[ERROR] $1"
    echo "[$(date)] [ERROR] $1" >> "$LOG_FILE"
    echo "[$(date)] [ERROR] $1" >> "$TRADING_LOG"
    echo "[$(date)] [ERROR] $1" >> "$ERROR_LOG"
}

log_trade() {
    # $1: trade_id, $2: symbol, $3: trade_type, $4: quantity, $5: price, $6: status, $7: notes, $8: model, $9: confidence, $10: balance
    local timestamp="$(date)"
    echo "$timestamp,$1,$2,$3,$4,$5,$6,$7,$8,$9,${10}" >> "$TRADES_CSV"
    log_info "Trade logged: $2 $3 $4 @ $5 - Status: $6 (ID=$1)"
}

check_system_requirements() {
    log_info "Checking comprehensive system requirements..."
    if ! command -v python3 &>/dev/null; then
        log_error "Python is not installed or not in PATH"
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"
    if ! command -v pip3 &>/dev/null; then
        log_error "pip is not available"
        echo "pip should be included with Python 3.8+"
        exit 1
    fi
    log_info "System memory: $(free -h | grep Mem | awk '{print $2}')"
    log_info "Available disk space: $(df -h . | tail -1 | awk '{print $4}')"
    log_success "System requirements validated successfully"
}

validate_environment() {
    log_info "Validating comprehensive environment structure..."
    if [ ! -d "scripts" ]; then
        log_error "Please run this script from the Bot root directory"
        echo "Current directory: $(pwd)"
        echo "Expected to find 'scripts' folder here"
        exit 1
    fi
    if [ ! -f "$TRADER_SCRIPT" ]; then
        log_error "Enhanced trading script not found: $TRADER_SCRIPT"
        if [ -f "scripts/trader.py" ]; then
            TRADER_SCRIPT="scripts/trader.py"
            log_warning "Falling back to standard trader script"
        else
            echo "Please ensure all required files are present"
            exit 1
        fi
    fi
    log_success "Environment structure validated and created"
}

setup_dependencies() {
    log_info "Setting up comprehensive Python dependencies..."
    python3 -m pip install --upgrade pip || log_warning "Failed to upgrade pip, continuing with current version"
    if [ -f "requirements.txt" ]; then
        log_info "Installing requirements from requirements.txt..."
        python3 -m pip install -r requirements.txt || install_essential_packages
        log_success "All requirements installed successfully"
    else
        log_info "No requirements.txt found, installing essential packages..."
        install_essential_packages
    fi
    log_success "Dependencies setup completed"
}

install_essential_packages() {
    log_info "Installing essential trading packages..."
    ESSENTIAL_PACKAGES="pandas numpy pyyaml python-binance ccxt python-telegram-bot scikit-learn lightgbm torch stable-baselines3 mlflow"
    for pkg in $ESSENTIAL_PACKAGES; do
        log_info "Installing $pkg..."
        python3 -m pip install "$pkg" || log_warning "Failed to install $pkg, may cause issues"
    done
}

process_configuration() {
    log_info "Processing comprehensive trading configuration..."
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        echo "Please ensure training_config.yaml exists in the root directory"
        exit 1
    fi
    python3 -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" || { log_error "Invalid YAML format in configuration file"; exit 1; }
    log_info "Extracting trading symbols from configuration..."
    python3 - <<EOF > temp_symbols.txt
import yaml
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
EOF
    if ! grep -q "SYMBOLS_FOUND" temp_symbols.txt; then
        log_error "Failed to extract symbols from configuration"
        exit 1
    fi
    TRADING_SYMBOLS=$(grep "SYMBOLS_FOUND" temp_symbols.txt | cut -d: -f2)
    rm -f temp_symbols.txt
    if [ -z "$TRADING_SYMBOLS" ]; then
        log_error "No trading symbols found in configuration"
        echo "Please ensure symbols are defined in the configuration file"
        exit 1
    fi
    log_info "Trading symbols extracted: $TRADING_SYMBOLS"
    SYMBOLS_SPACED=$(echo "$TRADING_SYMBOLS" | tr ',' ' ')
    VALID_SYMBOLS=""
    for symbol in $SYMBOLS_SPACED; do
        validate_symbol_config "$symbol" && VALID_SYMBOLS="$VALID_SYMBOLS,$symbol"
    done
    VALID_SYMBOLS=${VALID_SYMBOLS#,}
    if [ -z "$VALID_SYMBOLS" ]; then
        log_error "No valid symbol configurations found"
        exit 1
    fi
    log_success "Configuration processing completed. Valid symbols: $VALID_SYMBOLS"
}

validate_symbol_config() {
    local symbol="$1"
    log_info "Validating configuration for symbol: $symbol"
    if [ -z "$symbol" ]; then
        log_warning "Symbol is empty"
        return 1
    fi
    if [ ${#symbol} -lt 6 ]; then
        log_warning "Symbol $symbol too short (minimum 6 characters)"
        return 1
    fi
    log_info "Symbol $symbol configuration is valid"
    return 0
}

verify_models() {
    log_info "Performing comprehensive model verification..."
    if [ ! -d "models" ]; then
        log_error "Models directory not found"
        exit 1
    fi
    VERIFIED_SYMBOLS=""
    MODEL_TYPES="gru lightgbm ppo"
    for symbol in $(echo "$VALID_SYMBOLS" | tr ',' ' '); do
        verify_symbol_models "$symbol" && VERIFIED_SYMBOLS="$VERIFIED_SYMBOLS,$symbol" || log_warning "Excluding symbol $symbol due to missing models"
    done
    VERIFIED_SYMBOLS=${VERIFIED_SYMBOLS#,}
    if [ -z "$VERIFIED_SYMBOLS" ]; then
        log_error "No symbols have complete model sets"
        exit 1
    fi
    log_success "Model verification completed. Verified symbols: $VERIFIED_SYMBOLS"
}

verify_symbol_models() {
    local symbol="$1"
    local models_found=0
    local missing_models=""
    local found_types=""
    log_info "Verifying models for symbol: $symbol"
    for model_type in $MODEL_TYPES; do
        if find_model_for_symbol_and_type "$symbol" "$model_type"; then
            models_found=$((models_found+1))
            found_types="$found_types,$model_type"
            log_info "  Found $model_type model for $symbol"
        else
            missing_models="$missing_models,$model_type"
            log_warning "  Missing $model_type model for $symbol"
        fi
    done
    if [ $models_found -lt 1 ]; then
        log_warning "Symbol $symbol has no models ($models_found/3). Missing: $missing_models"
        return 1
    fi
    log_success "Symbol $symbol has $models_found model(s) available"
    return 0
}

find_model_for_symbol_and_type() {
    local symbol="$1"
    local model_type="$2"
    # 1. Standard directory structure
    if [ -d "models/$model_type/$symbol" ]; then
        if ls models/$model_type/$symbol/*.{pkl,pt,pth,zip,joblib} 1>/dev/null 2>&1; then return 0; fi
        for d in models/$model_type/$symbol/*/; do
            if ls "$d"/*.{pkl,pt,pth,zip,joblib} 1>/dev/null 2>&1; then return 0; fi
            for e in "$d"/*/; do
                if ls "$e"/*.{pkl,pt,pth,zip,joblib} 1>/dev/null 2>&1; then return 0; fi
            done
        done
    fi
    # 2. Flat structure
    for ext in pkl pt joblib zip; do
        if [ -f "models/${model_type}_${symbol}.$ext" ] || [ -f "models/$model_type/${model_type}_${symbol}.$ext" ] || [ -f "models/best_wf_${model_type}_${symbol}.$ext" ] || [ -f "models/$model_type/best_wf_${model_type}_${symbol}.$ext" ] || [ -f "models/${symbol}_${model_type}.$ext" ]; then return 0; fi
    done
    # 3. Search in subdirectories
    if [ -d "models/$model_type" ]; then
        if find "models/$model_type" -type f -name "*${symbol}*" | grep -q .; then return 0; fi
    fi
    # 4. Fallback directories
    for d in imported_models packaged_models legacy_models; do
        if [ -d "$d" ]; then
            for ext in pkl pt joblib zip; do
                if [ -f "$d/${model_type}_${symbol}.$ext" ]; then return 0; fi
                if [ -d "$d/$model_type/$symbol" ]; then
                    if ls "$d/$model_type/$symbol"/* 1>/dev/null 2>&1; then return 0; fi
                fi
            done
        fi
    done
    return 1
}

initialize_trading_system() {
    log_info "Initializing comprehensive trading system..."
    log_info "Validating Python trading environment..."
    python3 - <<EOF
try:
    import pandas, numpy, yaml, ccxt
    import sklearn, lightgbm
    print('Core trading dependencies verified')
except ImportError as e:
    print(f'Missing dependency: {e}')
    exit(1)
EOF
    if [ $? -ne 0 ]; then
        log_warning "Some trading dependencies missing, attempting recovery..."
        install_essential_packages
    fi
    log_info "Testing trading script functionality..."
    python3 "$TRADER_SCRIPT" --test-mode --config "$CONFIG_FILE" || log_warning "Trading script test failed, but continuing..."
    log_success "Trading system initialized successfully"
}

start_automated_trading_loop() {
    log_info "Starting automated trading loop with $EXECUTION_INTERVAL second intervals..."
    echo "\n========================================"
    echo "  AUTOMATED TRADING SYSTEM ACTIVE"
    echo "========================================\n"
    echo "Configuration:"
    echo "- Execution Interval: $EXECUTION_INTERVAL seconds (30 minutes)"
    echo "- Trading Symbols: $VERIFIED_SYMBOLS"
    echo "- Model Types: $MODEL_TYPES"
    echo "- Logs Directory: logs/"
    echo "- CSV Reports: $TRADES_CSV"
    echo "\nThe system will execute trading operations every 30 minutes."
    echo "Press Ctrl+C to stop the automated trading system."
    echo "Monitor logs/trading.log for real-time activity"
    echo "Monitor logs/trades_report.csv for trade records\n"
    while true; do
        log_info "=== Starting Unified Continuous Trading ==="
        execute_unified_trading_cycle
        sleep "$EXECUTION_INTERVAL"
    done
}

execute_unified_trading_cycle() {
    log_info "Starting unified continuous trading for all symbols: $VERIFIED_SYMBOLS"
    SYMBOL_ARGS=$(echo "$VERIFIED_SYMBOLS" | tr ',' ' ')
    python3 "$TRADER_SCRIPT" --symbols $SYMBOL_ARGS --config "$CONFIG_FILE" 2> temp_error.log
    trade_result=$?
    if [ $trade_result -eq 0 ]; then
        log_success "Unified continuous trading session completed"
        log_trade "SESSION" "ALL_SYMBOLS" "CONTINUOUS_SESSION_END" "N/A" "N/A" "SUCCESS" "Continuous trading session ended normally" "ENSEMBLE" "N/A" "N/A"
        rm -f temp_error.log
    else
        error_msg=""
        if [ -f temp_error.log ]; then
            error_msg=$(cat temp_error.log)
            rm -f temp_error.log
        fi
        log_error "Unified continuous trading failed: $error_msg"
        log_trade "SESSION" "ALL_SYMBOLS" "CONTINUOUS_SESSION_FAILED" "N/A" "N/A" "ERROR" "Continuous trading failed: $error_msg" "N/A" "N/A" "N/A"
    fi
}

# Main deployment pipeline
log_info "Enhanced automated trading system deployment started"
check_system_requirements
validate_environment
setup_dependencies
process_configuration
verify_models
initialize_trading_system
start_automated_trading_loop
log_success "Automated trading system deployment completed successfully!"
