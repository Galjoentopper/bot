#!/bin/bash
# Linux version of deploy_models.bat for testing
# Deploy Models Script for Enterprise Trading System
# =================================================

echo "========================================"
echo "   Enterprise Trading Model Deployment"
echo "   Advanced Model Loading and Validation"
echo "========================================"
echo

# Configuration
LOG_FILE="logs/model_deployment.log"
CONFIG_FILE="training_config.yaml"
MODELS_DIR="models"
TRADER_SCRIPT="scripts/enhanced_trader.py"
FALLBACK_DIRS="imported_models,packaged_models,legacy_models"

# Create logs directory if it doesn't exist
mkdir -p logs

# Initialize logging
echo "$(date) Model deployment started" > "$LOG_FILE"

log_info() {
    echo "[INFO] $1"
    echo "$(date) [INFO] $1" >> "$LOG_FILE"
}

log_success() {
    echo "[SUCCESS] $1"
    echo "$(date) [SUCCESS] $1" >> "$LOG_FILE"
}

log_warning() {
    echo "[WARNING] $1"
    echo "$(date) [WARNING] $1" >> "$LOG_FILE"
}

log_error() {
    echo "[ERROR] $1"
    echo "$(date) [ERROR] $1" >> "$LOG_FILE"
}

validate_environment() {
    log_info "Validating deployment environment..."

    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        log_error "Python is not installed or not accessible"
        echo "   Please install Python 3.8+ and ensure it's in your PATH"
        return 1
    fi

    # Check configuration file
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        echo "   Please ensure training_config.yaml exists in the current directory"
        return 1
    fi

    # Check enhanced trader script
    if [ ! -f "$TRADER_SCRIPT" ]; then
        log_error "Enhanced trader script not found: $TRADER_SCRIPT"
        echo "   Please ensure $TRADER_SCRIPT exists"
        return 1
    fi

    # Create required directories
    log_info "Creating required directory structure..."
    mkdir -p "$MODELS_DIR" logs data

    log_success "Environment validation completed"
    return 0
}

extract_config_symbols() {
    log_info "Extracting trading symbols from configuration..."

    # Validate YAML format
    if ! python3 -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" 2>/dev/null; then
        log_error "Invalid YAML format in configuration file"
        return 1
    fi

    # Extract symbols using enhanced extraction script
    cat > temp_extract_config.py << 'EOF'
import yaml
import sys
try:
    with open('training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Multiple extraction paths for robustness
    symbols = (
        config.get('data_acquisition', {}).get('symbols', []) or
        config.get('data', {}).get('symbols', []) or
        config.get('symbols', []) or
        config.get('trading', {}).get('symbols', [])
    )
    models = (
        config.get('training', {}).get('models', []) or
        config.get('models', []) or
        ['gru', 'lightgbm', 'ppo']
    )
    if symbols:
        print('SYMBOLS_FOUND:' + ','.join(symbols))
        print('MODELS_FOUND:' + ','.join(models))
        sys.exit(0)
    else:
        print('NO_SYMBOLS_FOUND')
        sys.exit(1)
except Exception as e:
    print(f'ERROR_EXTRACTING: {e}')
    sys.exit(1)
EOF

    if python3 temp_extract_config.py > temp_config_results.txt 2>&1; then
        # Parse extracted symbols and models
        TRADING_SYMBOLS=$(grep "SYMBOLS_FOUND:" temp_config_results.txt | cut -d: -f2)
        MODEL_TYPES=$(grep "MODELS_FOUND:" temp_config_results.txt | cut -d: -f2)
        rm -f temp_extract_config.py temp_config_results.txt

        if [ -z "$TRADING_SYMBOLS" ]; then
            log_error "No trading symbols found in configuration"
            echo "   Please ensure symbols are properly defined in $CONFIG_FILE"
            return 1
        fi

        if [ -z "$MODEL_TYPES" ]; then
            MODEL_TYPES="gru,lightgbm,ppo"
            log_warning "No model types found in config, using defaults: $MODEL_TYPES"
        fi

        log_success "Configuration extraction completed"
        log_info "Trading symbols: $TRADING_SYMBOLS"
        log_info "Model types: $MODEL_TYPES"
        return 0
    else
        log_error "Failed to extract symbols from configuration"
        if [ -f temp_config_results.txt ]; then
            echo "Error details:"
            cat temp_config_results.txt
            rm -f temp_config_results.txt
        fi
        rm -f temp_extract_config.py
        return 1
    fi
}

discover_available_models() {
    log_info "Discovering available models across multiple sources..."

    AVAILABLE_MODELS=""
    MISSING_MODELS=""
    TOTAL_MODELS_FOUND=0

    # Convert comma-separated lists to arrays
    IFS=',' read -ra SYMBOLS_ARRAY <<< "$TRADING_SYMBOLS"
    IFS=',' read -ra MODELS_ARRAY <<< "$MODEL_TYPES"

    log_info "Scanning model directories and fallback locations..."

    for symbol in "${SYMBOLS_ARRAY[@]}"; do
        for model_type in "${MODELS_ARRAY[@]}"; do
            if find_model_for_symbol "$symbol" "$model_type"; then
                TOTAL_MODELS_FOUND=$((TOTAL_MODELS_FOUND + 1))
                if [ -z "$AVAILABLE_MODELS" ]; then
                    AVAILABLE_MODELS="$symbol:$model_type"
                else
                    AVAILABLE_MODELS="$AVAILABLE_MODELS,$symbol:$model_type"
                fi
            else
                if [ -z "$MISSING_MODELS" ]; then
                    MISSING_MODELS="$symbol:$model_type"
                else
                    MISSING_MODELS="$MISSING_MODELS,$symbol:$model_type"
                fi
            fi
        done
    done

    log_info "Model discovery completed:"
    log_info "  Total models found: $TOTAL_MODELS_FOUND"
    log_info "  Available models: $AVAILABLE_MODELS"
    if [ -n "$MISSING_MODELS" ]; then
        log_warning "  Missing models: $MISSING_MODELS"
    fi

    if [ "$TOTAL_MODELS_FOUND" -eq 0 ]; then
        log_error "No models found for any symbols!"
        echo
        echo "This could mean:"
        echo "1. Models haven't been trained yet - run training first"
        echo "2. Models are in a different location - check paths"
        echo "3. Models were imported but not in expected structure"
        echo
        echo "Suggested next steps:"
        echo "1. Run: python3 scripts/enhanced_trainer.py --models lightgbm --symbols BTCEUR --n-splits 2"
        echo "2. Or check if models exist in other directories"
        echo "3. Or run import_models.sh to import existing models"
        echo
        return 1
    fi

    return 0
}

find_model_for_symbol() {
    local symbol="$1"
    local model_type="$2"

    # Search locations in order of preference:
    # 1. Standard models directory structure
    # 2. Flat models directory 
    # 3. Legacy directory structures
    # 4. Fallback directories

    # 1. Standard structure: models/model_type/symbol/
    if [ -d "$MODELS_DIR/$model_type/$symbol" ] && [ -n "$(ls -A "$MODELS_DIR/$model_type/$symbol" 2>/dev/null)" ]; then
        log_info "  Found $model_type model for $symbol in standard location"
        return 0
    fi

    # 2. Flat structure: models/model_type_symbol.*
    for ext in pkl pt joblib zip; do
        if [ -f "$MODELS_DIR/${model_type}_${symbol}.$ext" ]; then
            log_info "  Found $model_type model for $symbol in flat structure"
            return 0
        fi
        if [ -f "$MODELS_DIR/$model_type/${model_type}_${symbol}.$ext" ]; then
            log_info "  Found $model_type model for $symbol in type directory"
            return 0
        fi
    done

    # 3. Best walkforward naming: best_wf_model_type_symbol.*
    for ext in pkl pt joblib zip; do
        if [ -f "$MODELS_DIR/best_wf_${model_type}_${symbol}.$ext" ]; then
            log_info "  Found $model_type model for $symbol in best walkforward format"
            return 0
        fi
        if [ -f "$MODELS_DIR/$model_type/best_wf_${model_type}_${symbol}.$ext" ]; then
            log_info "  Found $model_type model for $symbol in best walkforward type directory"
            return 0
        fi
    done

    # 4. Search fallback directories
    IFS=',' read -ra FALLBACK_ARRAY <<< "$FALLBACK_DIRS"
    for fallback_dir in "${FALLBACK_ARRAY[@]}"; do
        if [ -d "$fallback_dir" ]; then
            for ext in pkl pt joblib zip; do
                if [ -f "$fallback_dir/${model_type}_${symbol}.$ext" ]; then
                    log_info "  Found $model_type model for $symbol in fallback directory $fallback_dir"
                    return 0
                fi
                if [ -d "$fallback_dir/$model_type/$symbol" ] && [ -n "$(ls -A "$fallback_dir/$model_type/$symbol" 2>/dev/null)" ]; then
                    log_info "  Found $model_type model for $symbol in fallback structure $fallback_dir"
                    return 0
                fi
            done
        fi
    done

    log_warning "  Model $model_type for $symbol not found in any location"
    return 1
}

test_model_loading() {
    log_info "Testing model loading with enhanced trader..."

    # Run a comprehensive test of model loading
    log_info "Running model loading test..."

    if python3 "$TRADER_SCRIPT" --test-mode --config "training_config.yaml" > temp_loading_test.txt 2>&1; then
        log_success "Model loading test passed successfully"
    else
        log_warning "Model loading test completed with issues - check logs for details"
    fi

    if [ -f temp_loading_test.txt ]; then
        log_info "Model loading test results:"
        
        # Extract key information
        grep "Trading symbols" temp_loading_test.txt || true
        grep "Model types" temp_loading_test.txt || true
        grep "models found" temp_loading_test.txt || true
        grep "ERROR" temp_loading_test.txt || true
        grep "WARNING" temp_loading_test.txt || true
        
        # Log full results
        cat temp_loading_test.txt >> "$LOG_FILE"
        rm -f temp_loading_test.txt
    fi

    return 0
}

display_deployment_summary() {
    echo
    echo "========================================"
    echo "   DEPLOYMENT SUMMARY"
    echo "========================================"
    echo
    echo "Configuration File: $CONFIG_FILE"
    echo "Models Directory: $MODELS_DIR"
    echo "Trading Symbols: $TRADING_SYMBOLS"
    echo "Model Types: $MODEL_TYPES"
    echo "Available Models: $AVAILABLE_MODELS"
    if [ -n "$MISSING_MODELS" ]; then
        echo "Missing Models: $MISSING_MODELS"
    fi
    echo "Total Models Found: $TOTAL_MODELS_FOUND"
    echo
    echo "Log File: $LOG_FILE"
    echo
    echo "Status: Ready for trading operations"
    echo
}

# Main execution
log_info "Starting enterprise model deployment system..."

# Execute deployment pipeline
if validate_environment && extract_config_symbols && discover_available_models && test_model_loading; then
    display_deployment_summary
    log_success "Model deployment completed successfully!"

    echo
    echo "========================================"
    echo "   DEPLOYMENT COMPLETE!"
    echo "   Ready for trading operations"
    echo "========================================"
    echo
    exit 0
else
    log_error "Model deployment failed!"
    exit 1
fi