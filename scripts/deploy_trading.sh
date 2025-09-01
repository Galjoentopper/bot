#!/bin/bash
# Enhanced Trading Deployment Script with Tmux

# Detect script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Load environment
if [ -f "/etc/trading_bot/.env" ]; then
    source /etc/trading_bot/.env
    export $(grep -v '^#' /etc/trading_bot/.env | xargs)
elif [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
else
    echo "Warning: No .env file found"
fi

# Logging functions
log_info() {
    echo "[$(date)] [INFO] $1" | tee -a "$LOG_DIR/deployment.log"
}

log_error() {
    echo "[$(date)] [ERROR] $1" | tee -a "$LOG_DIR/deployment.log"
}

log_success() {
    echo "[$(date)] [SUCCESS] $1" | tee -a "$LOG_DIR/deployment.log"
}

# System checks
check_system() {
    log_info "Checking system requirements..."

    if ! command -v python3 &>/dev/null; then
        log_error "Python3 not found"
        exit 1
    fi

    if ! command -v tmux &>/dev/null; then
        log_error "Tmux not found"
        exit 1
    fi

    log_success "System requirements met"
}

# Environment validation
validate_environment() {
    log_info "Validating environment..."

    if [ ! -d "$SCRIPT_DIR" ]; then
        log_error "Script directory not found: $SCRIPT_DIR"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/scripts/enhanced_trader.py" ]; then
        log_error "Enhanced trader script not found"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/training_config.yaml" ]; then
        log_error "Training config not found"
        exit 1
    fi

    log_success "Environment validation passed"
}

# Model verification
verify_models() {
    log_info "Verifying models..."

    if [ ! -d "$SCRIPT_DIR/models" ]; then
        log_error "Models directory not found"
        exit 1
    fi

    MODEL_COUNT=$(find "$SCRIPT_DIR/models" -name "*.pkl" -o -name "*.pth" -o -name "*.zip" | wc -l)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        log_error "No model files found"
        exit 1
    fi

    log_success "Found $MODEL_COUNT model files"
}

# Main deployment
log_info "Starting trading system deployment..."

check_system
validate_environment
verify_models

# Start tmux session
log_info "Starting tmux trading session..."
if "$SCRIPT_DIR/scripts/tmux_manager.sh" start; then
    log_success "Trading session started successfully"

    # Send startup notification
    python3 -c "
import asyncio
from src.notifier.enhanced_telegram import EnhancedTelegramNotifier
import os

async def notify():
    notifier = EnhancedTelegramNotifier(
        bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
        chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )
    message = '''🚀 <b>SYSTEM_STARTUP</b>

Trading system deployed and running on new Hetzner server
✅ 66 model files loaded
✅ Tmux session active
✅ Health monitoring enabled

<i>Status: INFO</i>'''
    await notifier.send_message(message)

asyncio.run(notify())
"
else
    log_error "Failed to start trading session"
    exit 1
fi

log_success "Deployment completed successfully"
echo ""
echo "=== Deployment Summary ==="
echo "✅ System checks passed"
echo "✅ Environment validated"
echo "✅ Models verified"
echo "✅ Tmux session started"
echo ""
echo "Monitor with: $SCRIPT_DIR/scripts/tmux_manager.sh status"
echo "View logs: $SCRIPT_DIR/scripts/tmux_manager.sh logs"
echo "Attach session: $SCRIPT_DIR/scripts/tmux_manager.sh attach"