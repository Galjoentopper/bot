#!/bin/bash

# Enhanced Trading Bot Deployment Script
# Deploys and starts the trading bot with proper error handling and logging

set -e  # Exit on any error

# Self-locate the bot directory (server/scripts -> bot)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
SERVICE_USER="trader"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=== Enhanced Trading Bot Deployment ==="
echo "Time: $(date)"
echo "User: $(whoami)"
echo "Script directory: $SCRIPT_DIR"

# Create necessary directories
echo "Creating directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$SCRIPT_DIR/backups/logs"
mkdir -p "$SCRIPT_DIR/backups/config"

# Change to script directory
cd "$SCRIPT_DIR" || {
    echo "❌ Error: Cannot access script directory $SCRIPT_DIR"
    exit 1
}

# Check if configuration exists
if [ ! -f "training_config.yaml" ] && [ ! -f "config.yaml" ]; then
    echo "⚠️  Warning: No configuration file found (training_config.yaml or config.yaml)"
    echo "   The trader will use default configuration"
fi

# Check Python environment
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Warning: No virtual environment found"
fi

# Install/update requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing/updating requirements..."
    pip install -q -r requirements.txt || echo "⚠️  Warning: Failed to install some requirements"
fi

# Check if trading scripts exist
TRADER_SCRIPT=""
if [ -f "scripts/enhanced_trader.py" ]; then
    TRADER_SCRIPT="scripts/enhanced_trader.py"
    echo "✅ Found enhanced trader script"
elif [ -f "scripts/trader.py" ]; then
    TRADER_SCRIPT="scripts/trader.py"
    echo "✅ Found trader script"
else
    echo "❌ Error: No trader script found (enhanced_trader.py or trader.py)"
    exit 1
fi

# Stop existing trading session
echo "Stopping existing trading sessions..."
./scripts/tmux_manager.sh stop || echo "No existing session to stop"

# Generate initial performance report
echo "Generating initial performance report..."
./scripts/generate_performance_report.sh

# Start new trading session
echo "Starting new trading session..."
if [ -f "training_config.yaml" ]; then
    CONFIG_ARG="--config training_config.yaml"
elif [ -f "config.yaml" ]; then
    CONFIG_ARG="--config config.yaml"
else
    CONFIG_ARG=""
fi

# Start the trading bot in tmux
./scripts/tmux_manager.sh start "$TRADER_SCRIPT" "$CONFIG_ARG"

# Wait a moment for the session to initialize
sleep 3

# Check if the trading session started successfully
if ./scripts/tmux_manager.sh status > /dev/null; then
    echo "✅ Trading session started successfully"
    
    # Show session status
    echo ""
    echo "=== Trading Session Status ==="
    ./scripts/tmux_manager.sh status
    
    echo ""
    echo "=== Deployment Commands ==="
    echo "View trading session: ./scripts/tmux_manager.sh attach"
    echo "Check status: ./scripts/tmux_manager.sh status"
    echo "View logs: ./scripts/tmux_manager.sh logs"
    echo "Stop trading: ./scripts/tmux_manager.sh stop"
    echo "Start monitoring: ./scripts/start_monitoring.sh"
    
else
    echo "❌ Error: Trading session failed to start"
    echo "Check logs with: ./scripts/tmux_manager.sh logs"
    exit 1
fi

# Log deployment
DEPLOY_LOG="$LOG_DIR/deployment.log"
echo "$(date): Deployment completed successfully - Script: $TRADER_SCRIPT, Config: $CONFIG_ARG" >> "$DEPLOY_LOG"

echo ""
echo "✅ Deployment completed successfully!"
echo "=== Next Steps ==="
echo "1. Monitor with: tmux attach -t trading_session"
echo "2. Check health: ./scripts/health_check.sh"
echo "3. Start monitoring dashboard: ./scripts/start_monitoring.sh"
echo "=================================="