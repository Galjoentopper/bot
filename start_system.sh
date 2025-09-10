#!/bin/bash
# Enhanced Trading System Startup Script with Dependency Checks

set -e  # Exit on any error

echo "🚀 Starting Trading System..."
echo "🔍 Running pre-flight checks..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to check dependencies
check_dependencies() {
    local failed=0

    echo "📋 Checking system dependencies..."

    # Check if virtual environment exists
    if [ ! -f "venv/bin/activate" ]; then
        echo "❌ Virtual environment not found at venv/bin/activate"
        echo "💡 Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        failed=1
    else
        echo "✅ Virtual environment found"
    fi

    # Check if models directory exists
    if [ ! -d "models" ]; then
        echo "❌ Models directory not found"
        echo "💡 Import models using: ./import_models.sh"
        failed=1
    else
        # Check for essential model files
        model_count=$(find models -name "*.pkl" -o -name "*.pt" -o -name "*.zip" | wc -l)
        if [ $model_count -lt 5 ]; then
            echo "⚠️  Only $model_count model files found (expected 15+)"
            echo "💡 Consider importing fresh models"
        else
            echo "✅ Models directory found with $model_count files"
        fi
    fi

    # Check if configuration file exists
    if [ ! -f "training_config.yaml" ]; then
        echo "❌ Configuration file training_config.yaml not found"
        failed=1
    else
        echo "✅ Configuration file found"
    fi

    # Check if enhanced tmux manager exists
    if [ ! -f "scripts/enhanced_tmux_manager.sh" ]; then
        echo "❌ Enhanced tmux manager not found"
        failed=1
    else
        echo "✅ Enhanced tmux manager found"
    fi

    # Check available memory
    available_mem=$(free -m | awk 'NR==2{printf "%d", $7}')
    if [ $available_mem -lt 1000 ]; then
        echo "⚠️  Low memory available: ${available_mem}MB (recommend 1GB+)"
        echo "💡 Consider stopping other processes or restarting server"
    else
        echo "✅ Sufficient memory available: ${available_mem}MB"
    fi

    # Check disk space
    disk_free=$(df . | awk 'NR==2{print $4}')
    disk_free_gb=$((disk_free / 1024 / 1024))
    if [ $disk_free_gb -lt 5 ]; then
        echo "⚠️  Low disk space: ${disk_free_gb}GB free"
        echo "💡 Consider cleaning up logs or temporary files"
    else
        echo "✅ Sufficient disk space: ${disk_free_gb}GB free"
    fi

    if [ $failed -eq 1 ]; then
        echo "❌ Dependency check failed. Please fix the issues above."
        exit 1
    fi

    echo "✅ All dependency checks passed!"
}

# Timeout handler
timeout_handler() {
    echo "⏰ Startup timeout reached. Stopping processes..."
    ./scripts/enhanced_tmux_manager.sh stop 2>/dev/null || true
    exit 1
}

# Set up timeout (5 minutes)
trap timeout_handler TERM
(sleep 300; kill -TERM $$) &
TIMEOUT_PID=$!

# Run dependency checks
check_dependencies

# Set up resource management
echo "⚙️  Configuring resource management..."
if [ -f "scripts/resource_monitor.sh" ]; then
    ./scripts/resource_monitor.sh limits
    ./scripts/resource_monitor.sh cleanup
    ./scripts/resource_monitor.sh check
else
    echo "⚠️  Resource monitor not found, continuing without resource management"
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
if ! source venv/bin/activate; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
echo "✅ Virtual environment activated"

# Start both services using tmux
echo "🚀 Starting trading services..."
if [ -f "scripts/enhanced_tmux_manager.sh" ]; then
    if ! ./scripts/enhanced_tmux_manager.sh start; then
        echo "❌ Failed to start trading system"
        kill $TIMEOUT_PID 2>/dev/null || true
        exit 1
    fi
else
    echo "❌ Enhanced tmux manager not found. Starting services manually..."
    # Fallback to direct execution
    if ! tmux new-session -d -s trading-bot "python3 scripts/enhanced_trader.py"; then
        echo "❌ Failed to start trading bot"
        kill $TIMEOUT_PID 2>/dev/null || true
        exit 1
    fi
    if ! tmux new-session -d -s telegram-bot "python3 telegram_bot_listener_systemd.py"; then
        echo "❌ Failed to start telegram bot"
        kill $TIMEOUT_PID 2>/dev/null || true
        exit 1
    fi
fi

# Clean up timeout
kill $TIMEOUT_PID 2>/dev/null || true

echo ""
echo "🎉 Trading system started successfully!"
echo "⏱️  Startup completed in $(date)"
echo ""
echo "📊 Available Telegram commands:"
echo "  /status     - System status"
echo "  /start      - Start trading"
echo "  /stop       - Stop trading"
echo "  /restart    - Restart system"
echo "  /performance- Performance metrics"
echo "  /health     - Health check"
echo "  /balance    - Account balance"
echo "  /trades     - Recent trades"
echo "  /logs       - View logs"
echo "  /config     - Configuration"
echo "  /help       - Command help"
echo "  /database   - Rebuild databases"
echo ""
echo "📊 System management commands:"
echo "  ./scripts/enhanced_tmux_manager.sh status    # Check status"
echo "  ./scripts/enhanced_tmux_manager.sh logs      # View logs"
echo "  ./scripts/enhanced_tmux_manager.sh stop      # Stop system"
echo ""
echo "🐧 Systemd commands:"
echo "  sudo systemctl start trading-bot             # Start trading service"
echo "  sudo systemctl start telegram-bot-listener   # Start telegram service"
echo "  sudo systemctl status trading-bot            # Check trading status"
echo "  sudo systemctl status telegram-bot-listener  # Check telegram status"
