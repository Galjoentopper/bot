#!/bin/bash
# Enhanced Trading System Stop Script

set -e

echo "🛑 Stopping Trading System..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Graceful shutdown function
graceful_shutdown() {
    echo "🔄 Initiating graceful shutdown..."
    
    # Stop tmux sessions
    if [ -f "scripts/enhanced_tmux_manager.sh" ]; then
        ./scripts/enhanced_tmux_manager.sh stop
    else
        echo "Enhanced tmux manager not found. Stopping sessions manually..."
        tmux kill-session -t trading_session 2>/dev/null || true
        tmux kill-session -t telegram_session 2>/dev/null || true
        tmux kill-session -t trading-bot 2>/dev/null || true
        tmux kill-session -t telegram-bot 2>/dev/null || true
    fi
    
    # Kill any remaining Python processes
    echo "🔍 Checking for remaining processes..."
    trading_pids=$(pgrep -f "enhanced_trader.py" || true)
    telegram_pids=$(pgrep -f "telegram_bot_listener" || true)
    
    if [ -n "$trading_pids" ]; then
        echo "⏹️  Stopping remaining trading processes: $trading_pids"
        kill -TERM $trading_pids 2>/dev/null || true
        sleep 3
        kill -KILL $trading_pids 2>/dev/null || true
    fi
    
    if [ -n "$telegram_pids" ]; then
        echo "⏹️  Stopping remaining telegram processes: $telegram_pids"
        kill -TERM $telegram_pids 2>/dev/null || true
        sleep 3
        kill -KILL $telegram_pids 2>/dev/null || true
    fi
    
    # Stop systemd services
    sudo systemctl stop trading-bot 2>/dev/null || true
    sudo systemctl stop telegram-bot-listener 2>/dev/null || true
    
    # Resource cleanup
    if [ -f "scripts/resource_monitor.sh" ]; then
        echo "🧹 Running resource cleanup..."
        ./scripts/resource_monitor.sh cleanup
    fi
    
    echo "✅ Graceful shutdown completed"
}

# Emergency shutdown function
emergency_shutdown() {
    echo "🚨 Emergency shutdown initiated..."
    
    # Kill all Python processes related to trading
    pkill -f "enhanced_trader.py" 2>/dev/null || true
    pkill -f "telegram_bot_listener" 2>/dev/null || true
    
    # Kill all tmux sessions
    tmux kill-server 2>/dev/null || true
    
    # Stop systemd services
    sudo systemctl stop trading-bot 2>/dev/null || true
    sudo systemctl stop telegram-bot-listener 2>/dev/null || true
    
    echo "✅ Emergency shutdown completed"
}

# Handle shutdown based on parameter
case "${1:-graceful}" in
    "graceful"|"")
        graceful_shutdown
        ;;
    "emergency"|"force")
        emergency_shutdown
        ;;
    "status")
        echo "System Status:"
        if pgrep -f "enhanced_trader.py" >/dev/null; then
            echo "🟢 Trading system is running"
        else
            echo "🔴 Trading system is stopped"
        fi
        
        if pgrep -f "telegram_bot_listener" >/dev/null; then
            echo "🟢 Telegram bot is running"
        else
            echo "🔴 Telegram bot is stopped"
        fi
        
        tmux list-sessions 2>/dev/null || echo "🔴 No tmux sessions active"
        ;;
    *)
        echo "Usage: $0 {graceful|emergency|force|status}"
        echo "  graceful  - Graceful shutdown (default)"
        echo "  emergency - Emergency shutdown (force kill)"
        echo "  force     - Same as emergency"
        echo "  status    - Show system status"
        exit 1
        ;;
esac

echo "🎯 Trading system shutdown complete!"
echo "📊 Use './start_system.sh' to restart the system"
