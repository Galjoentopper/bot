#!/bin/bash
# Enhanced Tmux Trading Session Manager with Telegram Bot Support

SESSION_NAME="trading_session"
TELEGRAM_SESSION_NAME="telegram_session"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Default timeout if not set
TRADING_TIMEOUT=${TRADING_TIMEOUT:-300}  # 5 minutes default

# Load environment
if [ -f "/etc/trading_bot/.env" ]; then
    source /etc/trading_bot/.env
elif [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "Warning: No .env file found"
fi

# Check if session exists
session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

telegram_session_exists() {
    tmux has-session -t "$TELEGRAM_SESSION_NAME" 2>/dev/null
}

# Create trading session
create_trading_session() {
    cd "$SCRIPT_DIR"

    # Create main session
    tmux new-session -d -s "$SESSION_NAME" -n "trading"

    # Main trading pane with timeout
    tmux send-keys -t "$SESSION_NAME:trading" "cd $SCRIPT_DIR && timeout $TRADING_TIMEOUT python3 scripts/enhanced_trader.py --config training_config.yaml --symbols BTCEUR,ETHEUR,ADAEUR,DOTEUR,LINKEUR 2>&1 | tee -a $LOG_DIR/trading_$(date +%Y%m%d_%H%M%S).log" C-m

    # Logs monitoring pane
    tmux split-window -h -t "$SESSION_NAME:trading"
    tmux send-keys -t "$SESSION_NAME:trading.right" "cd $LOG_DIR && tail -f trading_*.log" C-m

    # System monitoring pane
    tmux split-window -v -t "$SESSION_NAME:trading.right"
    tmux send-keys -t "$SESSION_NAME:trading.bottom" "htop" C-m

    # Health check pane
    tmux split-window -v -t "$SESSION_NAME:trading"
    tmux send-keys -t "$SESSION_NAME:trading.top" "cd $SCRIPT_DIR && watch -n 30 './scripts/health_check.sh'" C-m
}

# Create Telegram bot session
create_telegram_session() {
    cd "$SCRIPT_DIR"

    # Create Telegram session
    tmux new-session -d -s "$TELEGRAM_SESSION_NAME" -n "telegram"

    # Start telegram bot in telegram session
    tmux send-keys -t "$TELEGRAM_SESSION_NAME:telegram" "cd $SCRIPT_DIR && python3 telegram_bot_listener_fixed.py 2>&1 | tee -a $LOG_DIR/telegram_$(date +%Y%m%d_%H%M%S).log" C-m

    # Telegram logs monitoring
    tmux split-window -h -t "$TELEGRAM_SESSION_NAME:telegram"
    tmux send-keys -t "$TELEGRAM_SESSION_NAME:telegram.right" "cd $LOG_DIR && tail -f telegram_*.log" C-m
}

# Main command handling
case "$1" in
    start)
        echo "Starting trading system..."

        # Start trading session
        if session_exists; then
            echo "Trading session already running"
        else
            create_trading_session
            echo "✅ Trading session started"
        fi

        # Start Telegram session
        if telegram_session_exists; then
            echo "Telegram session already running"
        else
            create_telegram_session
            echo "✅ Telegram bot listener started"
        fi

        sleep 2
        echo ""
        echo "Active sessions:"
        tmux list-sessions 2>/dev/null || echo "No tmux sessions found"
        ;;
    stop)
        echo "Stopping trading system..."

        # Stop trading session
        if session_exists; then
            tmux send-keys -t "$SESSION_NAME:trading" C-c
            sleep 3
            tmux kill-session -t "$SESSION_NAME" 2>/dev/null
            echo "✅ Trading session stopped"
        else
            echo "No trading session running"
        fi

        # Stop Telegram session
        if telegram_session_exists; then
            tmux send-keys -t "$TELEGRAM_SESSION_NAME:telegram" C-c
            sleep 2
            tmux kill-session -t "$TELEGRAM_SESSION_NAME" 2>/dev/null
            echo "✅ Telegram session stopped"
        else
            echo "No Telegram session running"
        fi
        ;;
    status)
        echo "=== System Status ==="

        if session_exists; then
            echo "✅ Trading session is running"
            tmux list-windows -t "$SESSION_NAME"
        else
            echo "❌ Trading session not running"
        fi

        if telegram_session_exists; then
            echo "✅ Telegram bot listener is running"
            tmux list-windows -t "$TELEGRAM_SESSION_NAME"
        else
            echo "❌ Telegram bot listener not running"
        fi

        echo ""
        echo "All sessions:"
        tmux list-sessions 2>/dev/null || echo "No tmux sessions found"
        ;;
    attach)
        if session_exists; then
            echo "Attaching to trading session..."
            tmux attach-session -t "$SESSION_NAME"
        elif telegram_session_exists; then
            echo "Attaching to Telegram session..."
            tmux attach-session -t "$TELEGRAM_SESSION_NAME"
        else
            echo "No sessions to attach to"
        fi
        ;;
    logs)
        echo "=== Recent Logs ==="
        echo "Trading logs:"
        # Find the most recent trading log file
        RECENT_TRADING_LOG=$(ls -t "$LOG_DIR"/trading_*.log 2>/dev/null | head -1)
        if [ -n "$RECENT_TRADING_LOG" ]; then
            echo "Showing last 20 lines from: $(basename $RECENT_TRADING_LOG)"
            tail -20 "$RECENT_TRADING_LOG"
        else
            echo "No trading log files found in $LOG_DIR"
            echo "Available log files:"
            ls -la "$LOG_DIR"/ 2>/dev/null || echo "Log directory not found"
        fi

        echo ""
        echo "Telegram logs:"
        # Find the most recent telegram log file
        RECENT_TELEGRAM_LOG=$(ls -t "$LOG_DIR"/telegram_*.log 2>/dev/null | head -1)
        if [ -n "$RECENT_TELEGRAM_LOG" ]; then
            echo "Showing last 20 lines from: $(basename $RECENT_TELEGRAM_LOG)"
            tail -20 "$RECENT_TELEGRAM_LOG"
        else
            echo "No Telegram log files found in $LOG_DIR"
            echo "Available log files:"
            ls -la "$LOG_DIR"/ 2>/dev/null || echo "Log directory not found"
        fi
        ;;
    restart)
        echo "Restarting trading system..."
        $0 stop
        sleep 3
        $0 start
        ;;
    *)
        echo "Enhanced Tmux Trading Session Manager"
        echo "Usage: $0 {start|stop|status|attach|logs|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Start both trading session and Telegram bot listener"
        echo "  stop    - Stop both trading session and Telegram bot listener"
        echo "  status  - Show status of all sessions"
        echo "  attach  - Attach to running session (trading first, then Telegram)"
        echo "  logs    - Show recent logs from both services"
        echo "  restart - Restart all services"
        echo ""
        echo "Log files are stored in: $LOG_DIR/"
        echo "Manual log commands:"
        echo "  tail -f $LOG_DIR/trading_*.log    # Follow trading logs"
        echo "  tail -f $LOG_DIR/telegram_*.log   # Follow telegram logs"
        echo "  ls -la $LOG_DIR/                  # List all log files"
        echo ""
        echo "Environment variables:"
        echo "  TRADING_TIMEOUT - Trading session timeout in seconds (default: 300)"
        ;;
esac