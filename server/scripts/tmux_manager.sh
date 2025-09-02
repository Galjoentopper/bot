#!/bin/bash
# Tmux Trading Session Manager

SESSION_NAME="trading_session"
SCRIPT_DIR="/opt/trading_bot/bot"
LOG_DIR="/var/log/trading_bot"

# Load environment
source /etc/trading_bot/.env

# Check if session exists
session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# Create trading session
create_session() {
    cd "$SCRIPT_DIR"
    
    # Initialize logs first
    echo "Initializing logs..."
    ./scripts/init_logs.sh > /dev/null 2>&1

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

# Main command handling
case "$1" in
    start)
        if session_exists; then
            echo "Trading session already running"
            exit 1
        else
            create_session
            echo "Trading session started"
            sleep 2
            tmux list-sessions
        fi
        ;;
    stop)
        if session_exists; then
            tmux send-keys -t "$SESSION_NAME:trading" C-c
            sleep 3
            tmux kill-session -t "$SESSION_NAME" 2>/dev/null
            echo "Trading session stopped"
        else
            echo "No trading session running"
        fi
        ;;
    status)
        if session_exists; then
            echo "✅ Trading session is running"
            tmux list-windows -t "$SESSION_NAME"
        else
            echo "❌ No trading session running"
        fi
        ;;
    attach)
        if session_exists; then
            tmux attach-session -t "$SESSION_NAME"
        else
            echo "No trading session to attach to"
        fi
        ;;
    logs)
        if [ -f "$LOG_DIR/trading_$(date +%Y%m%d)*.log" 2>/dev/null ]; then
            tail -f "$LOG_DIR/trading_$(date +%Y%m%d)*.log" | head -20
        else
            echo "No recent log files found"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status|attach|logs}"
        echo "  start  - Start trading session"
        echo "  stop   - Stop trading session"
        echo "  status - Show session status"
        echo "  attach - Attach to running session"
        echo "  logs   - Show recent logs"
        ;;
esac