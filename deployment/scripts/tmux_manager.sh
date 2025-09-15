#!/bin/bash
# Tmux Trading Session Manager
set -euo pipefail

SESSION_NAME="trading_session"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Load environment
if [ -f "/etc/trading_bot/.env" ]; then
    source /etc/trading_bot/.env
elif [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "Warning: No .env file found"
fi

# Set default timeout if not set
# Set default timeout if not set
if [ -z "${TRADING_TIMEOUT:-}" ]; then
    TRADING_TIMEOUT="24h"
    echo "Setting default TRADING_TIMEOUT to $TRADING_TIMEOUT"
fi

# Use a dedicated tmux socket directory to avoid systemd PrivateTmp isolation
TMUX_SOCKET_DIR="${TMUX_TMPDIR:-$SCRIPT_DIR/tmux}"
mkdir -p "$TMUX_SOCKET_DIR"
chmod 700 "$TMUX_SOCKET_DIR"
TMUX_SOCKET="$TMUX_SOCKET_DIR/trading-bot.sock"

tmux_cmd() {
    TMUX_TMPDIR="$TMUX_SOCKET_DIR" tmux -S "$TMUX_SOCKET" "$@"
}

# Check if session exists
session_exists() {
    tmux_cmd has-session -t "$SESSION_NAME" 2>/dev/null
}

# Create trading session
create_session() {
    cd "$SCRIPT_DIR"

    # Create main session
    tmux_cmd new-session -d -s "$SESSION_NAME" -n "trading"

    # Main trading pane with timeout
    tmux_cmd send-keys -t "$SESSION_NAME:trading" "cd $SCRIPT_DIR && [ -f venv/bin/activate ] && source venv/bin/activate; timeout $TRADING_TIMEOUT python3 scripts/enhanced_trader.py --config training_config.yaml --symbols BTCEUR,ETHEUR,ADAEUR,DOTEUR,LINKEUR 2>&1 | tee -a $LOG_DIR/trading_$(date +%Y%m%d_%H%M%S).log" C-m

    # Logs monitoring pane
    tmux_cmd split-window -h -t "$SESSION_NAME:trading"
    tmux_cmd send-keys -t "$SESSION_NAME:trading.right" "cd $LOG_DIR && tail -f trading_*.log" C-m

    # System monitoring pane
    tmux_cmd split-window -v -t "$SESSION_NAME:trading.right"
    tmux_cmd send-keys -t "$SESSION_NAME:trading.bottom" "htop" C-m || true

    # Health check pane
    tmux_cmd split-window -v -t "$SESSION_NAME:trading"
    tmux_cmd send-keys -t "$SESSION_NAME:trading.top" "cd $SCRIPT_DIR && watch -n 30 './health_check.sh'" C-m
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
            tmux_cmd list-sessions || true
        fi
        ;;
    stop)
        if session_exists; then
            tmux_cmd send-keys -t "$SESSION_NAME:trading" C-c
            sleep 3
            tmux_cmd kill-session -t "$SESSION_NAME" 2>/dev/null
            echo "Trading session stopped"
        else
            echo "No trading session running"
        fi
        ;;
    status)
        if session_exists; then
            echo "✅ Trading session is running"
            tmux_cmd list-windows -t "$SESSION_NAME"
        else
            echo "❌ No trading session running"
        fi
        ;;
    attach)
        if session_exists; then
            echo "Using socket: $TMUX_SOCKET"
            TMUX="$TMUX_SOCKET" TMUX_TMPDIR="$TMUX_SOCKET_DIR" tmux -S "$TMUX_SOCKET" attach-session -t "$SESSION_NAME"
        else
            echo "No trading session to attach to"
        fi
        ;;
    logs)
        # Show the most recent trading log for today, if any
        latest_log=$(ls -t "$LOG_DIR"/trading_"$(date +%Y%m%d)"*.log 2>/dev/null | head -1 || true)
        if [ -n "${latest_log:-}" ] && [ -f "$latest_log" ]; then
            echo "Showing last 200 lines of: $latest_log"
            tail -n 200 "$latest_log"
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
