#!/bin/bash
# Trading System Stop Script

echo "🛑 Stopping Trading System..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Stop services using tmux manager if available
if [ -f "scripts/enhanced_tmux_manager.sh" ]; then
    ./scripts/enhanced_tmux_manager.sh stop
else
    echo "Stopping tmux sessions manually..."
    tmux kill-session -t trading-bot 2>/dev/null || true
    tmux kill-session -t telegram-bot 2>/dev/null || true
fi

# Stop systemd services
sudo systemctl stop trading-bot 2>/dev/null || true
sudo systemctl stop telegram-bot-listener 2>/dev/null || true

echo "✅ Trading system stopped!"
