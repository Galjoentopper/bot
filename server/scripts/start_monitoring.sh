#!/bin/bash

# Start Monitoring Dashboard Script
# Creates a tmux session with multiple panes for monitoring

SCRIPT_DIR="/opt/trading_bot/bot"
cd "$SCRIPT_DIR" || exit 1

# Kill existing monitoring session if it exists
tmux kill-session -t monitoring 2>/dev/null

# Create new monitoring session
tmux new-session -d -s monitoring

# Split into 4 panes
tmux split-window -h -t monitoring:0
tmux split-window -v -t monitoring:0.1
tmux split-window -v -t monitoring:0.0

# Pane 0 (top-left): Health check every 30 seconds
tmux send-keys -t monitoring:0.0 'watch -n 30 "./scripts/health_check.sh"' Enter

# Pane 1 (top-right): Performance metrics every 60 seconds
tmux send-keys -t monitoring:0.1 'watch -n 60 "cat logs/performance_metrics.json 2>/dev/null || echo \"Generating metrics...\" && ./scripts/generate_performance_report.sh > /dev/null 2>&1 && cat logs/performance_metrics.json 2>/dev/null || echo \"No metrics available\""' Enter

# Pane 2 (bottom-right): Trader daemon log
tmux send-keys -t monitoring:0.2 'tail -f logs/trader_daemon.log 2>/dev/null || (echo "Creating daemon log..." && mkdir -p logs && touch logs/trader_daemon.log && tail -f logs/trader_daemon.log)' Enter

# Pane 3 (bottom-left): System resources
tmux send-keys -t monitoring:0.3 'htop' Enter

# Set window name
tmux rename-window -t monitoring:0 'Monitor'

echo "Monitoring dashboard started. Use 'tmux attach -t monitoring' to view."
echo "Panes:"
echo "  Top-left: Health checks (30s refresh)"
echo "  Top-right: Performance metrics (60s refresh)"
echo "  Bottom-left: System resources (htop)"
echo "  Bottom-right: Trader daemon log"