#!/bin/bash
# Start monitoring dashboard

SESSION_NAME="monitoring"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create monitoring session
tmux new-session -d -s "$SESSION_NAME"

# System resources
tmux send-keys 'htop' C-m

# Trading logs
tmux split-window -h
tmux send-keys 'cd /opt/trading_bot && tail -f logs/trader_daemon.log' C-m

# Health monitoring
tmux split-window -v
tmux send-keys 'cd /opt/trading_bot && watch -n 30 ./health_check.sh' C-m

# Performance metrics
tmux split-window -v
tmux send-keys 'cd /opt/trading_bot && watch -n 60 "cat logs/performance_metrics.json | jq ."' C-m

echo "Monitoring dashboard started. Attach with: tmux attach-session -t $SESSION_NAME"