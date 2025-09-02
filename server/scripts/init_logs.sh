#!/bin/bash

# Log Initialization Script
# Creates necessary log files and directories for the trading bot

SCRIPT_DIR="/opt/trading_bot/bot"
LOG_DIR="$SCRIPT_DIR/logs"

echo "=== Initializing Trading Bot Logs ==="
echo "Time: $(date)"
echo "Log directory: $LOG_DIR"

# Create logs directory
mkdir -p "$LOG_DIR"

# Initialize trader daemon log
DAEMON_LOG="$LOG_DIR/trader_daemon.log"
if [ ! -f "$DAEMON_LOG" ]; then
    echo "$(date): Trading daemon log initialized" > "$DAEMON_LOG"
    echo "✅ Created trader_daemon.log"
else
    echo "✅ trader_daemon.log already exists"
fi

# Initialize performance metrics JSON
METRICS_FILE="$LOG_DIR/performance_metrics.json"
if [ ! -f "$METRICS_FILE" ]; then
    cat > "$METRICS_FILE" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "initialized",
    "portfolio_value": 0,
    "daily_pnl": 0,
    "total_return": 0,
    "sharpe_ratio": 0,
    "win_rate": 0,
    "active_positions": 0,
    "system": {
        "uptime": "$(uptime -p)",
        "load_average": "$(uptime | awk '{print $(NF-2)}' | tr -d ',')",
        "memory_total": "$(free -h | grep Mem | awk '{print $2}')",
        "memory_used": "$(free -h | grep Mem | awk '{print $3}')"
    }
}
EOF
    echo "✅ Created performance_metrics.json"
else
    echo "✅ performance_metrics.json already exists"
fi

# Initialize balance JSON
BALANCE_FILE="$LOG_DIR/balance.json"
if [ ! -f "$BALANCE_FILE" ]; then
    cat > "$BALANCE_FILE" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "cash_balance": 10000.0,
    "portfolio_value": 10000.0,
    "total_equity": 10000.0,
    "positions": {}
}
EOF
    echo "✅ Created balance.json"
else
    echo "✅ balance.json already exists"
fi

# Initialize trades report CSV
TRADES_FILE="$LOG_DIR/trades_report.csv"
if [ ! -f "$TRADES_FILE" ]; then
    echo "timestamp,trade_id,symbol,side,quantity,price,status,notes,model_used,confidence,balance" > "$TRADES_FILE"
    echo "✅ Created trades_report.csv"
else
    echo "✅ trades_report.csv already exists"
fi

# Create deployment log
DEPLOY_LOG="$LOG_DIR/deployment.log"
if [ ! -f "$DEPLOY_LOG" ]; then
    echo "$(date): Log initialization completed" > "$DEPLOY_LOG"
    echo "✅ Created deployment.log"
else
    echo "✅ deployment.log already exists"
fi

# Set proper permissions
chmod 644 "$LOG_DIR"/*.log "$LOG_DIR"/*.json "$LOG_DIR"/*.csv 2>/dev/null

echo ""
echo "=== Log Files Summary ==="
ls -la "$LOG_DIR"/

echo ""
echo "✅ Log initialization completed successfully!"
echo "=================================="