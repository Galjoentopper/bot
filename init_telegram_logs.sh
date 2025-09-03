#!/bin/bash
# Initialize log files for Telegram bot commands

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

echo "Initializing log files for Telegram commands..."

# Create logs directory
mkdir -p "$LOG_DIR"

# Initialize balance.json if it doesn't exist
BALANCE_FILE="$LOG_DIR/balance.json"
if [ ! -f "$BALANCE_FILE" ]; then
    cat > "$BALANCE_FILE" << 'EOF'
{
  "timestamp": "2025-09-03T00:00:00",
  "cash_balance": 10000.00,
  "portfolio_value": 0.00,
  "total_equity": 10000.00,
  "positions": {},
  "total_pnl": 0.00,
  "total_return": 0.00,
  "sharpe_ratio": 0.00,
  "current_drawdown": 0.00
}
EOF
    echo "✅ Created initial balance.json"
else
    echo "✅ balance.json already exists"
fi

# Initialize trades_report.csv if it doesn't exist
TRADES_FILE="$LOG_DIR/trades_report.csv"
if [ ! -f "$TRADES_FILE" ]; then
    echo "timestamp,trade_id,symbol,trade_type,quantity,price,status,notes,model_used,confidence,balance" > "$TRADES_FILE"
    echo "✅ Created trades_report.csv with header"
else
    echo "✅ trades_report.csv already exists"
fi

# Check permissions
echo "Setting correct permissions..."
chmod 644 "$BALANCE_FILE" 2>/dev/null || echo "Could not set permissions on balance.json"
chmod 644 "$TRADES_FILE" 2>/dev/null || echo "Could not set permissions on trades_report.csv"

echo ""
echo "Log file initialization complete!"
echo "Files created in: $LOG_DIR"
echo "- balance.json: $(wc -l < "$BALANCE_FILE") lines"
echo "- trades_report.csv: $(wc -l < "$TRADES_FILE") lines"
echo ""
echo "Test Telegram commands:"
echo "  /balance - Should show initial €10,000 balance"
echo "  /trades  - Should show 'No recent trades' message"
