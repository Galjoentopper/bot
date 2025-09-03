#!/bin/bash
# Health Check Script

LOG_DIR="$SCRIPT_DIR/logs"
SCRIPT_DIR="/opt/trading_bot/bot"

echo "=== Trading Bot Health Check ==="
echo "Time: $(date)"
echo "Server: $(hostname -I | awk '{print $1}')"
echo ""

# Check tmux session
if tmux has-session -t trading_session 2>/dev/null; then
    echo "✅ Tmux Session: Running"
    tmux list-windows -t trading_session 2>/dev/null
else
    echo "❌ Tmux Session: Not running"
fi

# Check Python process
if pgrep -f "enhanced_trader.py" > /dev/null; then
    echo "✅ Trading Process: Running"
    echo "   PID: $(pgrep -f "enhanced_trader.py")"
    echo "   CPU: $(ps -p $(pgrep -f "enhanced_trader.py") -o pcpu= 2>/dev/null)%"
    echo "   Memory: $(ps -p $(pgrep -f "enhanced_trader.py") -o pmem= 2>/dev/null)%"
else
    echo "❌ Trading Process: Not running"
fi

# Check log files
LOG_COUNT=$(find "$LOG_DIR" -name "trading_*.log" -type f | wc -l)
if [ "$LOG_COUNT" -gt 0 ]; then
    echo "✅ Log Files: $LOG_COUNT files present"
    LATEST_LOG=$(find "$LOG_DIR" -name "trading_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_LOG" ]; then
        LOG_SIZE=$(stat -c%s "$LATEST_LOG" 2>/dev/null || echo "0")
        echo "   Latest: $(basename "$LATEST_LOG") (${LOG_SIZE} bytes)"
    fi
else
    echo "❌ Log Files: None found"
fi

# System resources
echo ""
echo "=== System Resources ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2 " (" $4 " free)"}')"
echo "Disk: $(df -h /opt/trading_bot | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"

# Network connectivity
echo ""
echo "=== Network Status ==="
if ping -c 1 8.8.8.8 &>/dev/null; then
    echo "✅ Internet: Connected"
else
    echo "❌ Internet: Disconnected"
fi

# Recent trading activity
echo ""
echo "=== Recent Activity ==="
if [ -f "$SCRIPT_DIR/logs/trades_report.csv" ]; then
    RECENT_TRADES=$(tail -5 "$SCRIPT_DIR/logs/trades_report.csv" | wc -l)
    echo "Recent trades in log: $RECENT_TRADES"
    if [ -f "$SCRIPT_DIR/logs/trades_report.csv" ]; then
        LAST_TRADE=$(tail -1 "$SCRIPT_DIR/logs/trades_report.csv" | cut -d',' -f2,3,4)
        if [ -n "$LAST_TRADE" ]; then
            echo "Last trade: $LAST_TRADE"
        fi
    fi
else
    echo "No trades report found"
fi

echo "=================================="