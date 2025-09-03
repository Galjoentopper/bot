#!/bin/bash
# Quick test script to verify log functionality

echo "=== Testing Log File Discovery ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

echo "Script directory: $SCRIPT_DIR"
echo "Log directory: $LOG_DIR"
echo ""

echo "=== Checking log directory ==="
if [ -d "$LOG_DIR" ]; then
    echo "✅ Log directory exists"
    echo "Permissions: $(ls -ld "$LOG_DIR")"
    echo ""
    
    echo "=== All log files ==="
    ls -la "$LOG_DIR"/ 2>/dev/null
    echo ""
    
    echo "=== Trading log files ==="
    ls -la "$LOG_DIR"/trading_*.log 2>/dev/null || echo "No trading logs found"
    echo ""
    
    echo "=== Telegram log files ==="
    ls -la "$LOG_DIR"/telegram_*.log 2>/dev/null || echo "No telegram logs found"
    echo ""
    
    echo "=== Most recent trading log ==="
    RECENT_TRADING_LOG=$(ls -t "$LOG_DIR"/trading_*.log 2>/dev/null | head -1)
    if [ -n "$RECENT_TRADING_LOG" ]; then
        echo "Found: $RECENT_TRADING_LOG"
        echo "Size: $(wc -l < "$RECENT_TRADING_LOG") lines"
        echo "Last modified: $(ls -l "$RECENT_TRADING_LOG")"
    else
        echo "No trading logs found"
    fi
    echo ""
    
    echo "=== Most recent telegram log ==="
    RECENT_TELEGRAM_LOG=$(ls -t "$LOG_DIR"/telegram_*.log 2>/dev/null | head -1)
    if [ -n "$RECENT_TELEGRAM_LOG" ]; then
        echo "Found: $RECENT_TELEGRAM_LOG"
        echo "Size: $(wc -l < "$RECENT_TELEGRAM_LOG") lines"
        echo "Last modified: $(ls -l "$RECENT_TELEGRAM_LOG")"
    else
        echo "No telegram logs found"
    fi
else
    echo "❌ Log directory does not exist: $LOG_DIR"
    echo "Creating log directory..."
    mkdir -p "$LOG_DIR"
    echo "Created: $LOG_DIR"
fi

echo ""
echo "=== Testing tmux logs command ==="
if [ -f "./scripts/enhanced_tmux_manager.sh" ]; then
    echo "Running: ./scripts/enhanced_tmux_manager.sh logs"
    ./scripts/enhanced_tmux_manager.sh logs
else
    echo "❌ Enhanced tmux manager not found"
fi
