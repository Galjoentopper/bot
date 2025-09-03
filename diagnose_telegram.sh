#!/bin/bash
# Comprehensive diagnostic script for Telegram bot issues

echo "🔍 TELEGRAM BOT DIAGNOSTIC SCRIPT"
echo "================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

echo "📁 Project directory: $SCRIPT_DIR"
echo "📁 Log directory: $LOG_DIR"
echo ""

# Check log directory
echo "🗂️  LOG DIRECTORY STATUS"
echo "----------------------"
if [ -d "$LOG_DIR" ]; then
    echo "✅ Log directory exists"
    echo "   Permissions: $(ls -ld "$LOG_DIR" | awk '{print $1, $3, $4}')"
    echo "   Files in logs/:"
    ls -la "$LOG_DIR"/ 2>/dev/null || echo "   Cannot list files"
else
    echo "❌ Log directory missing: $LOG_DIR"
    echo "   Creating log directory..."
    mkdir -p "$LOG_DIR" && echo "   ✅ Created" || echo "   ❌ Failed to create"
fi
echo ""

# Check balance.json
echo "💰 BALANCE.JSON STATUS"
echo "---------------------"
BALANCE_FILE="$LOG_DIR/balance.json"
if [ -f "$BALANCE_FILE" ]; then
    echo "✅ balance.json exists"
    echo "   Path: $BALANCE_FILE"
    echo "   Size: $(du -h "$BALANCE_FILE" | cut -f1)"
    echo "   Last modified: $(ls -l "$BALANCE_FILE" | awk '{print $6, $7, $8}')"
    echo "   Content preview:"
    head -5 "$BALANCE_FILE" | sed 's/^/     /'
else
    echo "❌ balance.json missing: $BALANCE_FILE"
    echo "   This is needed for /balance command"
fi
echo ""

# Check trades_report.csv
echo "📈 TRADES_REPORT.CSV STATUS"
echo "--------------------------"
TRADES_FILE="$LOG_DIR/trades_report.csv"
if [ -f "$TRADES_FILE" ]; then
    echo "✅ trades_report.csv exists"
    echo "   Path: $TRADES_FILE"
    echo "   Size: $(du -h "$TRADES_FILE" | cut -f1)"
    echo "   Lines: $(wc -l < "$TRADES_FILE")"
    echo "   Last modified: $(ls -l "$TRADES_FILE" | awk '{print $6, $7, $8}')"
    if [ $(wc -l < "$TRADES_FILE") -gt 1 ]; then
        echo "   Recent trades:"
        tail -3 "$TRADES_FILE" | sed 's/^/     /'
    else
        echo "   No trades recorded yet (header only)"
    fi
else
    echo "❌ trades_report.csv missing: $TRADES_FILE"
    echo "   This is needed for /trades command"
fi
echo ""

# Check performance_metrics.json
echo "📊 PERFORMANCE_METRICS.JSON STATUS"
echo "----------------------------------"
PERFORMANCE_FILE="$LOG_DIR/performance_metrics.json"
if [ -f "$PERFORMANCE_FILE" ]; then
    echo "✅ performance_metrics.json exists"
    echo "   Path: $PERFORMANCE_FILE"
    echo "   Size: $(du -h "$PERFORMANCE_FILE" | cut -f1)"
    echo "   Last modified: $(ls -l "$PERFORMANCE_FILE" | awk '{print $6, $7, $8}')"
    echo "   Content preview:"
    head -5 "$PERFORMANCE_FILE" | sed 's/^/     /'
else
    echo "❌ performance_metrics.json missing: $PERFORMANCE_FILE"
    echo "   This is needed for /performance command"
fi
echo ""

# Check trading logs
echo "📊 TRADING LOGS STATUS"
echo "---------------------"
TRADING_LOGS=$(ls -t "$LOG_DIR"/trading_*.log 2>/dev/null | head -3)
if [ -n "$TRADING_LOGS" ]; then
    echo "✅ Trading logs found:"
    for log in $TRADING_LOGS; do
        echo "   - $(basename "$log") ($(du -h "$log" | cut -f1), $(wc -l < "$log") lines)"
    done
    echo "   Most recent log content (last 3 lines):"
    LATEST_LOG=$(ls -t "$LOG_DIR"/trading_*.log 2>/dev/null | head -1)
    tail -3 "$LATEST_LOG" 2>/dev/null | sed 's/^/     /' || echo "     Could not read log"
else
    echo "❌ No trading logs found in $LOG_DIR"
    echo "   Pattern searched: trading_*.log"
fi
echo ""

# Check telegram logs
echo "💬 TELEGRAM LOGS STATUS"
echo "----------------------"
TELEGRAM_LOGS=$(ls -t "$LOG_DIR"/telegram_*.log 2>/dev/null | head -3)
if [ -n "$TELEGRAM_LOGS" ]; then
    echo "✅ Telegram logs found:"
    for log in $TELEGRAM_LOGS; do
        echo "   - $(basename "$log") ($(du -h "$log" | cut -f1), $(wc -l < "$log") lines)"
    done
    echo "   Most recent log content (last 3 lines):"
    LATEST_LOG=$(ls -t "$LOG_DIR"/telegram_*.log 2>/dev/null | head -1)
    tail -3 "$LATEST_LOG" 2>/dev/null | sed 's/^/     /' || echo "     Could not read log"
else
    echo "❌ No telegram logs found in $LOG_DIR"
    echo "   Pattern searched: telegram_*.log"
fi
echo ""

# Check tmux sessions
echo "🖥️  TMUX SESSIONS STATUS"
echo "-----------------------"
if command -v tmux >/dev/null 2>&1; then
    SESSIONS=$(tmux list-sessions 2>/dev/null)
    if [ -n "$SESSIONS" ]; then
        echo "✅ Active tmux sessions:"
        echo "$SESSIONS" | sed 's/^/   /'
    else
        echo "⚠️  No active tmux sessions"
    fi
else
    echo "❌ tmux not available"
fi
echo ""

# Test tmux manager
echo "🔧 TMUX MANAGER TEST"
echo "-------------------"
if [ -f "$SCRIPT_DIR/scripts/enhanced_tmux_manager.sh" ]; then
    echo "✅ Enhanced tmux manager found"
    echo "   Testing logs command:"
    echo "   ====================="
    "$SCRIPT_DIR/scripts/enhanced_tmux_manager.sh" logs 2>&1 | sed 's/^/   /'
else
    echo "❌ Enhanced tmux manager not found"
    echo "   Expected: $SCRIPT_DIR/scripts/enhanced_tmux_manager.sh"
fi
echo ""

# Check telegram bot files
echo "🤖 TELEGRAM BOT FILES"
echo "--------------------"
BOT_FILES=("telegram_bot_listener.py" "telegram_bot_listener_fixed.py" "telegram_bot_listener_systemd.py")
for file in "${BOT_FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done
echo ""

# Configuration check
echo "⚙️  CONFIGURATION CHECK"
echo "----------------------"
if [ -f "$SCRIPT_DIR/training_config.yaml" ]; then
    echo "✅ training_config.yaml exists"
    if grep -q "telegram:" "$SCRIPT_DIR/training_config.yaml" 2>/dev/null; then
        echo "✅ Telegram configuration section found"
    else
        echo "⚠️  No telegram configuration section found"
    fi
else
    echo "❌ training_config.yaml missing"
fi

if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "✅ .env file exists"
else
    echo "⚠️  .env file not found (may use config file instead)"
fi
echo ""

# Summary and recommendations
echo "📋 SUMMARY & RECOMMENDATIONS"
echo "============================"

# Check for missing files
MISSING_FILES=()
[ ! -f "$BALANCE_FILE" ] && MISSING_FILES+=("balance.json")
[ ! -f "$TRADES_FILE" ] && MISSING_FILES+=("trades_report.csv")
[ ! -f "$PERFORMANCE_FILE" ] && MISSING_FILES+=("performance_metrics.json")

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Missing files for Telegram commands:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "🔧 FIX: Run the initialization script:"
    echo "   ./init_telegram_logs.sh"
    echo ""
fi

# Check if trading system is running
if [ -n "$TRADING_LOGS" ] && [ -n "$TELEGRAM_LOGS" ]; then
    echo "✅ Both trading and telegram logs present - system appears to be running"
elif [ -n "$TRADING_LOGS" ]; then
    echo "⚠️  Trading logs found but no telegram logs - telegram bot may not be running"
elif [ -n "$TELEGRAM_LOGS" ]; then
    echo "⚠️  Telegram logs found but no trading logs - trading system may not be running"
else
    echo "❌ No recent logs found - system may not be running"
    echo ""
    echo "🔧 FIX: Start the system:"
    echo "   ./start_system.sh"
fi
echo ""

echo "📞 TEST TELEGRAM COMMANDS"
echo "========================"
echo "Once fixed, test these commands in Telegram:"
echo "   /balance    - Should show current portfolio balance"
echo "   /trades     - Should show recent trading activity"
echo "   /status     - Should show system status"
echo "   /logs       - Should show recent system logs"
echo ""
echo "🎯 If issues persist, check:"
echo "   1. File permissions in logs/ directory"
echo "   2. Telegram bot configuration (bot_token, chat_id)"
echo "   3. Trading system is actively running and writing logs"
echo "   4. Correct Telegram bot script is being used"
