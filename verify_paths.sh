#!/bin/bash
# Comprehensive path verification script

echo "🔍 TRADING BOT PATH VERIFICATION"
echo "================================"

CORRECT_PATH="/opt/trading_bot/bot/"
INCORRECT_PATTERN="/opt/trading_bot/[^b]"

echo "Checking for incorrect path references..."
echo "Correct pattern should be: $CORRECT_PATH"
echo ""

# Check key files for path correctness
FILES_TO_CHECK=(
    "telegram_bot_listener.py"
    "telegram_bot_listener_fixed.py"
    "telegram_bot_listener_systemd.py"
    "src/notifier/enhanced_telegram.py"
    "telegram-bot-listener.service"
    "trading_bot_monitor"
    "scripts/enhanced_tmux_manager.sh"
    "deploy_full_system.sh"
)

echo "📋 CHECKING KEY FILES"
echo "--------------------"

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        # Count correct vs incorrect paths
        CORRECT_COUNT=$(grep -c "/opt/trading_bot/bot/" "$file" 2>/dev/null || echo "0")
        INCORRECT_COUNT=$(grep -cE "/opt/trading_bot/[^b]" "$file" 2>/dev/null || echo "0")
        
        if [ "$INCORRECT_COUNT" -gt 0 ]; then
            echo "❌ $file: $INCORRECT_COUNT incorrect paths found"
            echo "   Incorrect paths:"
            grep -nE "/opt/trading_bot/[^b]" "$file" | sed 's/^/     /'
        elif [ "$CORRECT_COUNT" -gt 0 ]; then
            echo "✅ $file: $CORRECT_COUNT correct paths"
        else
            echo "⚪ $file: No trading_bot paths found"
        fi
    else
        echo "⚠️  $file: File not found"
    fi
done

echo ""
echo "🔍 SEARCHING ALL FILES FOR INCORRECT PATHS"
echo "==========================================="

# Search all files for incorrect patterns
INCORRECT_FILES=$(find . -type f -name "*.py" -o -name "*.sh" -o -name "*.service" -o -name "*.yaml" | xargs grep -l "/opt/trading_bot/[^b]" 2>/dev/null)

if [ -n "$INCORRECT_FILES" ]; then
    echo "❌ Files with incorrect paths found:"
    for file in $INCORRECT_FILES; do
        echo "   $file"
        grep -n "/opt/trading_bot/[^b]" "$file" | sed 's/^/     /'
    done
else
    echo "✅ No files with incorrect paths found!"
fi

echo ""
echo "📊 SUMMARY"
echo "=========="

TOTAL_INCORRECT=$(find . -type f -name "*.py" -o -name "*.sh" -o -name "*.service" -o -name "*.yaml" | xargs grep -c "/opt/trading_bot/[^b]" 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
TOTAL_CORRECT=$(find . -type f -name "*.py" -o -name "*.sh" -o -name "*.service" -o -name "*.yaml" | xargs grep -c "/opt/trading_bot/bot/" 2>/dev/null | awk '{sum+=$1} END {print sum+0}')

echo "✅ Correct paths: $TOTAL_CORRECT"
echo "❌ Incorrect paths: $TOTAL_INCORRECT"

if [ "$TOTAL_INCORRECT" -eq 0 ]; then
    echo ""
    echo "🎉 ALL PATHS CORRECTED!"
    echo "The system should now work correctly with:"
    echo "  - Telegram commands (/balance, /trades, /performance, /config)"
    echo "  - Log file access"
    echo "  - Script execution"
    echo "  - Health checks"
else
    echo ""
    echo "⚠️  Some paths still need fixing"
    echo "Run this script again after fixing the issues above"
fi

echo ""
echo "🧪 NEXT STEPS"
echo "============"
echo "1. Run: ./init_telegram_logs.sh"
echo "2. Run: ./start_system.sh"
echo "3. Test Telegram commands:"
echo "   /balance - Portfolio balance"
echo "   /performance - Performance metrics"
echo "   /config - Configuration settings"
echo "   /trades - Recent trades"
