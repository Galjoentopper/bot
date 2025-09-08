#!/bin/bash
# Debug script for Telegram bot issues

echo "🔍 Telegram Bot Debug Script"
echo "============================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

echo ""
echo "📁 Current Directory: $SCRIPT_DIR"
echo "📊 Log Directory: $LOG_DIR"

echo ""
echo "📋 Checking tmux sessions..."
tmux list-sessions 2>/dev/null || echo "❌ No tmux sessions found"

echo ""
echo "📋 Checking running processes..."
ps aux | grep -E "(telegram|python)" | grep -v grep || echo "❌ No Python/Telegram processes found"

echo ""
echo "📋 Checking log files..."
ls -la $LOG_DIR/ 2>/dev/null || echo "❌ Log directory not found"

echo ""
echo "📋 Recent log files:"
find $LOG_DIR -name "*.log" -type f -mmin -60 2>/dev/null | head -10 || echo "❌ No recent log files found"

echo ""
echo "📋 Telegram configuration check..."
if [ -f "$SCRIPT_DIR/training_config.yaml" ]; then
    echo "✅ Configuration file found"
    grep -A 5 "telegram:" $SCRIPT_DIR/training_config.yaml || echo "❌ Telegram config not found"
else
    echo "❌ Configuration file not found"
fi

echo ""
echo "📋 Testing bot connectivity..."
BOT_TOKEN=$(grep "bot_token:" $SCRIPT_DIR/training_config.yaml | cut -d"'" -f2)
CHAT_ID=$(grep "chat_id:" $SCRIPT_DIR/training_config.yaml | cut -d"'" -f2)

if [ ! -z "$BOT_TOKEN" ] && [ ! -z "$CHAT_ID" ]; then
    echo "🔗 Testing bot token..."
    curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getMe" | grep -q "ok.*true" && echo "✅ Bot token valid" || echo "❌ Bot token invalid"

    echo "💬 Testing chat connectivity..."
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
      -d "chat_id=${CHAT_ID}" \
      -d "text=🤖 Debug: Bot connectivity test from server" | grep -q "ok.*true" && echo "✅ Chat ID valid" || echo "❌ Chat ID invalid"
else
    echo "❌ Bot token or chat ID not found in config"
fi

echo ""
echo "📋 Manual bot test..."
echo "Run this command to test the bot manually:"
echo "cd $SCRIPT_DIR && python3 telegram_bot_listener.py"
echo ""
echo "Then send /help in Telegram to test"

echo ""
echo "🎯 Next steps:"
echo "1. Check if tmux sessions are running: tmux list-sessions"
echo "2. Attach to telegram session: tmux attach-session -t telegram_session"
echo "3. Check for error messages in the attached session"
echo "4. Verify bot token and chat ID are correct"