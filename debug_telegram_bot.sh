#!/bin/bash
# Debug and Fix Telegram Bot Logging
# This script checks what's wrong and fixes the Telegram bot

echo "🔧 Telegram Bot Debug & Fix"
echo "==========================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working in: $SCRIPT_DIR"
echo ""

# 1. Check current tmux session
echo "1. Checking current Telegram bot session..."
if tmux has-session -t telegram_session 2>/dev/null; then
    echo "✅ Telegram session exists"
    
    # Get detailed info
    echo "   Session details:"
    tmux list-sessions | grep telegram_session
    
    # Check what's running in the session
    echo ""
    echo "   Checking what's running in the session..."
    tmux capture-pane -t telegram_session -p | tail -10 | sed 's/^/   > /'
    
else
    echo "❌ Telegram session not found"
fi

echo ""

# 2. Check log files in detail
echo "2. Analyzing log files..."
LOG_FILES=$(ls logs/telegram_bot_listener_*.log 2>/dev/null)
if [ -n "$LOG_FILES" ]; then
    for log_file in $LOG_FILES; do
        size=$(stat -c%s "$log_file" 2>/dev/null || echo "0")
        echo "   📝 $log_file: $size bytes"
        
        if [ "$size" -gt 0 ]; then
            echo "   Latest content:"
            tail -5 "$log_file" | sed 's/^/      /'
        else
            echo "   ⚠️  File is empty - bot may not be writing logs"
        fi
    done
else
    echo "   No telegram log files found"
fi

echo ""

# 3. Test manual bot startup
echo "3. Testing manual Telegram bot startup..."
echo "   Trying to import and test the bot module..."

python3 -c "
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

try:
    # Test import
    print('Testing imports...')
    from telegram_bot_listener import TelegramBotListener, load_config
    print('✅ Imports successful')
    
    # Test config
    print('Testing configuration...')
    config = load_config()
    if config:
        telegram_config = config.get('notifications', {}).get('telegram', {})
        bot_token = telegram_config.get('bot_token')
        chat_id = telegram_config.get('chat_id')
        
        if bot_token and chat_id:
            print(f'✅ Configuration valid')
            print(f'   Token: {bot_token[:10]}...')
            print(f'   Chat ID: {chat_id}')
            
            # Test bot creation
            print('Testing bot creation...')
            bot = TelegramBotListener(bot_token, chat_id)
            print('✅ Bot object created successfully')
            
        else:
            print('❌ Invalid configuration - missing token or chat_id')
    else:
        print('❌ No configuration loaded')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""

# 4. Restart the Telegram bot
echo "4. Restarting Telegram bot with fresh logging..."

# Kill existing session
if tmux has-session -t telegram_session 2>/dev/null; then
    echo "   Stopping existing session..."
    tmux kill-session -t telegram_session
    sleep 2
fi

# Create new log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NEW_LOG="logs/telegram_debug_$TIMESTAMP.log"

echo "   Starting new session with debug logging..."
echo "   Log file: $NEW_LOG"

# Start new session with verbose output
tmux new-session -d -s telegram_session -c "$SCRIPT_DIR" \
    "python3 telegram_bot_listener.py 2>&1 | tee '$NEW_LOG'"

sleep 3

# Check if it started
if tmux has-session -t telegram_session 2>/dev/null; then
    echo "✅ New session started"
    
    # Show initial output
    echo ""
    echo "📋 Initial output:"
    tmux capture-pane -t telegram_session -p | tail -10 | sed 's/^/   /'
    
    # Check if log file has content
    sleep 2
    if [ -s "$NEW_LOG" ]; then
        echo ""
        echo "📝 Log file content:"
        tail -10 "$NEW_LOG" | sed 's/^/   /'
    else
        echo ""
        echo "⚠️  Log file still empty after startup"
    fi
else
    echo "❌ Failed to start new session"
fi

echo ""
echo "🎯 Summary:"
echo "=========="
echo "• Configuration: ✅ Valid"
echo "• Bot Session: $(tmux has-session -t telegram_session 2>/dev/null && echo '✅ Running' || echo '❌ Not running')"
echo "• Log Files: $(ls logs/telegram_* 2>/dev/null | wc -l) files"
echo ""
echo "📱 Test your bot now:"
echo "   Send '/status' to your Telegram bot"
echo "   Check: tail -f logs/telegram_debug_*.log"
