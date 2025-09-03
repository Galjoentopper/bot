#!/bin/bash
# Test Telegram Bot Logging
# This script helps diagnose Telegram bot logging issues

echo "🔍 Telegram Bot Logging Diagnostics"
echo "===================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Check if logs directory exists
echo "1. Checking logs directory..."
if [ -d "logs" ]; then
    echo "✅ logs/ directory exists"
    echo "   Current log files:"
    ls -la logs/ | grep telegram || echo "   No telegram log files found"
else
    echo "⚠️  logs/ directory does not exist"
    echo "   Creating logs directory..."
    mkdir -p logs
fi

echo ""

# Check configuration files
echo "2. Checking configuration files..."
CONFIG_FOUND=false

if [ -f "training_config.yaml" ]; then
    echo "✅ training_config.yaml found"
    if grep -q "telegram" training_config.yaml; then
        echo "   Contains telegram configuration"
        CONFIG_FOUND=true
    else
        echo "   ⚠️  No telegram configuration found in training_config.yaml"
    fi
fi

if [ -f ".env" ]; then
    echo "✅ .env file found"
    if grep -q "TELEGRAM_BOT_TOKEN" .env; then
        echo "   Contains TELEGRAM_BOT_TOKEN"
        CONFIG_FOUND=true
    else
        echo "   ⚠️  No TELEGRAM_BOT_TOKEN found in .env"
    fi
fi

if [ ! -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "✅ TELEGRAM_BOT_TOKEN environment variable set"
    CONFIG_FOUND=true
fi

if [ "$CONFIG_FOUND" = false ]; then
    echo "❌ No Telegram configuration found anywhere"
    echo ""
    echo "📝 To fix this, you need to configure Telegram in one of these ways:"
    echo ""
    echo "Option 1: Add to training_config.yaml:"
    echo "notifications:"
    echo "  telegram:"
    echo "    bot_token: 'YOUR_BOT_TOKEN'"
    echo "    chat_id: 'YOUR_CHAT_ID'"
    echo ""
    echo "Option 2: Create .env file:"
    echo "TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN"
    echo "TELEGRAM_CHAT_ID=YOUR_CHAT_ID"
    echo ""
    echo "Option 3: Set environment variables:"
    echo "export TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN"
    echo "export TELEGRAM_CHAT_ID=YOUR_CHAT_ID"
fi

echo ""

# Test the telegram bot listener
echo "3. Testing Telegram bot listener (dry run)..."
if python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

try:
    from telegram_bot_listener import load_config
    config = load_config()
    
    if config:
        telegram_config = config.get('notifications', {}).get('telegram', {})
        bot_token = telegram_config.get('bot_token')
        chat_id = telegram_config.get('chat_id')
        
        if bot_token and chat_id:
            print('✅ Configuration loaded successfully')
            print(f'   Bot token: {bot_token[:10]}...')
            print(f'   Chat ID: {chat_id}')
        else:
            print('❌ Bot token or chat ID missing from configuration')
    else:
        print('❌ No configuration loaded')
        
except Exception as e:
    print(f'❌ Error testing configuration: {e}')
"; then
    echo "Configuration test completed"
else
    echo "⚠️  Configuration test failed"
fi

echo ""

# Check if bot is currently running
echo "4. Checking if Telegram bot is running..."
if tmux has-session -t telegram_session 2>/dev/null; then
    echo "✅ Telegram bot session exists"
    echo "   Session info:"
    tmux list-windows -t telegram_session 2>/dev/null
    
    # Check recent logs
    echo ""
    echo "📊 Recent Telegram bot activity:"
    if [ -f "logs/telegram_bot_listener"*.log ]; then
        echo "   Latest log entries:"
        tail -10 logs/telegram_bot_listener_*.log 2>/dev/null | tail -10
    else
        echo "   No log files found yet"
    fi
else
    echo "⚠️  Telegram bot session not running"
fi

echo ""
echo "🔧 Next Steps:"
echo "=============="
echo "1. If configuration is missing: Set up bot token and chat ID"
echo "2. If bot is not running: ./start_system.sh"
echo "3. Check logs: tail -f logs/telegram_bot_listener_*.log"
echo "4. Test commands in Telegram app"
echo ""
echo "📱 To test Telegram bot:"
echo "   Send '/status' to your bot in Telegram"
echo "   Check if response appears and logs are created"
