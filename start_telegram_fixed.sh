#!/bin/bash
# Start the fixed comprehensive Telegram bot

echo "🚀 Starting Comprehensive Trading Bot Telegram Listener..."

# Check if the fixed bot exists
if [ ! -f "telegram_bot_fixed.py" ]; then
    echo "❌ telegram_bot_fixed.py not found!"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please create it with your Telegram credentials."
    exit 1
fi

# Kill any existing telegram bots
echo "🛑 Stopping any existing Telegram bots..."
pkill -f telegram_bot_listener.py 2>/dev/null || true
pkill -f telegram_bot_fixed.py 2>/dev/null || true

sleep 2

# Start the comprehensive bot in the background
echo "▶️ Starting comprehensive Telegram bot with all trading commands..."
nohup python telegram_bot_fixed.py > logs/telegram_bot.log 2>&1 &
BOT_PID=$!

echo "✅ Telegram bot started with PID: $BOT_PID"
echo "📱 Available commands:"
echo "   • /status - Bot status"
echo "   • /start - Start trading"
echo "   • /stop - Stop trading"
echo "   • /restart - Restart trading"
echo "   • /performance - Performance metrics"
echo "   • /health - System health"
echo "   • /balance - Current balance"
echo "   • /trades - Recent trades"
echo "   • /logs - Recent logs"
echo "   • /ping - Test connectivity"
echo "   • /version - Bot version"

echo ""
echo "📝 Check logs with: tail -f logs/telegram_bot.log"
echo "🛑 Stop bot with: pkill -f telegram_bot_fixed.py"

echo "✅ Comprehensive Telegram bot is now running!"
