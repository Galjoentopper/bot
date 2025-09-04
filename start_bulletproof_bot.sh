#!/bin/bash

# Quick Start: Bulletproof Telegram Bot
# Run this to test the new telegram bot locally

echo "🚀 Starting Bulletproof Telegram Bot Test..."

# Check if we have the config
if [ ! -f "training_config.yaml" ]; then
    echo "❌ training_config.yaml not found"
    exit 1
fi

# Check if we have the required files
if [ ! -f "telegram_bot_bulletproof.py" ]; then
    echo "❌ telegram_bot_bulletproof.py not found"
    exit 1
fi

# Check if logs directory exists
mkdir -p logs

# Start the bulletproof telegram bot
echo "📡 Starting bulletproof Telegram bot..."
echo "   - Event loop: Handled in separate thread"
echo "   - Commands: 11 total (/status, /balance, /trades, etc.)"
echo "   - Log file: logs/telegram_bulletproof.log"
echo ""
echo "✨ Bot should start without event loop errors!"
echo ""

# Run the bot
python3 telegram_bot_bulletproof.py
