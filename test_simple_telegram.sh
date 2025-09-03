#!/bin/bash

echo "🚀 Testing Simple Telegram Bot"
echo "================================"

# Get script directory (bot root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if telegram bot config exists
echo "🔍 Checking configuration..."
if [ -f "training_config.yaml" ]; then
    echo "✅ Found training_config.yaml"
else
    echo "❌ training_config.yaml not found"
fi

if [ -f ".env" ]; then
    echo "✅ Found .env file"
else
    echo "⚠️ No .env file found"
fi

# Check environment variables
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "✅ TELEGRAM_BOT_TOKEN environment variable set"
else
    echo "⚠️ TELEGRAM_BOT_TOKEN environment variable not set"
fi

# Create logs directory
mkdir -p logs

echo ""
echo "🤖 Starting Simple Telegram Bot..."
echo "Press Ctrl+C to stop"
echo ""

# Run the simple bot
python3 telegram_bot_simple.py
