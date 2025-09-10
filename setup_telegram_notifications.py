#!/usr/bin/env python3
"""
Setup Telegram Notifications
============================

Helper script to set up Telegram notifications for the training pipeline.
"""

import os
import sys
from pathlib import Path

print("🤖 Telegram Notifications Setup")
print("=" * 50)

# Check if Telegram library is available
try:
    from telegram import Bot
    print("✅ python-telegram-bot library is installed")
except ImportError:
    print("❌ python-telegram-bot library not found")
    print("Install it with: pip install python-telegram-bot")
    sys.exit(1)

print("\n📱 To set up Telegram notifications, you need:")
print("1. Create a Telegram bot by messaging @BotFather")
print("2. Get your bot token from @BotFather")
print("3. Start a chat with your bot and send any message")
print("4. Get your chat ID")

print("\n🔧 Steps to get your chat ID:")
print("1. Send a message to your bot")
print("2. Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
print("3. Look for 'chat':{'id': XXXXXXXXX}")
print("4. That number is your CHAT_ID")

print(f"\n📄 Current .env file location: {Path('/notebooks/bot/.env')}")
print("\n🔧 Update your .env file with:")
print("TELEGRAM_BOT_TOKEN=your_actual_bot_token_here")
print("TELEGRAM_CHAT_ID=your_actual_chat_id_here")

print("\n🧪 After setting up, test with:")
print("python test_telegram_notifications.py")

# Check current configuration
env_file = Path("/notebooks/bot/.env")
if env_file.exists():
    with open(env_file, "r") as f:
        content = f.read()
        
    if "your_telegram_token_here" in content:
        print("\n⚠️ Telegram credentials are still placeholder values")
        print("Please update them with your actual bot token and chat ID")
    else:
        print("\n✅ Telegram credentials appear to be configured")
        print("Run 'python test_telegram_notifications.py' to test them")
else:
    print("\n❌ .env file not found")
    
print("\n" + "=" * 50)
print("📚 For more help, see: https://core.telegram.org/bots#6-botfather")