#!/usr/bin/env python3
"""
Simple Telegram Test Script
Tests the Telegram bot connection and message sending functionality.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.notifier.telegram import TelegramNotifier

    print("✅ Successfully imported TelegramNotifier")
except ImportError as e:
    print(f"❌ Failed to import TelegramNotifier: {e}")
    sys.exit(1)


async def test_telegram_connection():
    """Test Telegram connection with the configured credentials."""

    # Test configuration
    config = {
        "notifications": {
            "telegram": {
                "enabled": True,
                "bot_token": "7733436451:AAH6Sls8uL4fEgd6Ty7VEKSBIMauhaVkN4c",
                "chat_id": "7988790407",
            }
        }
    }

    print("🔧 Initializing Telegram notifier...")
    try:
        notifier = TelegramNotifier.from_config(config)
        print(f"📊 Telegram notifier status: enabled={notifier.enabled}")

        if not notifier.enabled:
            print("❌ Telegram notifier is disabled. Check configuration.")
            return False

        # Test connection
        print("📤 Testing connection with a simple message...")
        test_message = """
🧪 <b>TELEGRAM TEST</b>

This is a test message to verify Telegram integration is working correctly.

<b>Status:</b> Connection successful
<b>Time:</b> Test completed

<i>Automated test from trading system</i>
"""

        success = await notifier.send_message(test_message)
        if success:
            print("✅ Telegram message sent successfully!")
            return True
        else:
            print("❌ Failed to send Telegram message")
            return False

    except Exception as e:
        print(f"❌ Error during Telegram test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_manual_credentials():
    """Test with manually provided credentials."""

    print("\n🔧 Testing with manual credentials...")

    try:
        bot_token = "7733436451:AAH6Sls8uL4fEgd6Ty7VEKSBIMauhaVkN4c"
        chat_id = "7988790407"

        notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id, enabled=True)

        print(f"📊 Manual notifier status: enabled={notifier.enabled}")

        if not notifier.enabled:
            print("❌ Manual Telegram notifier is disabled")
            return False

        test_message = """
🧪 <b>MANUAL TELEGRAM TEST</b>

Testing with manually provided credentials.

<b>Bot Token:</b> {bot_token[:10]}...
<b>Chat ID:</b> {chat_id}

<i>Manual test from trading system</i>
"""

        success = await notifier.send_message(test_message)
        if success:
            print("✅ Manual Telegram message sent successfully!")
            return True
        else:
            print("❌ Failed to send manual Telegram message")
            return False

    except Exception as e:
        print(f"❌ Error during manual Telegram test: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_environment():
    """Check environment variables and configuration."""
    print("🔍 Checking environment...")

    # Check for environment variables
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if telegram_token:
        print(f"✅ Found TELEGRAM_BOT_TOKEN: {telegram_token[:10]}...")
    else:
        print("⚠️  TELEGRAM_BOT_TOKEN not found in environment")

    if telegram_chat_id:
        print(f"✅ Found TELEGRAM_CHAT_ID: {telegram_chat_id}")
    else:
        print("⚠️  TELEGRAM_CHAT_ID not found in environment")

    # Check Python path
    print(f"📁 Python path includes: {project_root}")
    print(f"📁 src directory exists: {(project_root / 'src').exists()}")


async def main():
    """Main test function."""
    print("🚀 Starting Telegram Integration Test")
    print("=" * 50)

    # Check environment
    check_environment()
    print()

    # Test 1: Configuration-based test
    print("📋 TEST 1: Configuration-based Telegram test")
    test1_result = await test_telegram_connection()
    print()

    # Test 2: Manual credentials test
    print("📋 TEST 2: Manual credentials Telegram test")
    test2_result = await test_manual_credentials()
    print()

    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Configuration test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"Manual test: {'✅ PASSED' if test2_result else '❌ FAILED'}")

    if test1_result or test2_result:
        print("\n🎉 Telegram integration is working!")
    else:
        print("\n❌ Telegram integration has issues. Check:")
        print("   - Bot token is correct")
        print("   - Chat ID is correct")
        print("   - Bot has permission to send messages")
        print("   - Network connectivity")
        print("   - python-telegram-bot library is installed")


if __name__ == "__main__":
    asyncio.run(main())
