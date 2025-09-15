#!/usr/bin/env python3
"""
Test Script for New Telegram Bot System
=======================================

Tests the focused, reliable Telegram bot implementation.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_telegram_configuration():
    """Test Telegram bot configuration."""
    print("🔧 Testing Telegram Configuration")
    print("=" * 40)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN environment variable not set")
        return False

    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID environment variable not set")
        return False

    # Basic token format validation
    if not bot_token.count(":") == 1:
        print("❌ TELEGRAM_BOT_TOKEN appears to be invalid format")
        return False

    try:
        # Chat ID should be numeric
        int(chat_id)
        print(f"✅ Bot token: {bot_token[:10]}...{bot_token[-10:]}")
        print(f"✅ Chat ID: {chat_id}")
        return True
    except ValueError:
        print("❌ TELEGRAM_CHAT_ID should be numeric")
        return False


async def test_bot_creation():
    """Test bot creation and initialization."""
    print("\n🤖 Testing Bot Creation")
    print("=" * 30)

    try:
        from src.notifications.telegram_bot import create_telegram_bot

        bot = create_telegram_bot()
        if not bot:
            print("❌ Failed to create bot instance")
            return False

        print("✅ Bot instance created successfully")

        # Test initialization
        if await bot.initialize():
            print("✅ Bot initialized successfully")
            return bot
        else:
            print("❌ Bot initialization failed")
            return False

    except Exception as e:
        print(f"❌ Error creating/initializing bot: {e}")
        return False


async def test_startup_message(bot):
    """Test startup message functionality."""
    print("\n🚀 Testing Startup Message")
    print("=" * 30)

    try:
        await bot._send_startup_message()
        print("✅ Startup message sent successfully")
        return True
    except Exception as e:
        print(f"❌ Error sending startup message: {e}")
        return False


async def test_commands(bot):
    """Test command functionality."""
    print("\n🎮 Testing Commands")
    print("=" * 25)

    # Mock update and context for testing
    class MockUpdate:
        class MockMessage:
            async def reply_text(self, text, parse_mode=None):
                print(f"📤 Bot would send: {text[:100]}...")

        message = MockMessage()

    class MockContext:
        args = []

    update = MockUpdate()
    context = MockContext()

    # Test key commands
    commands_to_test = [
        ("help", bot._cmd_help),
        ("status", bot._cmd_status),
        ("balance", bot._cmd_balance),
        ("health", bot._cmd_health),
        ("uptime", bot._cmd_uptime),
    ]

    success_count = 0
    for cmd_name, cmd_handler in commands_to_test:
        try:
            await cmd_handler(update, context)
            print(f"✅ /{cmd_name} command working")
            success_count += 1
        except Exception as e:
            print(f"❌ /{cmd_name} command failed: {e}")

    print(f"📊 Commands test: {success_count}/{len(commands_to_test)} working")
    return success_count > 0


async def test_error_notification(bot):
    """Test error notification functionality."""
    print("\n🚨 Testing Error Notifications")
    print("=" * 35)

    try:
        # Create a test error
        test_error = ValueError("Test error for notification")
        await bot.send_error_notification(test_error, "Test context")
        print("✅ Error notification sent successfully")
        return True
    except Exception as e:
        print(f"❌ Error notification failed: {e}")
        return False


async def test_trade_recording(bot):
    """Test trade recording functionality."""
    print("\n📊 Testing Trade Recording")
    print("=" * 30)

    try:
        # Create test trade data
        test_trade = {
            "symbol": "BTCEUR",
            "action": "BUY",
            "quantity": 0.001,
            "price": 95000,
            "realized_pnl": 15.75,
            "confidence": 0.85,
            "reason": "Test trade execution",
        }

        bot.record_trade(test_trade)
        print("✅ Trade recording successful")

        # Check daily stats were updated
        if bot.daily_stats["trades_count"] > 0:
            print(f"✅ Daily stats updated: {bot.daily_stats['trades_count']} trades")
            return True
        else:
            print("❌ Daily stats not updated")
            return False

    except Exception as e:
        print(f"❌ Trade recording failed: {e}")
        return False


async def main():
    """Run all Telegram bot tests."""
    print("🚀 Enhanced Telegram Bot Test Suite")
    print("=" * 50)

    # Test configuration
    if not test_telegram_configuration():
        print("\n❌ Configuration test failed. Please set environment variables:")
        print("export TELEGRAM_BOT_TOKEN=your_bot_token")
        print("export TELEGRAM_CHAT_ID=your_chat_id")
        return False

    # Test bot creation
    bot = await test_bot_creation()
    if not bot:
        print("\n❌ Bot creation failed")
        return False

    # Run functionality tests
    tests = [test_startup_message, test_commands, test_error_notification, test_trade_recording]

    results = []
    for test_func in tests:
        try:
            result = await test_func(bot)
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)

    # Cleanup
    try:
        await bot.stop()
        print("\n🛑 Bot stopped cleanly")
    except:
        pass

    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 25)

    passed = sum(results)
    total = len(results)

    test_names = ["startup_message", "commands", "error_notification", "trade_recording"]

    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total+1} tests passed")  # +1 for config test

    if passed == total:
        print("🎉 All tests passed! Telegram bot is ready.")
        print("\n💡 Next steps:")
        print("1. Run: python bin/telegram_bot")
        print("2. Send /help to your bot")
        print("3. Try commands like /status, /balance, /health")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
