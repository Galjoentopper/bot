#!/usr/bin/env python3
"""
Test script for Telegram bot commands
"""
import asyncio
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.notifier.enhanced_telegram import EnhancedTelegramNotifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_print(text: str, max_length: int = 100) -> None:
    """Print text safely, handling Unicode encoding issues."""
    try:
        # Try to encode and decode to handle Unicode properly
        safe_text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        preview = safe_text[:max_length]
        if len(safe_text) > max_length:
            preview += "..."
        print(preview)
    except Exception as e:
        print(f"[Unicode encoding error: {e}]")

async def test_telegram_commands():
    """Test all Telegram commands."""
    try:
        # Initialize notifier (this will work even without valid tokens for testing)
        notifier = EnhancedTelegramNotifier(
            bot_token="test_token",
            chat_id="test_chat_id"
        )

        # Test all commands
        commands_to_test = [
            '/status',
            '/start',
            '/stop',
            '/restart',
            '/performance',
            '/health',
            '/balance',
            '/trades',
            '/logs',
            '/config'
        ]

        print("Testing Telegram Commands")
        print("=" * 50)

        for command in commands_to_test:
            try:
                print(f"\nTesting command: {command}")
                response = await notifier.handle_command(command, [])
                print(f"Response received ({len(response)} chars)")
                print("Preview:", end=" ")
                safe_print(response)
            except Exception as e:
                print(f"Error testing {command}: {e}")

        print("\n" + "=" * 50)
        print("Command testing completed!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Starting Telegram Commands Test")
    success = asyncio.run(test_telegram_commands())
    sys.exit(0 if success else 1)