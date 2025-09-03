#!/usr/bin/env python3
"""
Debug script to test Telegram commands and capture detailed logs
"""
import asyncio
import logging
import os
import sys
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.notifier.enhanced_telegram import EnhancedTelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_commands():
    """Test all Telegram commands and capture detailed logs"""

    # Get environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables")
        return

    logger.info("Initializing EnhancedTelegramNotifier")
    notifier = EnhancedTelegramNotifier(bot_token, chat_id)

    # Test each command
    commands_to_test = [
        '/status',
        '/health',
        '/performance',
        '/balance',
        '/trades',
        '/logs',
        '/config'
    ]

    logger.info("Starting command tests...")

    for cmd in commands_to_test:
        logger.info(f"Testing command: {cmd}")
        try:
            result = await notifier.handle_command(cmd)
            logger.info(f"Command {cmd} result: {result[:100]}...")
        except Exception as e:
            logger.error(f"Command {cmd} failed: {e}", exc_info=True)

        # Small delay between commands
        await asyncio.sleep(1)

    logger.info("Command testing completed")

def main():
    """Main function"""
    logger.info("Starting Telegram commands debug test")

    # Run the test
    asyncio.run(test_commands())

    logger.info("Debug test completed. Check telegram_debug.log for details")

if __name__ == "__main__":
    main()