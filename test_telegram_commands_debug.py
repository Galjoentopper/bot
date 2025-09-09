#!/usr/bin/env python3
"""
Debug script to test Telegram commands and capture detailed logs
"""
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.notifier.enhanced_telegram import EnhancedTelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("telegram_debug.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        logger.info("Loading .env file")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
                        logger.info(f"Loaded env var: {key.strip()}")
    else:
        logger.warning(".env file not found")


async def test_commands():
    """Test all Telegram commands and capture detailed logs"""

    # Load environment variables from .env file first
    load_env_file()

    # Get environment variables
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables")
        logger.error(f"TELEGRAM_BOT_TOKEN: {'***' + bot_token[-4:] if bot_token else 'None'}")
        logger.error(f"TELEGRAM_CHAT_ID: {chat_id if chat_id else 'None'}")
        return

    logger.info("Initializing EnhancedTelegramNotifier")
    notifier = EnhancedTelegramNotifier(bot_token, chat_id)

    # Test each command
    commands_to_test = [
        "/status",
        "/health",
        "/performance",
        "/balance",
        "/trades",
        "/logs",
        "/config",
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
