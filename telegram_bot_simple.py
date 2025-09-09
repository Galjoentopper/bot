#!/usr/bin/env python3
"""
Simple Telegram Bot Listener for Testing
This is a simplified version to debug the logging issue.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


# Setup logging
def setup_logging():
    """Setup logging to both file and console."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"telegram_simple_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Simple Telegram Bot Logging initialized")
    logger.info(f"📝 Writing to: {log_file}")
    return logger


logger = setup_logging()


def load_config():
    """Load configuration for Telegram bot."""
    logger.info("📁 Loading configuration...")

    # Try environment variables first
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        logger.info("✅ Found configuration in environment variables")
        return {"bot_token": bot_token, "chat_id": chat_id}

    # Try .env file
    env_file = project_root / ".env"
    if env_file.exists():
        logger.info("📄 Checking .env file...")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    bot_token = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("TELEGRAM_CHAT_ID="):
                    chat_id = line.split("=", 1)[1].strip().strip("\"'")

        if bot_token and chat_id:
            logger.info("✅ Found configuration in .env file")
            return {"bot_token": bot_token, "chat_id": chat_id}

    # Try YAML config
    try:
        import yaml

        config_file = project_root / "training_config.yaml"
        if config_file.exists():
            logger.info("📄 Checking training_config.yaml...")
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                telegram_config = config.get("notifications", {}).get("telegram", {})
                bot_token = telegram_config.get("bot_token")
                chat_id = telegram_config.get("chat_id")

                if bot_token and chat_id:
                    logger.info("✅ Found configuration in training_config.yaml")
                    return {"bot_token": bot_token, "chat_id": chat_id}
    except Exception as e:
        logger.warning(f"Could not load YAML config: {e}")

    logger.error("❌ No valid configuration found")
    return {}


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command."""
    logger.info(f"📨 Received /status command from {update.effective_user.first_name}")

    response = """
🤖 <b>Simple Telegram Bot Status</b>

✅ Bot is running and responding
📅 Time: {time}
👤 User: {user}
💬 Chat: {chat}

🎯 Available commands:
• /status - Show this status
• /test - Test logging
• /ping - Simple ping response
    """.format(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user=update.effective_user.first_name,
        chat=update.effective_chat.id,
    )

    await update.message.reply_text(response, parse_mode="HTML")
    logger.info("✅ Status command completed")


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /test command."""
    logger.info(f"🧪 Received /test command from {update.effective_user.first_name}")

    # Write test log entries
    logger.info("🔍 This is a test INFO log entry")
    logger.warning("⚠️ This is a test WARNING log entry")
    logger.error("❌ This is a test ERROR log entry (not a real error)")

    response = "🧪 <b>Test completed!</b>\n\nCheck the log files for test entries."
    await update.message.reply_text(response, parse_mode="HTML")
    logger.info("✅ Test command completed")


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ping command."""
    logger.info(f"🏓 Received /ping command from {update.effective_user.first_name}")
    await update.message.reply_text("🏓 Pong!")
    logger.info("✅ Ping command completed")


async def main():
    """Main function."""
    logger.info("=" * 50)
    logger.info("🚀 Starting Simple Telegram Bot")
    logger.info("=" * 50)

    # Load config
    config = load_config()
    if not config:
        logger.error("❌ Failed to load configuration - exiting")
        return

    bot_token = config["bot_token"]
    chat_id = config["chat_id"]

    logger.info(f"🔑 Bot token: {bot_token[:10]}...")
    logger.info(f"💬 Chat ID: {chat_id}")

    # Create application
    logger.info("🔧 Creating Telegram application...")
    application = Application.builder().token(bot_token).build()

    # Add handlers
    logger.info("⚙️ Adding command handlers...")
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("test", cmd_test))
    application.add_handler(CommandHandler("ping", cmd_ping))

    logger.info("✅ Simple Telegram Bot setup complete!")
    logger.info("📱 Send /status, /test, or /ping to test the bot")
    logger.info("🔄 Starting polling...")

    try:
        # Start polling
        await application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("🔚 Simple Telegram Bot shutting down")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Program interrupted")
    except Exception as e:
        logger.error(f"❌ Program error: {e}")
