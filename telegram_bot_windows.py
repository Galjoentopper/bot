#!/usr/bin/env python3
"""
Windows-Compatible Telegram Bot Listener
Simplified version without Unicode emojis for Windows compatibility.
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
    log_file = log_dir / f"telegram_listener_{timestamp}.log"

    # Configure logging with better Windows compatibility
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # Override any existing logging config
    )

    logger = logging.getLogger(__name__)
    logger.info("STARTUP: Robust Telegram Bot Listener initialized")
    logger.info(f"LOG_FILE: Writing to: {log_file}")
    return logger


logger = setup_logging()


def load_config():
    """Load configuration for Telegram bot."""
    logger.info("CONFIG: Loading configuration...")

    # Try environment variables first
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        logger.info("CONFIG: Found configuration in environment variables")
        return {"bot_token": bot_token, "chat_id": chat_id}

    # Try .env file
    env_file = project_root / ".env"
    if env_file.exists():
        logger.info("CONFIG: Checking .env file...")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    bot_token = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("TELEGRAM_CHAT_ID="):
                    chat_id = line.split("=", 1)[1].strip().strip("\"'")

        if bot_token and chat_id:
            logger.info("CONFIG: Found configuration in .env file")
            return {"bot_token": bot_token, "chat_id": chat_id}

    # Try YAML config
    try:
        import yaml

        config_file = project_root / "training_config.yaml"
        if config_file.exists():
            logger.info("CONFIG: Checking training_config.yaml...")
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                telegram_config = config.get("notifications", {}).get("telegram", {})
                bot_token = telegram_config.get("bot_token")
                chat_id = telegram_config.get("chat_id")

                if bot_token and chat_id:
                    logger.info("CONFIG: Found configuration in training_config.yaml")
                    return {"bot_token": bot_token, "chat_id": chat_id}
    except Exception as e:
        logger.warning(f"CONFIG: Could not load YAML config: {e}")

    logger.error("CONFIG: No valid configuration found")
    return {}


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command."""
    logger.info(f"COMMAND: Received /status command from {update.effective_user.first_name}")

    response = f"""
<b>Telegram Bot Listener Status</b>

STATUS: Bot is running and responding
TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
USER: {update.effective_user.first_name}
CHAT: {update.effective_chat.id}

Available commands:
• /status - Show this status
• /test - Test logging functionality
• /ping - Simple ping response
• /logs - Show recent log entries
    """

    await update.message.reply_text(response, parse_mode="HTML")
    logger.info("COMMAND: Status command completed successfully")


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /test command."""
    logger.info(f"COMMAND: Received /test command from {update.effective_user.first_name}")

    # Write test log entries
    logger.info("TEST: This is a test INFO log entry")
    logger.warning("TEST: This is a test WARNING log entry")
    logger.error("TEST: This is a test ERROR log entry (not a real error)")

    response = """
<b>Test completed successfully!</b>

The following test entries were written to the log:
• INFO level test message
• WARNING level test message
• ERROR level test message

Check the log files in the logs/ directory for these entries.
    """
    await update.message.reply_text(response, parse_mode="HTML")
    logger.info("COMMAND: Test command completed successfully")


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ping command."""
    logger.info(f"COMMAND: Received /ping command from {update.effective_user.first_name}")
    await update.message.reply_text("PONG: Bot is responding")
    logger.info("COMMAND: Ping command completed successfully")


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /logs command - show recent log entries."""
    logger.info(f"COMMAND: Received /logs command from {update.effective_user.first_name}")

    try:
        # Find the most recent log file
        log_dir = project_root / "logs"
        log_files = list(log_dir.glob("telegram_listener_*.log"))

        if not log_files:
            await update.message.reply_text("No log files found.")
            return

        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)

        # Read last 10 lines
        with open(latest_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
            recent_lines = lines[-10:] if len(lines) > 10 else lines

        log_text = "".join(recent_lines)
        response = f"<b>Recent Log Entries:</b>\n<pre>{log_text}</pre>"

        # Telegram message limit is 4096 characters
        if len(response) > 4000:
            response = response[:4000] + "...\n[Log truncated]"

        await update.message.reply_text(response, parse_mode="HTML")
        logger.info("COMMAND: Logs command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in logs command: {e}")
        await update.message.reply_text(f"Error reading logs: {e}")


def main():
    """Main function - non-async version to avoid event loop issues."""
    logger.info("=" * 50)
    logger.info("STARTUP: Starting Robust Telegram Bot Listener")
    logger.info("=" * 50)

    # Load config
    config = load_config()
    if not config:
        logger.error("STARTUP: Failed to load configuration - exiting")
        return

    bot_token = config["bot_token"]
    chat_id = config["chat_id"]

    logger.info(f"STARTUP: Bot token configured: {bot_token[:10]}...")
    logger.info(f"STARTUP: Chat ID configured: {chat_id}")

    # Create application
    logger.info("STARTUP: Creating Telegram application...")
    application = Application.builder().token(bot_token).build()

    # Add handlers
    logger.info("STARTUP: Adding command handlers...")
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("test", cmd_test))
    application.add_handler(CommandHandler("ping", cmd_ping))
    application.add_handler(CommandHandler("logs", cmd_logs))

    logger.info("STARTUP: Robust Telegram Bot Listener setup complete!")
    logger.info("STARTUP: Send /status, /test, /ping, or /logs to test the bot")
    logger.info("STARTUP: Starting polling...")

    try:
        # Use blocking run_polling to avoid async issues
        application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("SHUTDOWN: Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"SHUTDOWN: Bot error: {e}")
        import traceback

        logger.error(f"SHUTDOWN: Traceback: {traceback.format_exc()}")
    finally:
        logger.info("SHUTDOWN: Robust Telegram Bot Listener has shut down")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("SHUTDOWN: Program interrupted by user")
    except Exception as e:
        logger.error(f"SHUTDOWN: Program error: {e}")
        import traceback

        logger.error(f"SHUTDOWN: Full traceback: {traceback.format_exc()}")
