#!/usr/bin/env python3
"""
Simple Telegram Bot Listener for Interactive Commands
Robust version that handles event loop conflicts properly.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class SimpleTelegramBot:
    """Simple and robust Telegram bot listener."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.application = None
        self.running = False

    async def start_bot(self):
        """Start the bot with proper event loop handling."""
        try:
            logger.info("Starting Simple Telegram Bot...")

            # Create application
            self.application = Application.builder().token(self.bot_token).build()

            # Add handlers
            self._setup_handlers()

            self.running = True
            logger.info("Bot started successfully!")
            logger.info("Available commands: /status, /start, /stop, /help")

            # Start polling
            await self.application.run_polling(allowed_updates=Update.ALL_TYPES)

        except Exception as e:
            logger.error(f"Bot startup error: {e}")
            raise

    def _setup_handlers(self):
        """Setup command handlers."""
        # Status command
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("stop", self._cmd_stop))
        self.application.add_handler(CommandHandler("help", self._cmd_help))

        # Unknown messages
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_unknown)
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        response = """🤖 <b>Trading System Status</b>

✅ Bot: Running
✅ Commands: Available
✅ System: Ready

<i>Use /help for available commands</i>"""
        await update.message.reply_text(response, parse_mode="HTML")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        response = "🚀 Trading system start command received!"
        await update.message.reply_text(response, parse_mode="HTML")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command."""
        response = "🛑 Trading system stop command received!"
        await update.message.reply_text(response, parse_mode="HTML")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """🤖 <b>Trading Bot Commands</b>

• /status - Get system status
• /start - Start trading system
• /stop - Stop trading system
• /help - Show this help

<i>Bot is running and ready to receive commands!</i>"""
        await update.message.reply_text(help_text, parse_mode="HTML")

    async def _handle_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown messages."""
        response = "🤔 Unknown command. Use /help to see available commands."
        await update.message.reply_text(response, parse_mode="HTML")

    async def stop_bot(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False
        if self.application:
            try:
                await self.application.stop()
                logger.info("Bot stopped successfully")
            except Exception as e:
                logger.warning(f"Bot stop warning: {e}")


def load_config() -> Dict[str, Any]:
    """Load configuration."""
    try:
        import yaml

        config_path = Path("training_config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        else:
            logger.error("Configuration file not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


async def run_bot():
    """Run the bot with proper error handling."""
    logger.info("Initializing Telegram Bot...")

    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return

    # Get Telegram configuration
    telegram_config = config.get("notifications", {}).get("telegram", {})
    bot_token = telegram_config.get("bot_token")
    chat_id = telegram_config.get("chat_id")

    if not bot_token or not chat_id:
        logger.error("Telegram bot_token or chat_id not configured")
        logger.error("Please check training_config.yaml telegram section")
        return

    logger.info("Configuration loaded successfully")

    # Create and run bot
    bot = SimpleTelegramBot(bot_token, chat_id)

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        # Create task to stop bot
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bot.stop_bot())
        except RuntimeError:
            # No running loop
            asyncio.create_task(bot.stop_bot())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.start_bot()
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        if bot.running:
            await bot.stop_bot()


def main():
    """Main entry point with event loop handling."""
    try:
        # Try to get existing event loop (for tmux/screen environments)
        loop = asyncio.get_running_loop()
        logger.info("Using existing event loop...")
        # Create task for bot
        loop.create_task(run_bot())
    except RuntimeError:
        # No existing loop, create new one
        logger.info("Creating new event loop...")
        try:
            asyncio.run(run_bot())
        except KeyboardInterrupt:
            logger.info("Bot shutdown requested")
        except Exception as e:
            logger.error(f"Bot failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
