#!/usr/bin/env python3
"""
Telegram Bot Listener for Interactive Commands
Runs alongside the trading system to handle incoming Telegram commands.
Fixed version that handles existing event loops properly.
"""

import asyncio
import logging
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

from src.notifier.enhanced_telegram import EnhancedTelegramNotifier

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBotListener:
    """Telegram bot listener for handling interactive commands."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.application = None
        self.enhanced_notifier = EnhancedTelegramNotifier(bot_token, chat_id)
        self.running = False

    async def start(self):
        """Start the Telegram bot listener."""
        logger.info("Starting Telegram Bot Listener...")

        # Create application
        self.application = Application.builder().token(self.bot_token).build()

        # Add command handlers
        self._add_command_handlers()

        # Add message handler for unknown commands
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_unknown)
        )

        # Start the bot
        self.running = True
        logger.info("Telegram Bot Listener started successfully")
        logger.info(
            "Available commands: /status, /start, /stop, /restart, /performance, /health, /balance, /trades, /logs, /config"
        )

        # Start polling
        await self.application.run_polling(allowed_updates=Update.ALL_TYPES)

    def _add_command_handlers(self):
        """Add all command handlers."""
        commands = [
            ("status", self._cmd_status),
            ("start", self._cmd_start),
            ("stop", self._cmd_stop),
            ("restart", self._cmd_restart),
            ("performance", self._cmd_performance),
            ("health", self._cmd_health),
            ("balance", self._cmd_balance),
            ("trades", self._cmd_recent_trades),
            ("logs", self._cmd_logs),
            ("config", self._cmd_config),
            ("help", self._cmd_help),
        ]

        for command, handler in commands:
            self.application.add_handler(CommandHandler(command, handler))

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            response = await self.enhanced_notifier._cmd_status([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text("❌ Error getting status", parse_mode="HTML")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        try:
            response = await self.enhanced_notifier._cmd_start([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Start command error: {e}")
            await update.message.reply_text("❌ Error starting system", parse_mode="HTML")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command."""
        try:
            response = await self.enhanced_notifier._cmd_stop([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Stop command error: {e}")
            await update.message.reply_text("❌ Error stopping system", parse_mode="HTML")

    async def _cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /restart command."""
        try:
            response = await self.enhanced_notifier._cmd_restart([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Restart command error: {e}")
            await update.message.reply_text("❌ Error restarting system", parse_mode="HTML")

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command."""
        try:
            response = await self.enhanced_notifier._cmd_performance([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Performance command error: {e}")
            await update.message.reply_text("❌ Error getting performance", parse_mode="HTML")

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /health command."""
        try:
            response = await self.enhanced_notifier._cmd_health([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Health command error: {e}")
            await update.message.reply_text("❌ Error getting health status", parse_mode="HTML")

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        try:
            response = await self.enhanced_notifier._cmd_balance([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Balance command error: {e}")
            await update.message.reply_text("❌ Error getting balance", parse_mode="HTML")

    async def _cmd_recent_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        try:
            response = await self.enhanced_notifier._cmd_recent_trades([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Trades command error: {e}")
            await update.message.reply_text("❌ Error getting recent trades", parse_mode="HTML")

    async def _cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs command."""
        try:
            response = await self.enhanced_notifier._cmd_logs([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Logs command error: {e}")
            await update.message.reply_text("❌ Error getting logs", parse_mode="HTML")

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command."""
        try:
            response = await self.enhanced_notifier._cmd_config([])
            await update.message.reply_text(response, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Config command error: {e}")
            await update.message.reply_text("❌ Error getting configuration", parse_mode="HTML")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """🤖 <b>Trading Bot Commands</b>

<b>System Control:</b>
• /start - Start the trading system
• /stop - Stop the trading system
• /restart - Restart the trading system
• /status - Get system status

<b>Information:</b>
• /performance - Get performance metrics
• /health - Get system health status
• /balance - Get current balance and positions
• /trades - Get recent trades
• /logs - Get recent system logs
• /config - Get configuration info

<b>Help:</b>
• /help - Show this help message

<i>All commands work with the tmux-managed trading system.</i>
"""
        await update.message.reply_text(help_text, parse_mode="HTML")

    async def _handle_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown messages."""
        message_text = update.message.text.lower()

        # Check for common greetings
        greetings = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if any(greeting in message_text for greeting in greetings):
            response = (
                "👋 Hello! I'm your trading bot assistant. Use /help to see available commands."
            )
        else:
            response = "🤔 I didn't understand that command. Use /help to see available commands."

        await update.message.reply_text(response, parse_mode="HTML")

    async def stop(self):
        """Stop the Telegram bot listener."""
        logger.info("Stopping Telegram Bot Listener...")
        self.running = False
        if self.application:
            try:
                await self.application.stop()
                logger.info("Telegram Bot Listener stopped successfully")
            except RuntimeError as e:
                if "This Application is not running" in str(e):
                    logger.info("Application was already stopped")
                else:
                    logger.error(f"Error stopping application: {e}")
            except Exception as e:
                logger.error(f"Unexpected error stopping application: {e}")
        else:
            logger.info("No application to stop")


def load_config() -> Dict[str, Any]:
    """Load configuration for Telegram bot."""
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


async def main():
    """Main function to run the Telegram bot listener."""
    logger.info("Starting Telegram Bot Listener Service")

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
        logger.info(
            "Please configure telegram.bot_token and telegram.chat_id in training_config.yaml"
        )
        return

    # Create and start the bot listener
    bot_listener = TelegramBotListener(bot_token, chat_id)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        # Don't try to create a new task if there's already a running loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bot_listener.stop())
        except RuntimeError:
            # No running loop, we can call stop directly
            asyncio.run(bot_listener.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot_listener.start()
    except KeyboardInterrupt:
        logger.info("Bot listener interrupted by user")
    except Exception as e:
        logger.error(f"Bot listener error: {e}")
    finally:
        # Only try to stop if we have a running application
        if hasattr(bot_listener, "application") and bot_listener.application:
            try:
                await bot_listener.stop()
            except Exception as e:
                logger.error(f"Error during final shutdown: {e}")


if __name__ == "__main__":
    # Handle existing event loops (common in tmux/screen environments)
    try:
        # Try to get existing event loop
        loop = asyncio.get_running_loop()
        logger.info("Detected existing event loop, using it...")
        # If there's already a running loop, we need to handle this differently
        # For now, we'll just run the main function directly in the existing loop
        loop.create_task(main())
    except RuntimeError:
        # No existing loop, we can use asyncio.run()
        logger.info("No existing event loop, starting fresh...")
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to start bot listener: {e}")
        sys.exit(1)
