#!/usr/bin/env python3
"""
Telegram Bot Listener for Interactive Commands
Runs alongside the trading system to handle incoming Telegram commands.
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.notifier.enhanced_telegram import EnhancedTelegramNotifier

# Configure logging with file output
def setup_logging():
    """Setup logging to both file and console."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"telegram_bot_listener_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - writing to {log_file}")
    return logger

logger = setup_logging()

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
        logger.info(f"Bot Token: {self.bot_token[:10]}...")
        logger.info(f"Chat ID: {self.chat_id}")

        try:
            # Create application
            logger.info("Creating Telegram application...")
            self.application = Application.builder().token(self.bot_token).build()

            # Add command handlers
            logger.info("Adding command handlers...")
            self._add_command_handlers()

            # Add message handler for unknown commands
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_unknown))

            # Start the bot
            self.running = True
            logger.info("Telegram Bot Listener started successfully")
            logger.info("Available commands: /status, /start, /stop, /restart, /performance, /health, /balance, /trades, /logs, /config")

            # Start polling
            logger.info("Starting polling for Telegram updates...")
            await self.application.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            raise

    def _add_command_handlers(self):
        """Add all command handlers."""
        commands = [
            ('status', self._cmd_status),
            ('start', self._cmd_start),
            ('stop', self._cmd_stop),
            ('restart', self._cmd_restart),
            ('performance', self._cmd_performance),
            ('health', self._cmd_health),
            ('balance', self._cmd_balance),
            ('trades', self._cmd_recent_trades),
            ('logs', self._cmd_logs),
            ('config', self._cmd_config),
            ('help', self._cmd_help),
        ]

        for command, handler in commands:
            self.application.add_handler(CommandHandler(command, handler))

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        logger.info(f"Received /status command from user {update.effective_user.id}")
        try:
            response = await self.enhanced_notifier._cmd_status(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
            logger.info("Status command completed successfully")
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text("❌ Error getting status", parse_mode='HTML')

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        logger.info(f"Received /start command from user {update.effective_user.id}")
        try:
            response = await self.enhanced_notifier._cmd_start(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
            logger.info("Start command completed successfully")
        except Exception as e:
            logger.error(f"Start command error: {e}")
            await update.message.reply_text("❌ Error starting system", parse_mode='HTML')

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command."""
        try:
            response = await self.enhanced_notifier._cmd_stop(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Stop command error: {e}")
            await update.message.reply_text("❌ Error stopping system", parse_mode='HTML')

    async def _cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /restart command."""
        try:
            response = await self.enhanced_notifier._cmd_restart(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Restart command error: {e}")
            await update.message.reply_text("❌ Error restarting system", parse_mode='HTML')

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command."""
        try:
            response = await self.enhanced_notifier._cmd_performance(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Performance command error: {e}")
            await update.message.reply_text("❌ Error getting performance", parse_mode='HTML')

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /health command."""
        try:
            response = await self.enhanced_notifier._cmd_health(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Health command error: {e}")
            await update.message.reply_text("❌ Error getting health status", parse_mode='HTML')

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        try:
            response = await self.enhanced_notifier._cmd_balance(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Balance command error: {e}")
            await update.message.reply_text("❌ Error getting balance", parse_mode='HTML')

    async def _cmd_recent_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        try:
            response = await self.enhanced_notifier._cmd_recent_trades(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Trades command error: {e}")
            await update.message.reply_text("❌ Error getting recent trades", parse_mode='HTML')

    async def _cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs command."""
        try:
            response = await self.enhanced_notifier._cmd_logs(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Logs command error: {e}")
            await update.message.reply_text("❌ Error getting logs", parse_mode='HTML')

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command."""
        try:
            response = await self.enhanced_notifier._cmd_config(context.args or [])
            await update.message.reply_text(response, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Config command error: {e}")
            await update.message.reply_text("❌ Error getting configuration", parse_mode='HTML')

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
        await update.message.reply_text(help_text, parse_mode='HTML')

    async def _handle_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown messages."""
        message_text = update.message.text.lower()

        # Check for common greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in message_text for greeting in greetings):
            response = "👋 Hello! I'm your trading bot assistant. Use /help to see available commands."
        else:
            response = "🤔 I didn't understand that command. Use /help to see available commands."

        await update.message.reply_text(response, parse_mode='HTML')

    async def stop(self):
        """Stop the Telegram bot listener."""
        logger.info("Stopping Telegram Bot Listener...")
        self.running = False
        if self.application:
            try:
                await self.application.stop()
                logger.info("Application stopped successfully")
            except RuntimeError as e:
                if "not running" in str(e):
                    logger.info("Application was already stopped")
                else:
                    logger.error(f"Error stopping application: {e}")
            except Exception as e:
                logger.error(f"Unexpected error stopping application: {e}")
        logger.info("Telegram Bot Listener stopped")

def load_config() -> Dict[str, Any]:
    """Load configuration for Telegram bot."""
    logger.info("Loading Telegram bot configuration...")
    
    # Try multiple configuration sources
    config_sources = [
        "training_config.yaml",
        "config.yaml",
        ".env"
    ]
    
    for config_file in config_sources:
        config_path = Path(config_file)
        if config_path.exists():
            logger.info(f"Found configuration file: {config_file}")
            
            if config_file.endswith('.yaml'):
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    logger.info("Successfully loaded YAML configuration")
                    return config
                except Exception as e:
                    logger.error(f"Error loading YAML config {config_file}: {e}")
                    
            elif config_file == '.env':
                try:
                    # Load environment variables from .env file
                    env_config = {}
                    with open(config_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                env_config[key.strip()] = value.strip().strip('"\'')
                    
                    # Convert to expected format
                    if 'TELEGRAM_BOT_TOKEN' in env_config and 'TELEGRAM_CHAT_ID' in env_config:
                        config = {
                            'notifications': {
                                'telegram': {
                                    'bot_token': env_config['TELEGRAM_BOT_TOKEN'],
                                    'chat_id': env_config['TELEGRAM_CHAT_ID']
                                }
                            }
                        }
                        logger.info("Successfully loaded .env configuration")
                        return config
                except Exception as e:
                    logger.error(f"Error loading .env config: {e}")
    
    # Also try environment variables directly
    import os
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        logger.info("Using environment variables for configuration")
        return {
            'notifications': {
                'telegram': {
                    'bot_token': bot_token,
                    'chat_id': chat_id
                }
            }
        }
    
    logger.error("No configuration found in any source")
    logger.info("Checked: training_config.yaml, config.yaml, .env, environment variables")
    return {}

async def main():
    """Main function to run the Telegram bot listener."""
    logger.info("=" * 50)
    logger.info("Starting Telegram Bot Listener Service")
    logger.info("=" * 50)

    # Load configuration
    logger.info("Step 1: Loading configuration...")
    config = load_config()
    if not config:
        logger.error("Failed to load configuration - exiting")
        return

    # Get Telegram configuration
    logger.info("Step 2: Extracting Telegram configuration...")
    telegram_config = config.get('notifications', {}).get('telegram', {})
    bot_token = telegram_config.get('bot_token')
    chat_id = telegram_config.get('chat_id')

    logger.info(f"Bot token found: {'Yes' if bot_token else 'No'}")
    logger.info(f"Chat ID found: {'Yes' if chat_id else 'No'}")

    if not bot_token or not chat_id:
        logger.error("❌ Telegram bot_token or chat_id not configured")
        logger.info("📝 Please configure telegram in one of these ways:")
        logger.info("   1. In training_config.yaml under notifications.telegram")
        logger.info("   2. In .env file: TELEGRAM_BOT_TOKEN=xxx, TELEGRAM_CHAT_ID=xxx")
        logger.info("   3. As environment variables: export TELEGRAM_BOT_TOKEN=xxx")
        return

    # Create and start the bot listener
    logger.info("Step 3: Creating bot listener...")
    bot_listener = TelegramBotListener(bot_token, chat_id)
    bot_started = False

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        if bot_started:
            asyncio.create_task(bot_listener.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Step 4: Starting bot listener...")
        await bot_listener.start()
        bot_started = True
    except KeyboardInterrupt:
        logger.info("Bot listener interrupted by user")
    except Exception as e:
        logger.error(f"Bot listener error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        if bot_started:
            logger.info("Stopping bot listener...")
            await bot_listener.stop()
        else:
            logger.info("Bot was never started, no need to stop")

if __name__ == "__main__":
    asyncio.run(main())