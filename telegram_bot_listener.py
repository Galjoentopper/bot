#!/usr/bin/env python3
"""
Robust Telegram Bot Listener
Enterprise-grade Telegram bot with comprehensive logging and error handling.
Cross-platform compatible for Windows development and Linux production.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

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
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(log_file), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing logging config
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
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        logger.info("CONFIG: Found configuration in environment variables")
        return {'bot_token': bot_token, 'chat_id': chat_id}
    
    # Try .env file
    env_file = project_root / '.env'
    if env_file.exists():
        logger.info("CONFIG: Checking .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('TELEGRAM_BOT_TOKEN='):
                    bot_token = line.split('=', 1)[1].strip().strip('"\'')
                elif line.startswith('TELEGRAM_CHAT_ID='):
                    chat_id = line.split('=', 1)[1].strip().strip('"\'')
        
        if bot_token and chat_id:
            logger.info("CONFIG: Found configuration in .env file")
            return {'bot_token': bot_token, 'chat_id': chat_id}
    
    # Try YAML config
    try:
        import yaml
        config_file = project_root / 'training_config.yaml'
        if config_file.exists():
            logger.info("CONFIG: Checking training_config.yaml...")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                telegram_config = config.get('notifications', {}).get('telegram', {})
                bot_token = telegram_config.get('bot_token')
                chat_id = telegram_config.get('chat_id')
                
                if bot_token and chat_id:
                    logger.info("CONFIG: Found configuration in training_config.yaml")
                    return {'bot_token': bot_token, 'chat_id': chat_id}
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
    
    await update.message.reply_text(response, parse_mode='HTML')
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
    await update.message.reply_text(response, parse_mode='HTML')
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
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            recent_lines = lines[-10:] if len(lines) > 10 else lines
        
        log_text = "".join(recent_lines)
        response = f"<b>Recent Log Entries:</b>\n<pre>{log_text}</pre>"
        
        # Telegram message limit is 4096 characters
        if len(response) > 4000:
            response = response[:4000] + "...\n[Log truncated]"
        
        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Logs command completed successfully")
        
    except Exception as e:
        logger.error(f"COMMAND: Error in logs command: {e}")
        await update.message.reply_text(f"Error reading logs: {e}")
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - start trading system."""
    logger.info(f"COMMAND: Received /start command from {update.effective_user.first_name}")

    try:
        # Start the trading system using tmux
        result = await asyncio.create_subprocess_shell(
            '/opt/trading_bot/scripts/tmux_manager.sh start',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await result.wait()

        if result.returncode == 0:
            response = "🚀 <b>Trading system started successfully</b>"
        else:
            response = "❌ <b>Failed to start trading system</b>"

        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Start command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in start command: {e}")
        await update.message.reply_text(f"❌ Error starting system: {e}")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stop command - stop trading system."""
    logger.info(f"COMMAND: Received /stop command from {update.effective_user.first_name}")

    try:
        # Stop the trading system using tmux
        result = await asyncio.create_subprocess_shell(
            '/opt/trading_bot/scripts/tmux_manager.sh stop',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await result.wait()

        response = "🛑 <b>Trading system stopped</b>"
        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Stop command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in stop command: {e}")
        await update.message.reply_text(f"❌ Error stopping system: {e}")

async def cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /restart command - restart trading system."""
    logger.info(f"COMMAND: Received /restart command from {update.effective_user.first_name}")

    try:
        # Stop first
        stop_result = await asyncio.create_subprocess_shell(
            '/opt/trading_bot/scripts/tmux_manager.sh stop',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await stop_result.wait()

        await asyncio.sleep(2)  # Wait a bit

        # Start again
        start_result = await asyncio.create_subprocess_shell(
            '/opt/trading_bot/scripts/tmux_manager.sh start',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await start_result.wait()

        if start_result.returncode == 0:
            response = "🔄 <b>Trading system restarted successfully</b>"
        else:
            response = "❌ <b>Failed to restart trading system</b>"

        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Restart command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in restart command: {e}")
        await update.message.reply_text(f"❌ Error restarting system: {e}")

async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /performance command - get performance metrics."""
    logger.info(f"COMMAND: Received /performance command from {update.effective_user.first_name}")

    try:
        # Read performance metrics from JSON file
        perf_file = Path("/opt/trading_bot/logs/performance_metrics.json")
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                metrics = json.load(f)

            response = f"""📊 <b>Performance Metrics</b>

<b>Portfolio Value:</b> €{metrics.get('portfolio_value', 0):,.2f}
<b>Daily P&L:</b> €{metrics.get('daily_pnl', 0):+,.2f}
<b>Total Return:</b> {metrics.get('total_return', 0):.2%}
<b>Sharpe Ratio:</b> {metrics.get('sharpe_ratio', 0):.2f}
<b>Win Rate:</b> {metrics.get('win_rate', 0):.1%}
<b>Active Positions:</b> {metrics.get('active_positions', 0)}

<i>Last updated: {metrics.get('timestamp', 'N/A')}</i>
"""
        else:
            response = "❌ <b>No performance data available</b>"

        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Performance command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in performance command: {e}")
        await update.message.reply_text(f"❌ Error getting performance: {e}")

async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /health command - get system health."""
    logger.info(f"COMMAND: Received /health command from {update.effective_user.first_name}")

    try:
        # Run health check script
        result = await asyncio.create_subprocess_shell(
            '/opt/trading_bot/scripts/health_check.sh',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode == 0:
            health_output = stdout.decode().strip()
            response = f"```bash\n{health_output}\n```"
        else:
            response = f"❌ <b>Health check failed:</b> {stderr.decode().strip()}"

        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Health command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in health command: {e}")
        await update.message.reply_text(f"❌ Error checking health: {e}")

async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /balance command - get current balance."""
    logger.info(f"COMMAND: Received /balance command from {update.effective_user.first_name}")

    try:
        # Read balance data from JSON file
        balance_file = Path("/opt/trading_bot/logs/balance.json")
        if balance_file.exists():
            with open(balance_file, 'r') as f:
                balance_data = json.load(f)

            positions_text = "\n".join([
                f"• {symbol}: €{value:,.2f}"
                for symbol, value in balance_data.get('positions', {}).items()
            ])

            response = f"""💰 <b>Current Balance</b>

<b>Cash Balance:</b> €{balance_data.get('cash_balance', 0):,.2f}
<b>Portfolio Value:</b> €{balance_data.get('portfolio_value', 0):,.2f}
<b>Total Equity:</b> €{balance_data.get('total_equity', 0):,.2f}

<b>Active Positions:</b>
{positions_text}
"""
        else:
            response = "❌ <b>Balance data not available</b>"

        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Balance command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in balance command: {e}")
        await update.message.reply_text(f"❌ Error getting balance: {e}")

async def cmd_recent_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /trades command - get recent trades."""
    logger.info(f"COMMAND: Received /trades command from {update.effective_user.first_name}")

    try:
        # Read recent trades from CSV file
        trades_file = Path("/opt/trading_bot/logs/trades_report.csv")
        if trades_file.exists():
            # Get last 5 trades
            result = await asyncio.create_subprocess_shell(
                'tail -5 /opt/trading_bot/logs/trades_report.csv',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                trades = stdout.decode().strip().split('\n')
                if len(trades) > 1:  # Skip header
                    trade_list = []
                    for trade in trades[1:]:
                        parts = trade.split(',')
                        if len(parts) >= 6:
                            trade_list.append(f"• {parts[1]}: {parts[2]} {parts[3]} @ €{parts[4]}")

                    response = f"""📈 <b>Recent Trades</b>

{"\n".join(trade_list)}
"""
                else:
                    response = "📊 <b>No recent trades found</b>"
            else:
                response = "❌ <b>Failed to read trades</b>"
        else:
            response = "❌ <b>Trades log not found</b>"

        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Trades command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in trades command: {e}")
        await update.message.reply_text(f"❌ Error getting trades: {e}")

async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /config command - get configuration info."""
    logger.info(f"COMMAND: Received /config command from {update.effective_user.first_name}")

    try:
        # Read basic config info from YAML file
        config_file = Path("/opt/trading_bot/training_config.yaml")
        if config_file.exists():
            # Get basic config info without sensitive data
            result = await asyncio.create_subprocess_shell(
                'grep -E "^(symbols|interval|initial_balance|max_position_size):" /opt/trading_bot/training_config.yaml',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                config_info = stdout.decode().strip()
                response = f"""⚙️ <b>Configuration</b>

```yaml
{config_info}
```
"""
            else:
                response = "❌ <b>Failed to read configuration</b>"
        else:
            response = "❌ <b>Configuration file not found</b>"

        await update.message.reply_text(response, parse_mode='HTML')
        logger.info("COMMAND: Config command completed successfully")

    except Exception as e:
        logger.error(f"COMMAND: Error in config command: {e}")
        await update.message.reply_text(f"❌ Error getting config: {e}")


def main():
    """Main function - non-async version to avoid event loop issues."""
    logger.info("="*50)
    logger.info("STARTUP: Starting Robust Telegram Bot Listener")
    logger.info("="*50)
    
    # Load config
    config = load_config()
    if not config:
        logger.error("STARTUP: Failed to load configuration - exiting")
        return
    
    bot_token = config['bot_token']
    chat_id = config['chat_id']
    
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