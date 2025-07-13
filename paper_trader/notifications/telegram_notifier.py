"""Telegram notifications for paper trading alerts and updates."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import telegram
from telegram.error import TelegramError

class TelegramNotifier:
    """Handles Telegram notifications for paper trading events."""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize bot
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize Telegram bot."""
        try:
            if self.bot_token and self.chat_id:
                self.bot = telegram.Bot(token=self.bot_token)
                self.logger.info("Telegram bot initialized successfully")
            else:
                self.logger.warning("Telegram credentials not provided - notifications disabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            self.bot = None
    
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send a message to Telegram chat."""
        if not self.bot:
            self.logger.debug("Telegram bot not available - skipping notification")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            self.logger.error(f"Telegram error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def send_position_opened(self, symbol: str, entry_price: float, quantity: float,
                                 take_profit: float, stop_loss: float, confidence: float,
                                 signal_strength: str, position_value: float) -> bool:
        """Send notification when a position is opened."""
        message = (
            f"🟢 <b>Position Opened</b>\n\n"
            f"📊 Symbol: <code>{symbol}</code>\n"
            f"💰 Entry Price: <code>{entry_price:.4f}</code>\n"
            f"📦 Quantity: <code>{quantity:.6f}</code>\n"
            f"💵 Position Value: <code>${position_value:.2f}</code>\n\n"
            f"🎯 Take Profit: <code>{take_profit:.4f}</code> (+1.00%)\n"
            f"🛑 Stop Loss: <code>{stop_loss:.4f}</code> (-1.00%)\n\n"
            f"🔮 Confidence: <code>{confidence:.1%}</code>\n"
            f"⚡ Signal: <code>{signal_strength}</code>\n\n"
            f"🕐 Time: <code>{datetime.now().strftime('%H:%M:%S')}</code>"
        )
        
        return await self.send_message(message)
    
    async def send_position_closed(self, symbol: str, entry_price: float, exit_price: float,
                                 quantity: float, pnl: float, pnl_pct: float,
                                 exit_reason: str, hold_time_hours: float) -> bool:
        """Send notification when a position is closed."""
        # Determine emoji based on P&L
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        reason_emoji = {
            'TAKE_PROFIT': '🎯',
            'STOP_LOSS': '🛑',
            'TRAILING_STOP': '📉',
            'TIME_EXIT': '⏰',
            'EMERGENCY_STOP': '🚨',
            'MANUAL': '👤'
        }.get(exit_reason, '❓')
        
        message = (
            f"{pnl_emoji} <b>Position Closed</b>\n\n"
            f"📊 Symbol: <code>{symbol}</code>\n"
            f"📈 Entry: <code>{entry_price:.4f}</code>\n"
            f"📉 Exit: <code>{exit_price:.4f}</code>\n"
            f"📦 Quantity: <code>{quantity:.6f}</code>\n\n"
            f"💰 P&L: <code>${pnl:+.2f}</code> ({pnl_pct:+.2%})\n"
            f"{reason_emoji} Reason: <code>{exit_reason}</code>\n"
            f"⏱️ Hold Time: <code>{hold_time_hours:.1f}h</code>\n\n"
            f"🕐 Time: <code>{datetime.now().strftime('%H:%M:%S')}</code>"
        )
        
        return await self.send_message(message)
    
    async def send_portfolio_update(self, portfolio_summary: Dict) -> bool:
        """Send hourly portfolio update."""
        try:
            # Extract key metrics
            total_capital = portfolio_summary['total_capital']
            total_return = portfolio_summary['total_return']
            total_return_pct = portfolio_summary['total_return_pct']
            realized_pnl = portfolio_summary['realized_pnl']
            unrealized_pnl = portfolio_summary['unrealized_pnl']
            num_positions = portfolio_summary['num_positions']
            win_rate = portfolio_summary['win_rate']
            total_trades = portfolio_summary['total_trades']
            max_drawdown = portfolio_summary['max_drawdown']
            
            # Portfolio emoji based on performance
            portfolio_emoji = "🟢" if total_return >= 0 else "🔴"
            
            message = (
                f"📊 <b>Portfolio Update</b> {portfolio_emoji}\n\n"
                f"💰 Total Capital: <code>${total_capital:.2f}</code>\n"
                f"📈 Total Return: <code>${total_return:+.2f}</code> ({total_return_pct:+.2%})\n\n"
                f"✅ Realized P&L: <code>${realized_pnl:+.2f}</code>\n"
                f"⏳ Unrealized P&L: <code>${unrealized_pnl:+.2f}</code>\n\n"
                f"📍 Open Positions: <code>{num_positions}</code>\n"
                f"📊 Total Trades: <code>{total_trades}</code>\n"
                f"🎯 Win Rate: <code>{win_rate:.1%}</code>\n"
                f"📉 Max Drawdown: <code>{max_drawdown:.2%}</code>\n\n"
            )
            
            # Add position details if any
            if portfolio_summary['positions']:
                message += "<b>Open Positions:</b>\n"
                for pos in portfolio_summary['positions']:
                    pos_emoji = "🟢" if pos['pnl'] >= 0 else "🔴"
                    message += (
                        f"{pos_emoji} <code>{pos['symbol']}</code>: "
                        f"<code>${pos['pnl']:+.2f}</code> ({pos['pnl_pct']:+.2%})\n"
                    )
                message += "\n"
            
            message += f"🕐 Time: <code>{datetime.now().strftime('%H:%M:%S')}</code>"
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio update message: {e}")
            return False
    
    async def send_error_alert(self, error_message: str, component: str = "System") -> bool:
        """Send error alert notification."""
        message = (
            f"🚨 <b>Error Alert</b>\n\n"
            f"🔧 Component: <code>{component}</code>\n"
            f"❌ Error: <code>{error_message}</code>\n\n"
            f"🕐 Time: <code>{datetime.now().strftime('%H:%M:%S')}</code>"
        )
        
        return await self.send_message(message)
    
    async def send_system_status(self, status: str, details: str = "") -> bool:
        """Send system status notification."""
        status_emoji = {
            'STARTED': '🟢',
            'STOPPED': '🔴',
            'PAUSED': '🟡',
            'ERROR': '🚨',
            'WARNING': '⚠️'
        }.get(status.upper(), '📢')
        
        message = (
            f"{status_emoji} <b>System Status: {status}</b>\n\n"
        )
        
        if details:
            message += f"📝 Details: <code>{details}</code>\n\n"
        
        message += f"🕐 Time: <code>{datetime.now().strftime('%H:%M:%S')}</code>"
        
        return await self.send_message(message)
    
    async def send_daily_summary(self, daily_stats: Dict) -> bool:
        """Send daily trading summary."""
        try:
            trades_today = daily_stats.get('trades_today', 0)
            pnl_today = daily_stats.get('pnl_today', 0.0)
            win_rate_today = daily_stats.get('win_rate_today', 0.0)
            best_trade = daily_stats.get('best_trade', {})
            worst_trade = daily_stats.get('worst_trade', {})
            
            summary_emoji = "🟢" if pnl_today >= 0 else "🔴"
            
            message = (
                f"📅 <b>Daily Summary</b> {summary_emoji}\n\n"
                f"📊 Trades Today: <code>{trades_today}</code>\n"
                f"💰 P&L Today: <code>${pnl_today:+.2f}</code>\n"
                f"🎯 Win Rate: <code>{win_rate_today:.1%}</code>\n\n"
            )
            
            if best_trade:
                message += (
                    f"🏆 Best Trade: <code>{best_trade.get('symbol', 'N/A')}</code> "
                    f"<code>${best_trade.get('pnl', 0):+.2f}</code>\n"
                )
            
            if worst_trade:
                message += (
                    f"📉 Worst Trade: <code>{worst_trade.get('symbol', 'N/A')}</code> "
                    f"<code>${worst_trade.get('pnl', 0):+.2f}</code>\n\n"
                )
            
            message += f"🕐 Date: <code>{datetime.now().strftime('%Y-%m-%d')}</code>"
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error creating daily summary message: {e}")
            return False
    
    async def send_market_alert(self, symbol: str, alert_type: str, message_text: str) -> bool:
        """Send market-related alerts."""
        alert_emoji = {
            'VOLATILITY': '⚡',
            'VOLUME': '📊',
            'PRICE': '💰',
            'TECHNICAL': '📈',
            'NEWS': '📰'
        }.get(alert_type.upper(), '📢')
        
        message = (
            f"{alert_emoji} <b>Market Alert</b>\n\n"
            f"📊 Symbol: <code>{symbol}</code>\n"
            f"🔔 Type: <code>{alert_type}</code>\n"
            f"📝 Alert: <code>{message_text}</code>\n\n"
            f"🕐 Time: <code>{datetime.now().strftime('%H:%M:%S')}</code>"
        )
        
        return await self.send_message(message)
    
    def is_enabled(self) -> bool:
        """Check if Telegram notifications are enabled."""
        return self.bot is not None
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        if not self.bot:
            return False
        
        try:
            await self.bot.get_me()
            test_message = (
                f"🤖 <b>Paper Trader Bot Test</b>\n\n"
                f"✅ Connection successful!\n"
                f"🕐 Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
            )
            return await self.send_message(test_message)
        except Exception as e:
            self.logger.error(f"Telegram connection test failed: {e}")
            return False