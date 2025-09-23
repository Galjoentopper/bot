"""
Integration layer between the trading system and Telegram notifications.
Replaces the old telegram_integration.py with improved architecture.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.logging_manager import get_system_logger

from ..core import MessagePriority
from ..telegram_service import get_telegram_service


class TradeSignificance(Enum):
    """Trade significance levels for notification filtering."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TradingAlert:
    """Structured trading alert data."""

    alert_type: str  # 'trade', 'signal', 'position', 'risk', 'system'
    symbol: str
    action: str
    confidence: float
    price: Optional[float] = None
    quantity: Optional[float] = None
    pnl: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class TradingBotIntegration:
    """
    Advanced integration layer between trading system and Telegram notifications.
    Provides intelligent filtering, formatting, and routing of trading events.
    """

    def __init__(self):
        self.logger = get_system_logger(__name__)
        self.telegram_service = get_telegram_service()

        # Configuration
        self._notification_config = {
            "trade_notifications": True,
            "signal_notifications": True,
            "risk_notifications": True,
            "system_notifications": True,
            "performance_reports": True,
            "min_confidence_threshold": 0.6,
            "min_trade_amount": 100.0,
            "min_pnl_threshold": 10.0,
            "rate_limit_trades": 5,  # max trades per minute to notify
            "rate_limit_signals": 10,  # max signals per minute to notify
        }

        # Rate limiting
        self._notification_history: List[datetime] = []
        self._trade_notification_history: List[datetime] = []
        self._signal_notification_history: List[datetime] = []

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {
            "trade_executed": [],
            "signal_generated": [],
            "position_opened": [],
            "position_closed": [],
            "risk_alert": [],
            "system_event": [],
        }

        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the integration layer.

        Returns:
            bool: True if initialization successful
        """
        try:
            # Check if Telegram service is running (it should be started before this)
            if not self.telegram_service.is_running:
                self.logger.error("Telegram service not running - cannot initialize integration")
                return False

            self.logger.info("Telegram service running, initializing integration...")
            self._initialized = True

            # Send initialization notification
            await self.send_system_notification(
                "🚀 Trading Bot Integration Initialized",
                "Telegram notifications are now active and monitoring trading activities.",
            )

            self.logger.info("Trading bot integration initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize trading integration: {e}")
            return False

    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """
        Send a trade execution notification.

        Args:
            trade_data: Trade execution data

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self._notification_config["trade_notifications"]:
            return False

        try:
            # Apply rate limiting
            if not self._check_trade_rate_limit():
                self.logger.debug("Trade notification rate limited")
                return False

            # Determine significance and filter
            significance = self._assess_trade_significance(trade_data)
            if significance == TradeSignificance.LOW:
                self.logger.debug("Trade not significant enough for notification")
                return False

            # Create trading alert
            alert = TradingAlert(
                alert_type="trade",
                symbol=trade_data.get("symbol", "UNKNOWN"),
                action=trade_data.get("side", "UNKNOWN"),
                confidence=trade_data.get("confidence", 0),
                price=trade_data.get("price"),
                quantity=trade_data.get("quantity"),
                pnl=trade_data.get("realized_pnl"),
                metadata=trade_data,
            )

            # Send notification
            priority = self._get_priority_for_significance(significance)
            formatted_message = self._format_trade_message(alert)

            success = await self.telegram_service.send_notification(
                message=formatted_message, priority=priority
            )

            if success:
                self._record_trade_notification()
                # Trigger event handlers
                await self._trigger_event_handlers("trade_executed", trade_data)

            return success

        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")
            return False

    async def send_signal_notification(self, signal_data: Dict[str, Any]) -> bool:
        """
        Send a trading signal notification.

        Args:
            signal_data: Trading signal data

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self._notification_config["signal_notifications"]:
            return False

        try:
            # Apply rate limiting
            if not self._check_signal_rate_limit():
                self.logger.debug("Signal notification rate limited")
                return False

            confidence = signal_data.get("confidence", 0)

            # Filter by confidence threshold
            if confidence < self._notification_config["min_confidence_threshold"]:
                self.logger.debug(f"Signal confidence {confidence:.1%} below threshold")
                return False

            # Create trading alert
            alert = TradingAlert(
                alert_type="signal",
                symbol=signal_data.get("symbol", "UNKNOWN"),
                action=signal_data.get("action", "HOLD"),
                confidence=confidence,
                price=signal_data.get("current_price"),
                metadata=signal_data,
            )

            # Determine priority based on confidence
            priority = MessagePriority.HIGH if confidence > 0.8 else MessagePriority.NORMAL

            # Send notification
            formatted_message = self._format_signal_message(alert)

            success = await self.telegram_service.send_notification(
                message=formatted_message, priority=priority
            )

            if success:
                self._record_signal_notification()
                # Trigger event handlers
                await self._trigger_event_handlers("signal_generated", signal_data)

            return success

        except Exception as e:
            self.logger.error(f"Error sending signal notification: {e}")
            return False

    async def send_risk_alert(self, risk_data: Dict[str, Any]) -> bool:
        """
        Send a risk management alert.

        Args:
            risk_data: Risk alert data

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self._notification_config["risk_notifications"]:
            return False

        try:
            risk_level = risk_data.get("level", "INFO").upper()

            # Determine priority based on risk level
            priority_map = {
                "INFO": MessagePriority.LOW,
                "WARNING": MessagePriority.NORMAL,
                "HIGH": MessagePriority.HIGH,
                "CRITICAL": MessagePriority.CRITICAL,
            }
            priority = priority_map.get(risk_level, MessagePriority.NORMAL)

            # Format message
            message = self._format_risk_message(risk_data)

            success = await self.telegram_service.send_notification(
                message=message, priority=priority
            )

            if success:
                # Trigger event handlers
                await self._trigger_event_handlers("risk_alert", risk_data)

            return success

        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
            return False

    async def send_position_notification(self, position_data: Dict[str, Any], action: str) -> bool:
        """
        Send position opened/closed notification.

        Args:
            position_data: Position data
            action: 'opened' or 'closed'

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized:
            return False

        try:
            # Create trading alert
            alert = TradingAlert(
                alert_type="position",
                symbol=position_data.get("symbol", "UNKNOWN"),
                action=action,
                confidence=position_data.get("confidence", 0),
                price=position_data.get("price"),
                quantity=position_data.get("size"),
                pnl=position_data.get("unrealized_pnl") if action == "closed" else None,
                metadata=position_data,
            )

            # Format message
            message = self._format_position_message(alert, action)

            # Determine priority
            pnl = position_data.get("realized_pnl", 0) if action == "closed" else 0
            priority = MessagePriority.HIGH if abs(pnl) > 100 else MessagePriority.NORMAL

            success = await self.telegram_service.send_notification(
                message=message, priority=priority
            )

            if success:
                event_type = "position_opened" if action == "opened" else "position_closed"
                await self._trigger_event_handlers(event_type, position_data)

            return success

        except Exception as e:
            self.logger.error(f"Error sending position notification: {e}")
            return False

    async def send_system_notification(
        self, title: str, message: str, alert_type: str = "info"
    ) -> bool:
        """
        Send a system notification.

        Args:
            title: Notification title
            message: Notification message
            alert_type: Type of alert (info, warning, error, success)

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self._notification_config["system_notifications"]:
            return False

        try:
            return await self.telegram_service.send_system_alert(
                alert_type, f"{title}\n\n{message}"
            )

        except Exception as e:
            self.logger.error(f"Error sending system notification: {e}")
            return False

    async def send_performance_report(self, performance_data: Dict[str, Any]) -> bool:
        """
        Send a performance report.

        Args:
            performance_data: Performance metrics

        Returns:
            bool: True if report sent successfully
        """
        if not self._initialized or not self._notification_config["performance_reports"]:
            return False

        try:
            message = self._format_performance_report(performance_data)

            success = await self.telegram_service.send_notification(
                message=message, priority=MessagePriority.NORMAL
            )

            return success

        except Exception as e:
            self.logger.error(f"Error sending performance report: {e}")
            return False

    # Configuration methods

    def update_config(self, config: Dict[str, Any]):
        """Update notification configuration."""
        self._notification_config.update(config)
        self.logger.info(f"Notification configuration updated: {config}")

    def get_config(self) -> Dict[str, Any]:
        """Get current notification configuration."""
        return self._notification_config.copy()

    # Event handler registration

    def register_event_handler(self, event_type: str, handler: Callable):
        """
        Register an event handler for specific trading events.

        Args:
            event_type: Type of event to handle
            handler: Async function to handle the event
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []

        self._event_handlers[event_type].append(handler)
        self.logger.info(f"Registered event handler for {event_type}")

    async def _trigger_event_handlers(self, event_type: str, data: Dict[str, Any]):
        """Trigger all registered handlers for an event type."""
        handlers = self._event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")

    # Private methods

    def _assess_trade_significance(self, trade_data: Dict[str, Any]) -> TradeSignificance:
        """Assess the significance of a trade for notification purposes."""
        confidence = trade_data.get("confidence", 0)
        quantity = abs(trade_data.get("quantity", 0))
        price = trade_data.get("price", 0)
        realized_pnl = abs(trade_data.get("realized_pnl", 0))

        # Calculate trade value
        trade_value = quantity * price

        # Assess significance based on multiple factors
        if confidence > 0.9 or trade_value > 1000 or realized_pnl > 100:
            return TradeSignificance.CRITICAL

        if (
            confidence > 0.8
            or trade_value > self._notification_config["min_trade_amount"]
            or realized_pnl > 50
        ):
            return TradeSignificance.HIGH

        if (
            confidence > self._notification_config["min_confidence_threshold"]
            or trade_value > 50
            or realized_pnl > self._notification_config["min_pnl_threshold"]
        ):
            return TradeSignificance.MEDIUM

        return TradeSignificance.LOW

    def _get_priority_for_significance(self, significance: TradeSignificance) -> MessagePriority:
        """Map trade significance to message priority."""
        mapping = {
            TradeSignificance.CRITICAL: MessagePriority.CRITICAL,
            TradeSignificance.HIGH: MessagePriority.HIGH,
            TradeSignificance.MEDIUM: MessagePriority.NORMAL,
            TradeSignificance.LOW: MessagePriority.LOW,
        }
        return mapping.get(significance, MessagePriority.NORMAL)

    def _check_trade_rate_limit(self) -> bool:
        """Check if trade notifications are within rate limits."""
        return self._check_rate_limit(
            self._trade_notification_history,
            self._notification_config["rate_limit_trades"],
        )

    def _check_signal_rate_limit(self) -> bool:
        """Check if signal notifications are within rate limits."""
        return self._check_rate_limit(
            self._signal_notification_history,
            self._notification_config["rate_limit_signals"],
        )

    def _check_rate_limit(self, history: List[datetime], limit: int) -> bool:
        """Generic rate limit checker."""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(minutes=1)

        # Remove old entries
        history[:] = [t for t in history if t > cutoff_time]

        return len(history) < limit

    def _record_trade_notification(self):
        """Record a trade notification for rate limiting."""
        self._trade_notification_history.append(datetime.now(timezone.utc))

    def _record_signal_notification(self):
        """Record a signal notification for rate limiting."""
        self._signal_notification_history.append(datetime.now(timezone.utc))

    # Message formatting methods

    def _format_trade_message(self, alert: TradingAlert) -> str:
        """Format a trade execution message."""
        side_emoji = "🟢" if alert.action.upper() in ["BUY", "LONG"] else "🔴"
        pnl_emoji = "💰" if (alert.pnl or 0) > 0 else "💸" if (alert.pnl or 0) < 0 else "🔄"

        message = f"""
{side_emoji} <b>Trade Executed</b>

📈 <b>Symbol:</b> {alert.symbol}
🎯 <b>Action:</b> {alert.action.upper()}
💱 <b>Price:</b> ${alert.price:.4f}
📊 <b>Quantity:</b> {abs(alert.quantity):.6f}
📈 <b>Confidence:</b> {alert.confidence:.1%}
"""

        if alert.pnl is not None:
            message += f"{pnl_emoji} <b>P&L:</b> ${alert.pnl:+,.2f}\n"

        message += f"⏰ <b>Time:</b> {alert.timestamp.strftime('%H:%M:%S UTC')}"

        return message

    def _format_signal_message(self, alert: TradingAlert) -> str:
        """Format a trading signal message."""
        if alert.action.upper() == "BUY":
            action_emoji = "🟢"
        elif alert.action.upper() == "SELL":
            action_emoji = "🔴"
        else:
            action_emoji = "🟡"

        confidence_bar = "🟩" * int(alert.confidence * 10) + "⬜" * (10 - int(alert.confidence * 10))

        message = f"""
{action_emoji} <b>Trading Signal</b>

📈 <b>Symbol:</b> {alert.symbol}
🎯 <b>Action:</b> {alert.action.upper()}
💱 <b>Price:</b> ${alert.price:.4f}
📊 <b>Confidence:</b> {alert.confidence:.1%} {confidence_bar}
"""

        # Add additional metadata if available
        metadata = alert.metadata or {}
        if "target_price" in metadata:
            message += f"🎯 <b>Target:</b> ${metadata['target_price']:.4f}\n"
        if "stop_loss" in metadata:
            message += f"🛑 <b>Stop Loss:</b> ${metadata['stop_loss']:.4f}\n"

        message += f"⏰ <b>Time:</b> {alert.timestamp.strftime('%H:%M:%S UTC')}"

        return message

    def _format_position_message(self, alert: TradingAlert, action: str) -> str:
        """Format a position notification message."""
        action_emoji = "🟢" if action == "opened" else "🔴"
        action_text = "Position Opened" if action == "opened" else "Position Closed"

        message = f"""
{action_emoji} <b>{action_text}</b>

📈 <b>Symbol:</b> {alert.symbol}
💱 <b>Price:</b> ${alert.price:.4f}
📊 <b>Size:</b> {abs(alert.quantity):.6f}
"""

        if action == "closed" and alert.pnl is not None:
            pnl_emoji = "💰" if alert.pnl > 0 else "💸"
            message += f"{pnl_emoji} <b>P&L:</b> ${alert.pnl:+,.2f}\n"

        message += f"⏰ <b>Time:</b> {alert.timestamp.strftime('%H:%M:%S UTC')}"

        return message

    def _format_risk_message(self, risk_data: Dict[str, Any]) -> str:
        """Format a risk alert message."""
        level = risk_data.get("level", "INFO").upper()

        level_emojis = {"INFO": "ℹ️", "WARNING": "⚠️", "HIGH": "🟠", "CRITICAL": "🔴"}

        emoji = level_emojis.get(level, "📢")

        message = f"""
{emoji} <b>Risk Alert</b>

<b>Level:</b> {level}
<b>Message:</b> {risk_data.get('message', 'Risk condition detected')}
"""

        # Add specific risk metrics if available
        if "portfolio_risk" in risk_data:
            message += f"📊 <b>Portfolio Risk:</b> {risk_data['portfolio_risk']:.1f}%\n"

        if "drawdown" in risk_data:
            message += f"📉 <b>Drawdown:</b> {risk_data['drawdown']:.1f}%\n"

        message += f"⏰ <b>Time:</b> {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"

        return message

    def _format_performance_report(self, performance_data: Dict[str, Any]) -> str:
        """Format a performance report message."""
        total_pnl = performance_data.get("total_pnl", 0)
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        message = f"""
{pnl_emoji} <b>Performance Report</b>

💰 <b>Total P&L:</b> ${total_pnl:+,.2f}
📊 <b>ROI:</b> {performance_data.get('roi_percent', 0):+.1f}%
🎯 <b>Win Rate:</b> {performance_data.get('win_rate', 0):.1f}%
📈 <b>Trades:</b> {performance_data.get('total_trades', 0)}

<b>Risk Metrics:</b>
📉 Max Drawdown: {performance_data.get('max_drawdown', 0):.1f}%
📊 Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}
"""

        period = performance_data.get("period", "24h")
        message += f"\n⏰ <b>Period:</b> {period} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

        return message

    @property
    def is_initialized(self) -> bool:
        """Check if integration is initialized."""
        return self._initialized

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "initialized": self._initialized,
            "config": self._notification_config,
            "recent_trade_notifications": len(self._trade_notification_history),
            "recent_signal_notifications": len(self._signal_notification_history),
            "event_handlers": {k: len(v) for k, v in self._event_handlers.items()},
        }


# Global integration instance
_trading_integration = None


def get_trading_integration() -> TradingBotIntegration:
    """Get singleton trading integration instance."""
    global _trading_integration
    if _trading_integration is None:
        _trading_integration = TradingBotIntegration()
    return _trading_integration
