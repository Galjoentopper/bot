"""
Telegram System Adapter
Provides integration points for the trading system to connect with the new unified Telegram system.
Replaces old notification patterns with the new architecture.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.core.logging_manager import get_system_logger
from src.notifications import MessagePriority, get_telegram_service
from src.notifications.integrations.trader_integration import get_trading_integration


@dataclass
class NotificationConfig:
    """Configuration for notification behavior."""

    enabled: bool = True
    min_confidence: float = 0.6
    min_trade_amount: float = 100.0
    rate_limit_per_minute: int = 10
    include_charts: bool = False
    detailed_reports: bool = True


class TelegramAdapter:
    """
    Adapter layer for trading system to connect with unified Telegram service.

    This class provides a clean interface for the trading system to send notifications
    without needing to know about the internal Telegram architecture.
    """

    def __init__(self):
        self.logger = get_system_logger(__name__)
        self.telegram_service = get_telegram_service()
        self.trading_integration = get_trading_integration()

        self.config = NotificationConfig()
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the Telegram adapter."""
        try:
            # Initialize trading integration (this will handle the Telegram service initialization)
            if not await self.trading_integration.initialize():
                self.logger.error("Failed to initialize trading integration")
                return False

            self._initialized = True
            self.logger.info("Telegram adapter initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram adapter: {e}")
            return False

    # High-level notification methods for trading system integration

    async def notify_trade_executed(self, trade_data: Dict[str, Any]) -> bool:
        """
        Notify about a trade execution.

        Args:
            trade_data: Dictionary containing trade information:
                - symbol: str
                - side: str (BUY/SELL)
                - quantity: float
                - price: float
                - confidence: float (0-1)
                - realized_pnl: float (optional)
                - timestamp: datetime (optional)

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self.config.enabled:
            return False

        try:
            return await self.trading_integration.send_trade_notification(trade_data)
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")
            return False

    async def notify_trading_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Notify about a new trading signal.

        Args:
            signal_data: Dictionary containing signal information:
                - symbol: str
                - action: str (BUY/SELL/HOLD)
                - confidence: float (0-1)
                - current_price: float
                - target_price: float (optional)
                - stop_loss: float (optional)
                - timestamp: datetime (optional)

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self.config.enabled:
            return False

        try:
            return await self.trading_integration.send_signal_notification(signal_data)
        except Exception as e:
            self.logger.error(f"Error sending signal notification: {e}")
            return False

    async def notify_position_opened(self, position_data: Dict[str, Any]) -> bool:
        """
        Notify about a position being opened.

        Args:
            position_data: Dictionary containing position information:
                - symbol: str
                - side: str (LONG/SHORT)
                - size: float
                - entry_price: float
                - confidence: float (0-1)
                - timestamp: datetime (optional)

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self.config.enabled:
            return False

        try:
            return await self.trading_integration.send_position_notification(
                position_data, "opened"
            )
        except Exception as e:
            self.logger.error(f"Error sending position opened notification: {e}")
            return False

    async def notify_position_closed(self, position_data: Dict[str, Any]) -> bool:
        """
        Notify about a position being closed.

        Args:
            position_data: Dictionary containing position information:
                - symbol: str
                - side: str (LONG/SHORT)
                - size: float
                - exit_price: float
                - realized_pnl: float
                - timestamp: datetime (optional)

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self.config.enabled:
            return False

        try:
            return await self.trading_integration.send_position_notification(
                position_data, "closed"
            )
        except Exception as e:
            self.logger.error(f"Error sending position closed notification: {e}")
            return False

    async def notify_risk_alert(
        self, risk_level: str, message: str, details: Dict[str, Any] = None
    ) -> bool:
        """
        Send a risk management alert.

        Args:
            risk_level: str (INFO, WARNING, HIGH, CRITICAL)
            message: str - Alert message
            details: Optional dict with additional risk metrics

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self.config.enabled:
            return False

        try:
            risk_data = {"level": risk_level, "message": message, **(details or {})}
            return await self.trading_integration.send_risk_alert(risk_data)
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
            return False

    async def notify_system_event(self, event_type: str, title: str, message: str) -> bool:
        """
        Send a system event notification.

        Args:
            event_type: str (info, warning, error, success)
            title: str - Event title
            message: str - Event description

        Returns:
            bool: True if notification sent successfully
        """
        if not self._initialized or not self.config.enabled:
            return False

        try:
            return await self.trading_integration.send_system_notification(
                title, message, event_type
            )
        except Exception as e:
            self.logger.error(f"Error sending system event notification: {e}")
            return False

    async def send_daily_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Send a daily performance report.

        Args:
            report_data: Dictionary containing performance metrics:
                - total_pnl: float
                - roi_percent: float
                - win_rate: float
                - total_trades: int
                - max_drawdown: float
                - sharpe_ratio: float
                - period: str (default: "24h")

        Returns:
            bool: True if report sent successfully
        """
        if not self._initialized or not self.config.enabled:
            return False

        try:
            return await self.trading_integration.send_performance_report(report_data)
        except Exception as e:
            self.logger.error(f"Error sending daily report: {e}")
            return False

    # Configuration methods

    def configure(self, config: Dict[str, Any]):
        """
        Update adapter configuration.

        Args:
            config: Dictionary with configuration options:
                - enabled: bool
                - min_confidence: float
                - min_trade_amount: float
                - rate_limit_per_minute: int
                - include_charts: bool
                - detailed_reports: bool
        """
        try:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"Updated config {key} = {value}")
                else:
                    self.logger.warning(f"Unknown config option: {key}")

            # Update trading integration config
            integration_config = {
                "trade_notifications": self.config.enabled,
                "signal_notifications": self.config.enabled,
                "min_confidence_threshold": self.config.min_confidence,
                "min_trade_amount": self.config.min_trade_amount,
                "rate_limit_trades": self.config.rate_limit_per_minute,
                "rate_limit_signals": self.config.rate_limit_per_minute,
            }

            self.trading_integration.update_config(integration_config)

        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get current adapter configuration."""
        return {
            "enabled": self.config.enabled,
            "min_confidence": self.config.min_confidence,
            "min_trade_amount": self.config.min_trade_amount,
            "rate_limit_per_minute": self.config.rate_limit_per_minute,
            "include_charts": self.config.include_charts,
            "detailed_reports": self.config.detailed_reports,
        }

    # Event handler registration

    def register_trade_handler(self, handler: Callable):
        """Register a handler for trade events."""
        self.trading_integration.register_event_handler("trade_executed", handler)

    def register_signal_handler(self, handler: Callable):
        """Register a handler for signal events."""
        self.trading_integration.register_event_handler("signal_generated", handler)

    def register_position_handler(self, handler: Callable):
        """Register a handler for position events."""
        self.trading_integration.register_event_handler("position_opened", handler)
        self.trading_integration.register_event_handler("position_closed", handler)

    def register_risk_handler(self, handler: Callable):
        """Register a handler for risk events."""
        self.trading_integration.register_event_handler("risk_alert", handler)

    def register_system_handler(self, handler: Callable):
        """Register a handler for system events."""
        self.trading_integration.register_event_handler("system_event", handler)

    # Status and monitoring

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status and statistics."""
        return {
            "initialized": self._initialized,
            "config": self.get_config(),
            "telegram_service_running": (
                self.telegram_service._running
                if hasattr(self.telegram_service, "_running")
                else False
            ),
            "integration_statistics": (
                self.trading_integration.get_statistics() if self._initialized else {}
            ),
        }

    @property
    def is_initialized(self) -> bool:
        """Check if adapter is initialized."""
        return self._initialized

    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled."""
        return self.config.enabled

    async def test_notification(
        self, message: str = "Test notification from Telegram adapter"
    ) -> bool:
        """
        Send a test notification to verify the system is working.

        Args:
            message: Optional test message

        Returns:
            bool: True if test notification sent successfully
        """
        if not self._initialized:
            self.logger.error("Adapter not initialized")
            return False

        try:
            return await self.telegram_service.send_notification(
                message=f"🧪 <b>Test Notification</b>\n\n{message}\n\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                priority=MessagePriority.LOW,
            )
        except Exception as e:
            self.logger.error(f"Error sending test notification: {e}")
            return False

    async def shutdown(self):
        """Shutdown the adapter and cleanup resources."""
        try:
            if self._initialized:
                # The trading integration will handle Telegram service shutdown
                self._initialized = False
                self.logger.info("Telegram adapter shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during adapter shutdown: {e}")


# Global adapter instance
_telegram_adapter = None


def get_telegram_adapter() -> TelegramAdapter:
    """Get singleton Telegram adapter instance."""
    global _telegram_adapter
    if _telegram_adapter is None:
        _telegram_adapter = TelegramAdapter()
    return _telegram_adapter


# Convenience functions for backward compatibility and easy integration


async def send_trade_alert(
    symbol: str,
    action: str,
    price: float,
    confidence: float,
    quantity: float = None,
    pnl: float = None,
) -> bool:
    """
    Convenience function to send a trade alert.
    Provides backward compatibility with existing trading system code.
    """
    adapter = get_telegram_adapter()

    if not adapter.is_initialized:
        await adapter.initialize()

    trade_data = {
        "symbol": symbol,
        "side": action,
        "price": price,
        "confidence": confidence,
        "quantity": quantity or 0,
        "realized_pnl": pnl,
        "timestamp": datetime.now(timezone.utc),
    }

    return await adapter.notify_trade_executed(trade_data)


async def send_signal_alert(
    symbol: str,
    action: str,
    confidence: float,
    current_price: float,
    target_price: float = None,
    stop_loss: float = None,
) -> bool:
    """
    Convenience function to send a signal alert.
    Provides backward compatibility with existing trading system code.
    """
    adapter = get_telegram_adapter()

    if not adapter.is_initialized:
        await adapter.initialize()

    signal_data = {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "current_price": current_price,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "timestamp": datetime.now(timezone.utc),
    }

    return await adapter.notify_trading_signal(signal_data)


async def send_risk_alert(level: str, message: str) -> bool:
    """
    Convenience function to send a risk alert.
    Provides backward compatibility with existing trading system code.
    """
    adapter = get_telegram_adapter()

    if not adapter.is_initialized:
        await adapter.initialize()

    return await adapter.notify_risk_alert(level, message)


async def send_system_alert(title: str, message: str, alert_type: str = "info") -> bool:
    """
    Convenience function to send a system alert.
    Provides backward compatibility with existing trading system code.
    """
    adapter = get_telegram_adapter()

    if not adapter.is_initialized:
        await adapter.initialize()

    return await adapter.notify_system_event(alert_type, title, message)
