"""
Unified Telegram Notification System
==================================

A comprehensive, secure, and high-performance Telegram notification system
for the trading bot. Replaces all legacy Telegram implementations.

Features:
- Secure credential management with validation
- Priority-based message queueing with persistence
- Command registry with authentication and rate limiting
- Advanced error handling and retry logic
- Health monitoring and system integration
- Production-ready architecture

Usage:
    from src.notifications import get_telegram_service, MessagePriority

    service = get_telegram_service()
    await service.initialize()
    await service.start()

    # Send notifications
    await service.send_notification("Trading alert!", MessagePriority.HIGH)
    await service.send_trading_alert(alert_data)
    await service.send_system_alert("error", "System issue detected")
"""

from .core import (
    CommandRegistry,
    MessagePriority,
    MessageQueue,
    TelegramClient,
    TelegramCommand,
    get_command_registry,
    get_telegram_client,
    telegram_command,
)
from .telegram_service import TelegramService, get_telegram_service

__all__ = [
    # Main service
    "TelegramService",
    "get_telegram_service",
    # Core components
    "TelegramClient",
    "get_telegram_client",
    "MessageQueue",
    "MessagePriority",
    "CommandRegistry",
    "get_command_registry",
    # Command decorators and classes
    "telegram_command",
    "TelegramCommand",
]

__version__ = "2.0.0"
