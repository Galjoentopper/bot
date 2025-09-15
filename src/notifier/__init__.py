"""
Legacy Notifier Compatibility Layer
===================================

This package provides a thin compatibility layer for the old
`src.notifier` imports. The production system has migrated to the
unified Telegram notification architecture under `src.notifications`.

For backward compatibility, we expose `TelegramNotifier` and
`NotificationManager` that delegate to the new unified system.
"""

from .telegram import NotificationManager, TelegramNotifier  # noqa: F401

# For code paths that still import EnhancedTelegramNotifier, alias to
# TelegramNotifier for graceful compatibility.
EnhancedTelegramNotifier = TelegramNotifier  # noqa: E305,F401

__all__ = [
    "TelegramNotifier",
    "NotificationManager",
    "EnhancedTelegramNotifier",
]
