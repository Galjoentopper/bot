"""
Backward-compatible Telegram notifier shim.

Maps legacy `src.notifier.telegram.TelegramNotifier` interface to the
new unified Telegram notification system under `src.notifications`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.lock_manager import get_telegram_lock_manager
from src.core.logging_manager import get_system_logger
from src.notifications import get_telegram_service


@dataclass
class _NotifierConfig:
    enabled: bool = True


class TelegramNotifier:
    """
    Legacy-compatible notifier with minimal surface used by the trader.

    Exposes:
      - enabled: bool
      - send_message_sync(text: str) -> bool
      - async send_message(text: str) -> bool
      - from_config(config) -> TelegramNotifier
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._logger = get_system_logger("legacy_telegram_notifier")
        self._service = get_telegram_service()
        self._lock_manager = get_telegram_lock_manager()
        self._initialized = False

    async def _ensure_started(self) -> bool:
        """Ensure Telegram is usable without spawning a new instance.

        Legacy shim is disabled from starting the service. It will only use an
        already-running unified Telegram service to avoid duplicate instances.
        """
        if not self.enabled:
            return False
        try:
            # Check if telegram service is running across processes using lock manager
            lock_info = self._lock_manager.manager.get_lock_info()
            service_running = (
                lock_info.get("locked", False)
                and lock_info.get("process_running", False)
                and not lock_info.get("is_stale", True)
            )

            if not service_running:
                self._logger.warning(
                    "Telegram legacy shim disabled - unified service not running. "
                    "Please run bin/telegram_bot or system_manager to start the unified service."
                )
                return False

            # Also check local singleton if available
            if hasattr(self._service, "is_running") and self._service.is_running:
                self._logger.debug("Using local telegram service instance")
            else:
                self._logger.debug("Telegram service running in separate process")

            self._initialized = True
            return True
        except Exception as e:
            self._logger.error(f"Telegram service availability error: {e}")
            return False

    async def send_message(self, message: str) -> bool:
        if not await self._ensure_started():
            return False
        try:
            # If local service is running, use it directly
            if hasattr(self._service, "is_running") and self._service.is_running:
                return await self._service.send_notification(message)
            else:
                # Service is running in different process - queue message via message queue
                # The running service will pick it up and send it
                from src.notifications.core import MessagePriority, MessageQueue

                queue = MessageQueue(
                    queue_file="logs/telegram_queue.json",
                    max_queue_size=1000,
                    persistence_enabled=True,
                )

                # Queue the message for the running service to process
                await queue.enqueue(message, priority=MessagePriority.NORMAL)
                self._logger.debug(f"Queued message for telegram service: {message[:50]}...")
                return True
        except Exception as e:
            self._logger.error(f"Error sending Telegram message: {e}")
            return False

    def send_message_sync(self, message: str) -> bool:
        """
        Synchronous convenience used by existing code paths.

        Note: calling this from within a running event loop raises
        RuntimeError("cannot be called from a running event loop"),
        which upstream code handles by switching to a thread loop.
        """
        return asyncio.run(self.send_message(message))

    async def send_alert(self, message: str, priority: str = "high") -> bool:
        """
        Send alert message (legacy compatibility method).
        Maps to send_message with high priority.
        """
        return await self.send_message(message)

    def send_alert_sync(self, message: str, priority: str = "high") -> bool:
        """Synchronous version of send_alert for legacy compatibility."""
        return asyncio.run(self.send_alert(message, priority))

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "TelegramNotifier":
        enabled = True
        try:
            if isinstance(config, dict):
                # Respect typical config structure if present
                notif_cfg = config.get("notifications", {}).get("telegram", {})
                if isinstance(notif_cfg, dict) and "enabled" in notif_cfg:
                    enabled = bool(notif_cfg.get("enabled", True))
        except Exception:
            # Default to enabled; environment/credentials will govern runtime behavior
            enabled = True
        return cls(enabled=enabled)


class NotificationManager:
    """Minimal placeholder for legacy imports. Not used by current code."""

    def __init__(self):
        self._logger = get_system_logger("legacy_notification_manager")

    # Kept for compatibility; does nothing in the unified system.
    def add_notifier(self, name: str, notifier: TelegramNotifier) -> None:
        self._logger.debug(f"Ignoring add_notifier for '{name}' (compat layer)")
