"""
Unified Telegram Service
Main orchestrator for all Telegram bot functionality.
Replaces all existing Telegram implementations with a single, robust solution.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.core.lock_manager import get_telegram_lock_manager
from src.core.logging_manager import get_system_logger
from src.security import get_credential_manager

from .core import (
    MessagePriority,
    MessageQueue,
    get_command_registry,
    get_telegram_client,
    telegram_command,
)
from .handlers import AdminCommandHandler, SystemCommandHandler, TradingCommandHandler


class TelegramService:
    """
    Unified Telegram service providing notifications and command handling.
    Single point of entry for all Telegram functionality.
    """

    def __init__(self):
        self.logger = get_system_logger(__name__)

        # Lock manager
        self.lock_manager = get_telegram_lock_manager()

        # Core components
        self.client = get_telegram_client()
        self.message_queue = MessageQueue(
            queue_file="logs/telegram_queue.json", max_queue_size=1000, persistence_enabled=True
        )
        self.command_registry = get_command_registry()

        # Telegram application
        self.application: Optional[Application] = None
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._polling_task: Optional[asyncio.Task] = None

        # Command handlers
        self.trading_handler = TradingCommandHandler()
        self.system_handler = SystemCommandHandler()
        self.admin_handler = AdminCommandHandler()

        # Initialize components
        self._setup_signal_handlers()
        self._register_commands()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _register_commands(self):
        """Register all command handlers."""
        try:
            # Register command handlers
            self.trading_handler.register_commands(self.command_registry)
            self.system_handler.register_commands(self.command_registry)
            self.admin_handler.register_commands(self.command_registry)

            # Register built-in commands
            self.command_registry.register_command(
                name="help",
                handler=self._handle_help,
                description="Show available commands",
                admin_only=False,
                rate_limit=5,
            )

            self.command_registry.register_command(
                name="status",
                handler=self._handle_status,
                description="Show system status",
                admin_only=False,
                rate_limit=10,
            )

            self.logger.info("All command handlers registered successfully")

        except Exception as e:
            self.logger.error(f"Failed to register commands: {e}")

    async def initialize(self) -> bool:
        """
        Initialize the Telegram service.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Telegram service...")

            # Ensure single instance via enhanced lock manager
            if not self.lock_manager.acquire_service_lock(timeout=10):
                self.logger.error("Another Telegram service instance is running (lock present)")
                return False

            # Initialize client
            if not await self.client.initialize():
                self.logger.error("Failed to initialize Telegram client")
                return False

            # Load credentials for bot application
            credential_manager = get_credential_manager()
            telegram_creds = credential_manager.telegram_credentials

            if not telegram_creds:
                self.logger.error("No Telegram credentials available")
                return False

            # Create Telegram application
            self.application = Application.builder().token(telegram_creds.bot_token).build()

            # Initialize the application (required for python-telegram-bot v20+)
            await self.application.initialize()

            # Register command handlers with application
            await self._setup_telegram_handlers()

            self.logger.info("Telegram service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram service: {e}")
            # Best-effort release lock on failure
            with suppress(Exception):
                self.lock_manager.release_service_lock()
            return False

    async def _setup_telegram_handlers(self):
        """Setup Telegram application handlers."""
        try:
            # Command handler for all registered commands
            async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                command_name = update.message.text.split()[0][1:]  # Remove '/'
                await self.command_registry.execute_command(command_name, update, context)

            # Register command patterns
            for command_name in self.command_registry._commands.keys():
                self.application.add_handler(CommandHandler(command_name, handle_command))

            # Register aliases
            for alias in self.command_registry._aliases.keys():
                self.application.add_handler(CommandHandler(alias, handle_command))

            # Handle unknown commands
            async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text(
                    "❌ Unknown command. Use /help to see available commands."
                )

            # Catch-all for unhandled commands
            self.application.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

            self.logger.info("Telegram handlers registered")

        except Exception as e:
            self.logger.error(f"Failed to setup Telegram handlers: {e}")

    async def start(self) -> bool:
        """
        Start the Telegram service.

        Returns:
            bool: True if startup successful
        """
        if self._running:
            self.logger.warning("Telegram service is already running")
            return True

        try:
            self.logger.info("Starting Telegram service...")

            # Start message queue worker
            await self.message_queue.start()

            # Mark service as running before spawning background workers so they enter
            # their run loop immediately instead of exiting on the initial `_running` check.
            self._running = True

            # Start message worker task
            self._worker_task = asyncio.create_task(self._message_worker())

            # Start Telegram polling (support multiple PTB versions)
            if self.application:
                # Prefer updater-based polling if available
                updater = getattr(self.application, "updater", None)
                if updater is not None:
                    await self.application.start()
                    await updater.start_polling()
                else:
                    # Fallback: run_polling in background if available
                    if hasattr(self.application, "run_polling"):
                        self._polling_task = asyncio.create_task(self.application.run_polling())
                        self.logger.info("Telegram polling started via run_polling task")
                    else:
                        await self.application.start()
                        self.logger.warning(
                            "No polling method found on Telegram application; commands may be inactive"
                        )

            # Send startup notification
            await self.send_notification(
                "🚀 Trading Bot Telegram Service Started", priority=MessagePriority.HIGH
            )

            self.logger.info("Telegram service started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start Telegram service: {e}")
            await self.shutdown()
            return False

    async def _message_worker(self):
        """Background worker to process message queue."""
        self.logger.info("Message worker started")

        try:
            while self._running:
                try:
                    # Get next message from queue
                    message = await self.message_queue.dequeue()

                    if message:
                        # Send message using client
                        success = await self.client.send_message(
                            message=message.message,
                            parse_mode=message.parse_mode,
                            priority=message.priority <= MessagePriority.HIGH.value,
                        )

                        if success:
                            self.message_queue.mark_message_sent()
                            self.logger.debug("Message sent successfully from queue")
                        else:
                            # Requeue for retry
                            await self.message_queue.requeue_with_retry(message)
                            self.logger.warning("Message send failed, requeued for retry")

                    else:
                        # No messages in queue, wait briefly
                        await asyncio.sleep(0.5)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in message worker: {e}")
                    await asyncio.sleep(1)  # Brief pause to prevent tight error loop

        finally:
            self.logger.info("Message worker stopped")

    async def shutdown(self):
        """Gracefully shutdown the Telegram service."""
        if not self._running:
            # Still attempt to release lock if held
            with suppress(Exception):
                self.lock_manager.release_service_lock()
            return

        self.logger.info("Shutting down Telegram service...")

        try:
            self._running = False

            # Send shutdown notification
            await self.send_notification(
                "🛑 Trading Bot Telegram Service Shutting Down", priority=MessagePriority.HIGH
            )

            # Stop worker task
            if self._worker_task:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass

            # Stop Telegram polling
            if self.application:
                try:
                    updater = getattr(self.application, "updater", None)
                    if updater is not None:
                        await updater.stop()
                    if self._polling_task and not self._polling_task.done():
                        self._polling_task.cancel()
                        try:
                            await self._polling_task
                        except asyncio.CancelledError:
                            pass
                finally:
                    try:
                        await self.application.stop()
                    except Exception as e:
                        self.logger.warning(f"Error stopping Telegram application: {e}")

            # Stop message queue
            await self.message_queue.stop()

            # Cleanup client
            await self.client.cleanup()

            # Cleanup queue
            await self.message_queue.cleanup()

            self.logger.info("Telegram service shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            # Always release instance lock
            with suppress(Exception):
                self.lock_manager.release_service_lock()

    # Public API Methods

    @property
    def is_running(self) -> bool:
        """Read-only running state for compatibility with integrations/shims."""
        return self._running

    async def send_notification(
        self,
        message: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        parse_mode: str = "HTML",
    ) -> bool:
        """
        Send a notification message.

        Args:
            message: Message text to send
            priority: Message priority level
            parse_mode: Telegram parse mode

        Returns:
            bool: True if message was queued successfully
        """
        if not self._running:
            self.logger.warning("Service not running, message not sent")
            return False

        return await self.message_queue.enqueue(
            message=message, priority=priority, parse_mode=parse_mode
        )

    async def send_trading_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Send a trading alert with formatted data.

        Args:
            alert_data: Trading alert information

        Returns:
            bool: True if alert was sent successfully
        """
        try:
            # Format trading alert
            symbol = alert_data.get("symbol", "UNKNOWN")
            action = alert_data.get("action", "UNKNOWN")
            price = alert_data.get("price", 0)
            confidence = alert_data.get("confidence", 0)
            timestamp = alert_data.get("timestamp", datetime.now(timezone.utc))

            message = f"""
🔔 <b>Trading Alert</b>

📈 <b>Symbol:</b> {symbol}
🎯 <b>Action:</b> {action.upper()}
💰 <b>Price:</b> ${price:.4f}
📊 <b>Confidence:</b> {confidence:.1%}
⏰ <b>Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

            # Determine priority based on confidence
            priority = MessagePriority.HIGH if confidence > 0.8 else MessagePriority.NORMAL

            return await self.send_notification(message, priority=priority)

        except Exception as e:
            self.logger.error(f"Failed to send trading alert: {e}")
            return False

    async def send_system_alert(self, alert_type: str, message: str) -> bool:
        """
        Send a system alert with appropriate formatting.

        Args:
            alert_type: Type of alert (error, warning, info)
            message: Alert message

        Returns:
            bool: True if alert was sent successfully
        """
        try:
            icons = {"error": "❌", "warning": "⚠️", "info": "ℹ️", "success": "✅"}

            icon = icons.get(alert_type.lower(), "📢")
            priority = MessagePriority.CRITICAL if alert_type == "error" else MessagePriority.HIGH

            formatted_message = f"""
{icon} <b>System Alert</b>

<b>Type:</b> {alert_type.upper()}
<b>Message:</b> {message}
<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

            return await self.send_notification(formatted_message, priority=priority)

        except Exception as e:
            self.logger.error(f"Failed to send system alert: {e}")
            return False

    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status.

        Returns:
            Dict with service metrics
        """
        queue_status = await self.message_queue.get_queue_status()
        return {
            "service_running": self._running,
            "client_status": self.client.get_health_status(),
            "queue_status": queue_status,
            "command_stats": self.command_registry.get_statistics(),
            "uptime": datetime.now(timezone.utc).isoformat(),
            "worker_task_running": self._worker_task is not None and not self._worker_task.done(),
        }

    # Built-in Command Handlers

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        try:
            chat_id = str(update.effective_chat.id)
            commands = self.command_registry.get_command_list(chat_id)

            public_commands = [c for c in commands if not c["admin_only"]]
            admin_commands = [c for c in commands if c["admin_only"]]

            def fmt_cmd(cmd):
                alias_txt = (
                    f" <i>(aliases: {', '.join('/' + a for a in cmd['aliases'])})</i>"
                    if cmd["aliases"]
                    else ""
                )
                return f"/{cmd['name']} — {cmd['description']}{alias_txt}"

            lines = []
            lines.append("🤖 <b>Trading Bot — Command Reference</b>")
            lines.append("━━━━━━━━━━━━━━━━━")
            if public_commands:
                lines.append("📋 <b>General</b>")
                for c in sorted(public_commands, key=lambda x: x["name"]):
                    lines.append(fmt_cmd(c))
                lines.append("")
            if admin_commands:
                lines.append("🔐 <b>Admin</b>")
                for c in sorted(admin_commands, key=lambda x: x["name"]):
                    lines.append(fmt_cmd(c))
                lines.append("")
            lines.append(
                f"💡 <i>Total commands: {len(commands)} | Admin: {len(admin_commands)} | Public: {len(public_commands)}</i>"
            )

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling help command: {e}")
            await update.message.reply_text("❌ Error retrieving help information")

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            status = await self.get_service_status()
            queue_status = status["queue_status"]

            status_text = f"""
🤖 <b>Telegram Service Status</b>

🟢 <b>Service:</b> {'Running' if status['service_running'] else 'Stopped'}
🔗 <b>Client:</b> {'Connected' if status['client_status']['healthy'] else 'Disconnected'}
📨 <b>Queue:</b> {queue_status['queue_size']} messages pending
📋 <b>Commands:</b> {status['command_stats']['enabled_commands']} enabled
⚡ <b>Worker:</b> {'Active' if status['worker_task_running'] else 'Inactive'}

<b>Queue Statistics:</b>
• Total queued: {queue_status['statistics']['messages_queued']}
• Sent: {queue_status['statistics']['messages_sent']}
• Failed: {queue_status['statistics']['messages_failed']}
• Dead letters: {queue_status['dead_letter_size']}
"""

            await update.message.reply_text(status_text, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling status command: {e}")
            await update.message.reply_text("❌ Error retrieving status information")


# Global service instance
_telegram_service = None


def get_telegram_service() -> TelegramService:
    """Get singleton Telegram service instance."""
    global _telegram_service
    if _telegram_service is None:
        _telegram_service = TelegramService()
    return _telegram_service


async def main():
    """Main entry point for standalone service execution."""
    service = get_telegram_service()

    try:
        # Initialize and start service
        if await service.initialize():
            await service.start()

            # Keep service running
            while service._running:
                await asyncio.sleep(1)
        else:
            print("Failed to initialize Telegram service")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
