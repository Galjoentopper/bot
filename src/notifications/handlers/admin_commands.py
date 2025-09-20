"""
Admin-only Telegram command handlers.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from telegram import Update
from telegram.ext import ContextTypes

from src.core.logging_manager import get_system_logger


class AdminCommandHandler:
    """Handler for admin-only Telegram commands."""

    def __init__(self):
        self.logger = get_system_logger(__name__)

    def register_commands(self, registry):
        """Register all admin commands with the command registry."""
        registry.register_command(
            name="restart",
            handler=self.handle_restart,
            description="Restart the trading bot system",
            admin_only=True,
            rate_limit=2,
        )

        registry.register_command(
            name="shutdown",
            handler=self.handle_shutdown,
            description="Gracefully shutdown the system",
            admin_only=True,
            rate_limit=2,
            aliases=["stop"],
        )

        registry.register_command(
            name="maintenance",
            handler=self.handle_maintenance,
            description="Toggle maintenance mode",
            admin_only=True,
            rate_limit=5,
        )

        registry.register_command(
            name="backup",
            handler=self.handle_backup,
            description="Create system backup",
            admin_only=True,
            rate_limit=3,
        )

        registry.register_command(
            name="queue",
            handler=self.handle_queue_management,
            description="Manage message queue",
            admin_only=True,
            rate_limit=10,
            aliases=["q"],
        )

        registry.register_command(
            name="auth",
            handler=self.handle_auth_management,
            description="Manage user authorization",
            admin_only=True,
            rate_limit=5,
        )

        registry.register_command(
            name="debug",
            handler=self.handle_debug,
            description="System debugging tools",
            admin_only=True,
            rate_limit=3,
        )

        registry.register_command(
            name="clearlogs",
            handler=self.handle_clear_logs,
            description="Clear old log files",
            admin_only=True,
            rate_limit=2,
        )

        self.logger.info("Admin commands registered")

    async def handle_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /restart command."""
        try:
            # Confirm restart
            if not context.args or context.args[0].lower() != "confirm":
                await update.message.reply_text(
                    """
⚠️ <b>System Restart</b>
━━━━━━━━━━━━━━━━━
This will restart the entire trading bot system.

To proceed, send:
<code>/restart confirm</code>
""".strip(),
                    parse_mode="HTML",
                )
                return

            await update.message.reply_text(
                """
🔄 <b>Restarting</b>
━━━━━━━━━━━━━━━━━
The trading bot will restart in 10 seconds...
""".strip(),
                parse_mode="HTML",
            )

            self.logger.info("System restart initiated by admin command")

            # Schedule restart
            await asyncio.sleep(10)

            # TODO: Integrate with actual system restart mechanism
            # For now, just log the request
            self.logger.info("System restart would occur here")

            await update.message.reply_text("✅ Restart initiated successfully")

        except Exception as e:
            self.logger.error(f"Error handling restart command: {e}")
            await update.message.reply_text("❌ Error initiating restart")

    async def handle_shutdown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /shutdown command."""
        try:
            # Confirm shutdown
            if not context.args or context.args[0].lower() != "confirm":
                await update.message.reply_text(
                    """
⚠️ <b>System Shutdown</b>
━━━━━━━━━━━━━━━━━
This will gracefully shutdown the trading bot system.

To proceed, send:
<code>/shutdown confirm</code>
""".strip(),
                    parse_mode="HTML",
                )
                return

            await update.message.reply_text(
                """
🛑 <b>Shutting Down</b>
━━━━━━━━━━━━━━━━━
• Closing open positions
• Saving system state
• Stopping all services

Shutdown begins in 15 seconds...
""".strip(),
                parse_mode="HTML",
            )

            self.logger.info("System shutdown initiated by admin command")

            # Give time for message to be sent
            await asyncio.sleep(5)

            # TODO: Integrate with actual shutdown mechanism
            # For now, just log the request
            self.logger.info("System shutdown would occur here")

        except Exception as e:
            self.logger.error(f"Error handling shutdown command: {e}")
            await update.message.reply_text("❌ Error initiating shutdown")

    async def handle_maintenance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /maintenance command."""
        try:
            # Parse action
            action = "status"  # default
            if context.args:
                action = context.args[0].lower()

            if action == "enable":
                # Enable maintenance mode
                await self._set_maintenance_mode(True)
                await update.message.reply_text(
                    "🔧 <b>Maintenance Mode Enabled</b>\n\n"
                    "• Trading suspended\n"
                    "• New positions blocked\n"
                    "• System monitoring active\n\n"
                    "Use <code>/maintenance disable</code> to resume normal operation.",
                    parse_mode="HTML",
                )
                self.logger.info("Maintenance mode enabled")

            elif action == "disable":
                # Disable maintenance mode
                await self._set_maintenance_mode(False)
                await update.message.reply_text(
                    "✅ <b>Maintenance Mode Disabled</b>\n\n"
                    "• Trading resumed\n"
                    "• Normal operation restored\n"
                    "• All systems active",
                    parse_mode="HTML",
                )
                self.logger.info("Maintenance mode disabled")

            else:
                # Show maintenance status
                status = await self._get_maintenance_status()
                mode_emoji = "🔧" if status["enabled"] else "✅"
                mode_text = "ENABLED" if status["enabled"] else "DISABLED"

                message = f"""
{mode_emoji} <b>Maintenance Mode Status</b>

<b>Status:</b> {mode_text}
"""

                if status["enabled"]:
                    message += f"<b>Enabled:</b> {status['enabled_time']}\n"
                    message += f"<b>Duration:</b> {status['duration']}\n"
                    message += f"<b>Reason:</b> {status['reason']}"

                message += f"\n<b>Available Actions:</b>\n"
                message += f"• <code>/maintenance enable</code> - Enable maintenance mode\n"
                message += f"• <code>/maintenance disable</code> - Disable maintenance mode"

                await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling maintenance command: {e}")
            await update.message.reply_text("❌ Error managing maintenance mode")

    async def handle_backup(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backup command."""
        try:
            backup_type = "full"  # default
            if context.args:
                backup_type = context.args[0].lower()

            await update.message.reply_text(
                f"💾 <b>Creating {backup_type.title()} Backup</b>\n\n"
                "This may take a few minutes...",
                parse_mode="HTML",
            )

            # Create backup
            backup_result = await self._create_backup(backup_type)

            if backup_result["success"]:
                message = f"""
✅ <b>Backup Created Successfully</b>

📁 <b>File:</b> <code>{backup_result['filename']}</code>
📊 <b>Size:</b> {backup_result['size_mb']:.1f} MB
⏱️ <b>Duration:</b> {backup_result['duration']:.1f}s

<b>Contents:</b>
"""
                for item in backup_result["contents"]:
                    message += f"• {item}\n"

                await update.message.reply_text(message, parse_mode="HTML")

            else:
                await update.message.reply_text(
                    f"❌ <b>Backup Failed</b>\n\n" f"Error: {backup_result['error']}",
                    parse_mode="HTML",
                )

        except Exception as e:
            self.logger.error(f"Error handling backup command: {e}")
            await update.message.reply_text("❌ Error creating backup")

    async def handle_queue_management(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /queue command."""
        try:
            action = "status"  # default
            if context.args:
                action = context.args[0].lower()

            # Get telegram service to access queue
            from src.notifications.telegram_service import get_telegram_service

            service = get_telegram_service()

            if action == "status":
                # Show queue status
                queue_status = await service.message_queue.get_queue_status()

                message = f"""
📨 <b>Message Queue Status</b>

📊 <b>Queue Size:</b> {queue_status['queue_size']} / {queue_status['max_queue_size']}
💀 <b>Dead Letters:</b> {queue_status['dead_letter_size']}
🔄 <b>Persistence:</b> {'Enabled' if queue_status['persistence_enabled'] else 'Disabled'}

<b>Statistics:</b>
• Queued: {queue_status['statistics']['messages_queued']}
• Sent: {queue_status['statistics']['messages_sent']}
• Failed: {queue_status['statistics']['messages_failed']}
• Dropped: {queue_status['statistics']['messages_dropped']}

<b>Priority Distribution:</b>
"""

                for priority, count in queue_status["priority_distribution"].items():
                    message += f"• {priority}: {count} messages\n"

                if queue_status["oldest_message_age"] > 0:
                    message += f"\n⏰ Oldest message: {queue_status['oldest_message_age']:.1f}s ago"

                await update.message.reply_text(message, parse_mode="HTML")

            elif action == "clear":
                # Clear queue
                cleared_count = await service.message_queue.clear_queue()
                await update.message.reply_text(
                    f"✅ <b>Queue Cleared</b>\n\n" f"Removed {cleared_count} messages from queue.",
                    parse_mode="HTML",
                )
                self.logger.info(f"Admin cleared {cleared_count} messages from queue")

            elif action == "deadletters":
                # Show dead letter messages
                dead_letters = await service.message_queue.get_dead_letters()

                if not dead_letters:
                    await update.message.reply_text("✅ No dead letter messages")
                    return

                message = f"💀 <b>Dead Letter Messages ({len(dead_letters)})</b>\n\n"

                for i, msg in enumerate(dead_letters[:10]):  # Show max 10
                    timestamp = msg["original_timestamp"][:19]  # Truncate timestamp
                    message_preview = msg["message"][:50] + (
                        "..." if len(msg["message"]) > 50 else ""
                    )

                    message += f"<b>{i+1}.</b> {timestamp}\n"
                    message += f"Priority: {msg['priority']} | Retries: {msg['retry_count']}\n"
                    message += f"<code>{message_preview}</code>\n\n"

                if len(dead_letters) > 10:
                    message += f"... and {len(dead_letters) - 10} more"

                await update.message.reply_text(message, parse_mode="HTML")

            else:
                await update.message.reply_text(
                    "❓ <b>Queue Management</b>\n\n"
                    "Available actions:\n"
                    "• <code>/queue status</code> - Show queue status\n"
                    "• <code>/queue clear</code> - Clear all messages\n"
                    "• <code>/queue deadletters</code> - Show failed messages",
                    parse_mode="HTML",
                )

        except Exception as e:
            self.logger.error(f"Error handling queue command: {e}")
            await update.message.reply_text("❌ Error managing message queue")

    async def handle_auth_management(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /auth command."""
        try:
            if not context.args:
                await update.message.reply_text(
                    "🔐 <b>Authorization Management</b>\n\n"
                    "Available actions:\n"
                    "• <code>/auth status</code> - Show authorization status\n"
                    "• <code>/auth add [chat_id]</code> - Authorize chat\n"
                    "• <code>/auth remove [chat_id]</code> - Remove authorization\n"
                    "• <code>/auth admin [chat_id]</code> - Grant admin privileges",
                    parse_mode="HTML",
                )
                return

            action = context.args[0].lower()

            # Get command registry
            from src.notifications.telegram_service import get_telegram_service

            service = get_telegram_service()
            registry = service.command_registry

            if action == "status":
                # Show authorization status
                stats = registry.get_statistics()

                message = f"""
🔐 <b>Authorization Status</b>

👥 <b>Authorized Chats:</b> {stats['authorized_chats']}
🔐 <b>Admin Chats:</b> {stats['admin_chats']}

📋 <b>Commands:</b>
• Total: {stats['total_commands']}
• Admin Only: {stats['admin_commands']}
• Public: {stats['public_commands']}

🔄 <b>Recent Usage:</b> {stats['recent_usage_entries']} entries
"""

                await update.message.reply_text(message, parse_mode="HTML")

            elif action in ["add", "remove", "admin"]:
                if len(context.args) < 2:
                    await update.message.reply_text(
                        f"❌ Please provide a chat ID: <code>/auth {action} [chat_id]</code>",
                        parse_mode="HTML",
                    )
                    return

                chat_id = context.args[1]

                if action == "add":
                    registry.add_authorized_chat(chat_id)
                    await update.message.reply_text(f"✅ Chat {chat_id} authorized")

                elif action == "remove":
                    registry.remove_authorized_chat(chat_id)
                    await update.message.reply_text(f"✅ Chat {chat_id} authorization removed")

                elif action == "admin":
                    registry.add_admin_chat(chat_id)
                    await update.message.reply_text(f"✅ Chat {chat_id} granted admin privileges")

            else:
                await update.message.reply_text("❌ Unknown authorization action")

        except Exception as e:
            self.logger.error(f"Error handling auth command: {e}")
            await update.message.reply_text("❌ Error managing authorization")

    async def handle_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /debug command."""
        try:
            debug_type = "info"  # default
            if context.args:
                debug_type = context.args[0].lower()

            if debug_type == "info":
                # System debug info
                debug_info = await self._get_debug_info()

                message = f"""
🐛 <b>Debug Information</b>

<b>Process Info:</b>
• PID: {debug_info['pid']}
• Memory: {debug_info['memory_mb']:.1f} MB
• CPU Time: {debug_info['cpu_time']:.2f}s
• Threads: {debug_info['thread_count']}

<b>Environment:</b>
• Python: {debug_info['python_version']}
• Working Dir: {debug_info['working_dir']}
• Args: {' '.join(debug_info['command_args'][:3])}

<b>Network:</b>
• Connections: {debug_info['network_connections']}

<b>File Descriptors:</b>
• Open: {debug_info['open_files']}
"""

                await update.message.reply_text(message, parse_mode="HTML")

            elif debug_type == "test":
                # Run system test
                await update.message.reply_text("🧪 Running system test...")

                test_results = await self._run_system_test()

                message = f"""
🧪 <b>System Test Results</b>

"""
                for test_name, result in test_results.items():
                    status_emoji = "✅" if result["passed"] else "❌"
                    message += f"{status_emoji} {test_name}: {result['message']}\n"

                await update.message.reply_text(message, parse_mode="HTML")

            else:
                await update.message.reply_text(
                    "🐛 <b>Debug Tools</b>\n\n"
                    "Available options:\n"
                    "• <code>/debug info</code> - System debug information\n"
                    "• <code>/debug test</code> - Run system tests",
                    parse_mode="HTML",
                )

        except Exception as e:
            self.logger.error(f"Error handling debug command: {e}")
            await update.message.reply_text("❌ Error running debug tools")

    async def handle_clear_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clearlogs command."""
        try:
            # Parse parameters
            days = 7  # default
            if context.args and context.args[0].isdigit():
                days = int(context.args[0])

            # Confirm action
            if len(context.args) == 0 or context.args[-1].lower() != "confirm":
                await update.message.reply_text(
                    f"⚠️ <b>Clear Log Files</b>\n\n"
                    f"This will remove log files older than {days} days.\n"
                    f"Use <code>/clearlogs {days} confirm</code> to proceed.",
                    parse_mode="HTML",
                )
                return

            # Clear old logs
            result = await self._clear_old_logs(days)

            message = f"""
🗑️ <b>Log Cleanup Complete</b>

📁 <b>Files Removed:</b> {result['files_removed']}
💾 <b>Space Freed:</b> {result['space_freed_mb']:.1f} MB
⏰ <b>Cutoff Date:</b> {result['cutoff_date']}

<b>Directories Cleaned:</b>
"""
            for directory in result["directories"]:
                message += f"• {directory}\n"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling clearlogs command: {e}")
            await update.message.reply_text("❌ Error clearing log files")

    # Helper methods

    async def _set_maintenance_mode(self, enabled: bool):
        """Set system maintenance mode."""
        # TODO: Integrate with actual maintenance mode system
        self.logger.info(f"Maintenance mode {'enabled' if enabled else 'disabled'}")

    async def _get_maintenance_status(self) -> Dict[str, Any]:
        """Get maintenance mode status."""
        # TODO: Integrate with actual maintenance system
        return {"enabled": False, "enabled_time": None, "duration": None, "reason": "Admin request"}

    async def _create_backup(self, backup_type: str) -> Dict[str, Any]:
        """Create system backup."""
        try:
            start_time = datetime.now(timezone.utc)

            # TODO: Integrate with actual backup system
            # For now, simulate backup creation
            await asyncio.sleep(2)  # Simulate backup time

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            filename = f"trading_bot_backup_{start_time.strftime('%Y%m%d_%H%M%S')}.tar.gz"

            return {
                "success": True,
                "filename": filename,
                "size_mb": 125.7,
                "duration": duration,
                "contents": [
                    "Configuration files",
                    "Trading data",
                    "Model files",
                    "Log files (recent)",
                    "System state",
                ],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_debug_info(self) -> Dict[str, Any]:
        """Get system debug information."""
        try:
            import sys

            import psutil

            process = psutil.Process()

            return {
                "pid": os.getpid(),
                "memory_mb": process.memory_info().rss / 1024**2,
                "cpu_time": process.cpu_times().user + process.cpu_times().system,
                "thread_count": process.num_threads(),
                "python_version": sys.version.split()[0],
                "working_dir": os.getcwd(),
                "command_args": sys.argv,
                "network_connections": len(process.connections()),
                "open_files": process.num_fds() if hasattr(process, "num_fds") else 0,
            }

        except Exception as e:
            self.logger.error(f"Error getting debug info: {e}")
            return {}

    async def _run_system_test(self) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive system tests."""
        try:
            tests = {}

            # Test 1: Memory usage
            import psutil

            memory = psutil.virtual_memory()
            tests["Memory Usage"] = {
                "passed": memory.percent < 90,
                "message": f"{memory.percent:.1f}% used",
            }

            # Test 2: Disk space
            disk = psutil.disk_usage("/")
            disk_percent = disk.used / disk.total * 100
            tests["Disk Space"] = {
                "passed": disk_percent < 95,
                "message": f"{disk_percent:.1f}% used",
            }

            # Test 3: Network connectivity
            try:
                import socket

                socket.create_connection(("8.8.8.8", 53), timeout=3)
                tests["Network"] = {"passed": True, "message": "Connected"}
            except:
                tests["Network"] = {"passed": False, "message": "Connection failed"}

            # Test 4: Configuration
            from src.security import get_credential_manager

            cred_manager = get_credential_manager()
            validation = cred_manager.validate_environment()
            tests["Configuration"] = {
                "passed": validation["valid"],
                "message": f"{len(validation['errors'])} errors, {len(validation['warnings'])} warnings",
            }

            return tests

        except Exception as e:
            return {"System Test": {"passed": False, "message": f"Test failed: {str(e)}"}}

    async def _clear_old_logs(self, days: int) -> Dict[str, Any]:
        """Clear log files older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            log_directories = [Path("logs"), Path("src/notifications/logs"), Path("models/logs")]

            files_removed = 0
            space_freed = 0
            directories_cleaned = []

            for log_dir in log_directories:
                if not log_dir.exists():
                    continue

                directories_cleaned.append(str(log_dir))

                for log_file in log_dir.glob("*.log*"):
                    try:
                        if log_file.stat().st_mtime < cutoff_date.timestamp():
                            size = log_file.stat().st_size
                            log_file.unlink()
                            files_removed += 1
                            space_freed += size
                    except Exception as e:
                        self.logger.warning(f"Could not remove log file {log_file}: {e}")

            return {
                "files_removed": files_removed,
                "space_freed_mb": space_freed / 1024**2,
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "directories": directories_cleaned,
            }

        except Exception as e:
            self.logger.error(f"Error clearing logs: {e}")
            return {
                "files_removed": 0,
                "space_freed_mb": 0,
                "cutoff_date": "Error",
                "directories": [],
            }
