"""
Unified command registry for Telegram bot commands.
Provides authentication, validation, and centralized command management.
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

from telegram import Update
from telegram.ext import ContextTypes

from src.core.logging_manager import get_system_logger
from src.security import get_credential_manager


@dataclass
class CommandInfo:
    """Information about a registered command."""

    name: str
    description: str
    handler: Callable
    admin_only: bool
    rate_limit: int  # calls per minute
    enabled: bool


class CommandRegistry:
    """
    Centralized registry for Telegram bot commands with security and rate limiting.
    """

    def __init__(self):
        self.logger = get_system_logger(__name__)

        # Command storage
        self._commands: Dict[str, CommandInfo] = {}
        self._aliases: Dict[str, str] = {}  # alias -> command_name

        # Security
        self._admin_chat_ids: Set[str] = set()
        self._authorized_chat_ids: Set[str] = set()

        # Rate limiting
        self._command_usage: Dict[str, List[datetime]] = {}  # chat_id:command -> [timestamps]
        self._rate_limit_window = timedelta(minutes=1)

        # Load configuration
        self._load_security_config()

    def _load_security_config(self):
        """Load security configuration from environment."""
        try:
            credential_manager = get_credential_manager()
            telegram_creds = credential_manager.telegram_credentials

            if telegram_creds:
                # The chat ID from credentials is automatically authorized
                self._authorized_chat_ids.add(telegram_creds.chat_id)
                # For now, treat the main chat as admin (can be configured separately later)
                self._admin_chat_ids.add(telegram_creds.chat_id)

                self.logger.info(
                    f"Loaded security config - {len(self._authorized_chat_ids)} authorized chats"
                )

        except Exception as e:
            self.logger.error(f"Failed to load security config: {e}")

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        admin_only: bool = False,
        rate_limit: int = 10,  # calls per minute
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """
        Register a new command with the registry.

        Args:
            name: Command name (without /)
            handler: Async function to handle the command
            description: Command description for help
            admin_only: Whether command requires admin privileges
            rate_limit: Maximum calls per minute per user
            aliases: Alternative names for the command

        Returns:
            bool: True if registration successful
        """
        try:
            # Validate handler signature
            sig = inspect.signature(handler)
            expected_params = ["update", "context"]
            actual_params = list(sig.parameters.keys())

            if not all(param in actual_params for param in expected_params):
                self.logger.error(
                    f"Invalid handler signature for {name}: expected {expected_params}"
                )
                return False

            # Register main command
            command_info = CommandInfo(
                name=name,
                description=description or f"Execute {name} command",
                handler=handler,
                admin_only=admin_only,
                rate_limit=rate_limit,
                enabled=True,
            )

            self._commands[name] = command_info

            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name

            self.logger.info(
                f"Registered command: /{name} (admin: {admin_only}, rate_limit: {rate_limit}/min)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to register command {name}: {e}")
            return False

    def unregister_command(self, name: str) -> bool:
        """Remove a command from the registry."""
        try:
            if name in self._commands:
                del self._commands[name]

                # Remove aliases
                aliases_to_remove = [alias for alias, cmd in self._aliases.items() if cmd == name]
                for alias in aliases_to_remove:
                    del self._aliases[alias]

                self.logger.info(f"Unregistered command: /{name}")
                return True
            else:
                self.logger.warning(f"Command /{name} not found for unregistration")
                return False

        except Exception as e:
            self.logger.error(f"Failed to unregister command {name}: {e}")
            return False

    async def execute_command(
        self, command_name: str, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """
        Execute a registered command with security and rate limiting.

        Args:
            command_name: Name of command to execute
            update: Telegram update object
            context: Telegram context object

        Returns:
            bool: True if command executed successfully
        """
        try:
            # Resolve aliases
            actual_command = self._aliases.get(command_name, command_name)

            # Check if command exists
            if actual_command not in self._commands:
                await update.message.reply_text(f"❌ Unknown command: /{command_name}")
                return False

            command_info = self._commands[actual_command]

            # Check if command is enabled
            if not command_info.enabled:
                await update.message.reply_text(f"❌ Command /{command_name} is currently disabled")
                return False

            # Security checks
            chat_id = str(update.effective_chat.id)

            # Check authorization
            if not self._is_authorized(chat_id):
                await update.message.reply_text(
                    "❌ Unauthorized - this bot is restricted to authorized users only"
                )
                self.logger.warning(
                    f"Unauthorized command attempt from chat {chat_id}: /{command_name}"
                )
                return False

            # Check admin privileges
            if command_info.admin_only and not self._is_admin(chat_id):
                await update.message.reply_text("❌ Admin privileges required for this command")
                self.logger.warning(
                    f"Non-admin attempted admin command from chat {chat_id}: /{command_name}"
                )
                return False

            # Rate limiting
            if not self._check_rate_limit(chat_id, actual_command, command_info.rate_limit):
                await update.message.reply_text(
                    f"❌ Rate limit exceeded for /{command_name}. Please wait before trying again."
                )
                return False

            # Execute command
            self.logger.info(f"Executing command /{actual_command} from chat {chat_id}")

            # Record usage for rate limiting
            self._record_command_usage(chat_id, actual_command)

            # Execute the handler
            await command_info.handler(update, context)

            self.logger.debug(f"Command /{actual_command} executed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing command /{command_name}: {e}")
            try:
                await update.message.reply_text(f"❌ Error executing command: {str(e)}")
            except Exception:
                pass  # Ignore errors when sending error message
            return False

    def _is_authorized(self, chat_id: str) -> bool:
        """Check if chat ID is authorized to use the bot."""
        return chat_id in self._authorized_chat_ids

    def _is_admin(self, chat_id: str) -> bool:
        """Check if chat ID has admin privileges."""
        return chat_id in self._admin_chat_ids

    def _check_rate_limit(self, chat_id: str, command: str, rate_limit: int) -> bool:
        """Check if command usage is within rate limits."""
        key = f"{chat_id}:{command}"
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - self._rate_limit_window

        # Get recent usage
        if key not in self._command_usage:
            self._command_usage[key] = []

        recent_usage = [
            timestamp for timestamp in self._command_usage[key] if timestamp > cutoff_time
        ]

        # Update usage list
        self._command_usage[key] = recent_usage

        # Check rate limit
        return len(recent_usage) < rate_limit

    def _record_command_usage(self, chat_id: str, command: str):
        """Record command usage for rate limiting."""
        key = f"{chat_id}:{command}"
        current_time = datetime.now(timezone.utc)

        if key not in self._command_usage:
            self._command_usage[key] = []

        self._command_usage[key].append(current_time)

        # Cleanup old entries (keep only last hour for efficiency)
        cutoff_time = current_time - timedelta(hours=1)
        self._command_usage[key] = [
            timestamp for timestamp in self._command_usage[key] if timestamp > cutoff_time
        ]

    def get_command_list(
        self, chat_id: str = None, admin_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get list of available commands for a user.

        Args:
            chat_id: Chat ID to check permissions for
            admin_only: Include only admin commands

        Returns:
            List of command information
        """
        commands = []
        is_admin = chat_id and self._is_admin(chat_id)

        for name, info in self._commands.items():
            if not info.enabled:
                continue

            # Filter admin commands
            if admin_only and not info.admin_only:
                continue

            # Skip admin commands for non-admin users
            if info.admin_only and not is_admin:
                continue

            # Get aliases for this command
            aliases = [alias for alias, cmd in self._aliases.items() if cmd == name]

            commands.append(
                {
                    "name": name,
                    "description": info.description,
                    "admin_only": info.admin_only,
                    "rate_limit": info.rate_limit,
                    "aliases": aliases,
                }
            )

        return sorted(commands, key=lambda x: x["name"])

    def enable_command(self, name: str) -> bool:
        """Enable a command."""
        if name in self._commands:
            self._commands[name].enabled = True
            self.logger.info(f"Enabled command: /{name}")
            return True
        return False

    def disable_command(self, name: str) -> bool:
        """Disable a command."""
        if name in self._commands:
            self._commands[name].enabled = False
            self.logger.info(f"Disabled command: /{name}")
            return True
        return False

    def add_authorized_chat(self, chat_id: str):
        """Add a chat ID to authorized list."""
        self._authorized_chat_ids.add(str(chat_id))
        self.logger.info(f"Added authorized chat: {chat_id}")

    def remove_authorized_chat(self, chat_id: str):
        """Remove a chat ID from authorized list."""
        self._authorized_chat_ids.discard(str(chat_id))
        self.logger.info(f"Removed authorized chat: {chat_id}")

    def add_admin_chat(self, chat_id: str):
        """Add a chat ID to admin list."""
        self._admin_chat_ids.add(str(chat_id))
        # Ensure admin chats are also authorized
        self._authorized_chat_ids.add(str(chat_id))
        self.logger.info(f"Added admin chat: {chat_id}")

    def remove_admin_chat(self, chat_id: str):
        """Remove a chat ID from admin list."""
        self._admin_chat_ids.discard(str(chat_id))
        self.logger.info(f"Removed admin chat: {chat_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_commands = len(self._commands)
        enabled_commands = len([c for c in self._commands.values() if c.enabled])
        admin_commands = len([c for c in self._commands.values() if c.admin_only])

        return {
            "total_commands": total_commands,
            "enabled_commands": enabled_commands,
            "disabled_commands": total_commands - enabled_commands,
            "admin_commands": admin_commands,
            "public_commands": total_commands - admin_commands,
            "total_aliases": len(self._aliases),
            "authorized_chats": len(self._authorized_chat_ids),
            "admin_chats": len(self._admin_chat_ids),
            "recent_usage_entries": len(self._command_usage),
        }


# Global command registry
_command_registry = None


def get_command_registry() -> CommandRegistry:
    """Get singleton command registry instance."""
    global _command_registry
    if _command_registry is None:
        _command_registry = CommandRegistry()
    return _command_registry


# Decorator for easy command registration
def telegram_command(
    name: str = None,
    description: str = "",
    admin_only: bool = False,
    rate_limit: int = 10,
    aliases: List[str] = None,
):
    """
    Decorator to register a function as a Telegram command.

    Args:
        name: Command name (defaults to function name)
        description: Command description
        admin_only: Require admin privileges
        rate_limit: Rate limit per minute
        aliases: Command aliases
    """

    def decorator(func):
        command_name = name or func.__name__
        registry = get_command_registry()

        # Register the command
        registry.register_command(
            name=command_name,
            handler=func,
            description=description,
            admin_only=admin_only,
            rate_limit=rate_limit,
            aliases=aliases or [],
        )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator


class TelegramCommand:
    """Base class for structured Telegram commands."""

    def __init__(self, name: str, description: str = "", admin_only: bool = False):
        self.name = name
        self.description = description
        self.admin_only = admin_only
        self.registry = get_command_registry()

    async def execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Override this method to implement command logic."""
        raise NotImplementedError("Command must implement execute method")

    def register(self):
        """Register this command with the registry."""
        return self.registry.register_command(
            name=self.name,
            handler=self.execute,
            description=self.description,
            admin_only=self.admin_only,
        )
