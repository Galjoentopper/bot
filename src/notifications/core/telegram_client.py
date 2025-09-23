"""
Core Telegram API client with secure credential handling.
Single point of truth for all Telegram API interactions.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.error import Forbidden, NetworkError, RetryAfter, TelegramError

from src.core.logging_manager import get_system_logger
from src.security import get_credential_manager


class TelegramClient:
    """
    Secure, high-performance Telegram API client.
    Handles authentication, rate limiting, and error recovery.
    """

    def __init__(self):
        self.logger = get_system_logger(__name__)
        self.credential_manager = get_credential_manager()

        self._bot: Optional[Bot] = None
        self._chat_id: Optional[str] = None
        self._initialized = False
        self._rate_limit_delay = 1.0  # seconds between messages
        self._last_message_time = 0.0

        # Connection health tracking
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._is_healthy = True

    async def initialize(self) -> bool:
        """
        Initialize the Telegram client with secure credentials.

        Returns:
            bool: True if initialization successful
        """
        try:
            # Load credentials securely
            if not self.credential_manager.load_credentials():
                self.logger.error("Failed to load Telegram credentials")
                return False

            telegram_creds = self.credential_manager.telegram_credentials
            if not telegram_creds:
                self.logger.error("No Telegram credentials available")
                return False

            # Initialize bot with credentials
            self._bot = Bot(token=telegram_creds.bot_token)
            self._chat_id = telegram_creds.chat_id

            # Test connection
            await self._test_connection()

            self._initialized = True
            self.logger.info("Telegram client initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram client: {e}")
            return False

    async def _test_connection(self) -> bool:
        """
        Test Telegram API connection and validate bot token.

        Returns:
            bool: True if connection is healthy
        """
        if not self._bot:
            return False

        try:
            # Get bot info to validate token
            bot_info = await self._bot.get_me()
            self.logger.info(f"Connected to Telegram bot: {bot_info.username}")

            # Optionally send a connection test message (disabled by default)
            if os.environ.get("TELEGRAM_CONNECTION_TEST", "0") in ("1", "true", "True"):
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text="🔧 Trading Bot Connection Test - System Online",
                    parse_mode=ParseMode.HTML,
                )

            self._consecutive_failures = 0
            self._is_healthy = True
            return True

        except Forbidden as e:
            self.logger.error(f"Bot access forbidden - check chat permissions: {e}")
            self._is_healthy = False
            return False
        except TelegramError as e:
            self.logger.error(f"Telegram API error during connection test: {e}")
            self._consecutive_failures += 1
            self._is_healthy = self._consecutive_failures < self._max_consecutive_failures
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during connection test: {e}")
            self._consecutive_failures += 1
            self._is_healthy = self._consecutive_failures < self._max_consecutive_failures
            return False

    async def send_message(
        self,
        message: str,
        parse_mode: str = ParseMode.HTML,
        priority: bool = False,
        max_retries: int = 3,
    ) -> bool:
        """
        Send message to Telegram with rate limiting and retry logic.

        Args:
            message: Message text to send
            parse_mode: Telegram parse mode (HTML, Markdown)
            priority: If True, skip rate limiting
            max_retries: Maximum number of retry attempts

        Returns:
            bool: True if message sent successfully
        """
        if not self._initialized or not self._bot or not self._chat_id:
            self.logger.error("Telegram client not initialized")
            return False

        # Rate limiting (unless priority message)
        if not priority:
            await self._enforce_rate_limit()

        # Retry logic
        for attempt in range(max_retries + 1):
            try:
                # Send message
                await self._bot.send_message(
                    chat_id=self._chat_id, text=message, parse_mode=parse_mode
                )

                # Update health status on success
                self._consecutive_failures = 0
                self._is_healthy = True
                self._last_message_time = asyncio.get_event_loop().time()

                self.logger.debug(f"Message sent successfully (attempt {attempt + 1})")
                return True

            except RetryAfter as e:
                # Telegram rate limiting - wait and retry
                wait_time = e.retry_after + 1
                self.logger.warning(f"Rate limited by Telegram, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                continue

            except NetworkError as e:
                # Network issues - exponential backoff
                if attempt < max_retries:
                    wait_time = min(2**attempt, 30)  # Cap at 30 seconds
                    self.logger.warning(f"Network error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Network error after {max_retries} retries: {e}")
                    self._consecutive_failures += 1
                    break

            except Forbidden as e:
                # Bot blocked or permissions revoked - don't retry
                self.logger.error(f"Bot access forbidden - message not sent: {e}")
                self._is_healthy = False
                return False

            except TelegramError as e:
                # Other Telegram API errors
                if attempt < max_retries and "flood control" in str(e).lower():
                    # Flood control - wait longer
                    wait_time = min(10 + (attempt * 5), 60)
                    self.logger.warning(f"Flood control, waiting {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Telegram API error: {e}")
                    self._consecutive_failures += 1
                    break

            except Exception as e:
                # Unexpected errors
                self.logger.error(f"Unexpected error sending message: {e}")
                self._consecutive_failures += 1
                break

        # Update health status on failure
        self._is_healthy = self._consecutive_failures < self._max_consecutive_failures
        return False

    async def _enforce_rate_limit(self):
        """Enforce rate limiting between messages."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_message_time

        if time_since_last < self._rate_limit_delay:
            wait_time = self._rate_limit_delay - time_since_last
            await asyncio.sleep(wait_time)

    async def send_photo(
        self, photo_path: str, caption: str = "", parse_mode: str = ParseMode.HTML
    ) -> bool:
        """
        Send photo to Telegram chat.

        Args:
            photo_path: Path to photo file
            caption: Optional caption text
            parse_mode: Telegram parse mode

        Returns:
            bool: True if photo sent successfully
        """
        if not self._initialized or not self._bot or not self._chat_id:
            self.logger.error("Telegram client not initialized")
            return False

        try:
            with open(photo_path, "rb") as photo_file:
                await self._bot.send_photo(
                    chat_id=self._chat_id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode=parse_mode,
                )

            self.logger.debug(f"Photo sent successfully: {photo_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send photo {photo_path}: {e}")
            return False

    async def send_document(
        self, document_path: str, caption: str = "", parse_mode: str = ParseMode.HTML
    ) -> bool:
        """
        Send document to Telegram chat.

        Args:
            document_path: Path to document file
            caption: Optional caption text
            parse_mode: Telegram parse mode

        Returns:
            bool: True if document sent successfully
        """
        if not self._initialized or not self._bot or not self._chat_id:
            self.logger.error("Telegram client not initialized")
            return False

        try:
            with open(document_path, "rb") as document_file:
                await self._bot.send_document(
                    chat_id=self._chat_id,
                    document=document_file,
                    caption=caption,
                    parse_mode=parse_mode,
                )

            self.logger.debug(f"Document sent successfully: {document_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send document {document_path}: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if client is properly initialized."""
        return self._initialized

    @property
    def is_healthy(self) -> bool:
        """Check if client connection is healthy."""
        return self._is_healthy and self._consecutive_failures < self._max_consecutive_failures

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get detailed health status for monitoring.

        Returns:
            Dict with health metrics
        """
        return {
            "initialized": self._initialized,
            "healthy": self.is_healthy,
            "consecutive_failures": self._consecutive_failures,
            "max_failures": self._max_consecutive_failures,
            "bot_available": self._bot is not None,
            "chat_id_configured": self._chat_id is not None,
            "credential_hash": self.credential_manager.get_credential_hash("telegram_token"),
            "last_message_time": self._last_message_time,
        }

    async def cleanup(self):
        """Clean up resources and close connections."""
        if self._bot:
            # Close any open sessions
            try:
                await self._bot.close()
            except Exception as e:
                self.logger.warning(f"Error during bot cleanup: {e}")

        self._initialized = False
        self._bot = None
        self.logger.info("Telegram client cleaned up")


# Global client instance
_telegram_client = None


def get_telegram_client() -> TelegramClient:
    """Get singleton Telegram client instance."""
    global _telegram_client
    if _telegram_client is None:
        _telegram_client = TelegramClient()
    return _telegram_client
