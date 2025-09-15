"""
Comprehensive Test Suite for Unified Telegram System
==================================================

Tests all components of the new unified Telegram notification system:
- Security and credential management
- Message queue functionality
- Command registry and authentication
- Client initialization and health
- Integration with trading system
- Error handling and recovery
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from telegram import Bot, Chat, Message, Update, User
from telegram.ext import ContextTypes

from src.adapters.telegram_adapter import (
    TelegramAdapter,
    get_telegram_adapter,
    send_signal_alert,
    send_trade_alert,
)
from src.notifications import TelegramService, get_telegram_service
from src.notifications.core import (
    CommandRegistry,
    MessagePriority,
    MessageQueue,
    TelegramClient,
    get_command_registry,
    get_telegram_client,
    telegram_command,
)
from src.notifications.integrations.trader_integration import (
    TradeSignificance,
    TradingAlert,
    TradingBotIntegration,
)

# Import components to test
from src.security import (
    CredentialManager,
    ExchangeCredentials,
    TelegramCredentials,
    get_credential_manager,
)


class TestCredentialManager:
    """Test the secure credential management system."""

    def setup_method(self):
        """Setup test environment."""
        # Clear any existing singleton
        import src.security.credential_manager

        src.security.credential_manager._credential_manager = None

    def test_telegram_credentials_validation(self):
        """Test Telegram credentials validation."""
        # Valid credentials
        valid_creds = TelegramCredentials(
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz", chat_id="987654321"
        )
        assert valid_creds.bot_token == "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
        assert valid_creds.chat_id == "987654321"

        # Invalid token format
        with pytest.raises(ValueError, match="Invalid Telegram bot token format"):
            TelegramCredentials(bot_token="invalid_token", chat_id="123456789")

        # Invalid chat ID
        with pytest.raises(ValueError, match="Invalid Telegram chat ID"):
            TelegramCredentials(
                bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz", chat_id="invalid_chat_id"
            )

        # Placeholder values
        with pytest.raises(ValueError, match="appears to be placeholder"):
            TelegramCredentials(bot_token="your_telegram_bot_token_here", chat_id="123456789")

    def test_credential_manager_validation(self):
        """Test credential manager environment validation."""
        manager = CredentialManager()

        # Test with missing credentials
        with patch.dict(os.environ, {}, clear=True):
            validation = manager.validate_environment()
            assert not validation["valid"]
            assert len(validation["errors"]) >= 2
            assert "TELEGRAM_BOT_TOKEN" in str(validation["errors"])
            assert "TELEGRAM_CHAT_ID" in str(validation["errors"])

        # Test with placeholder credentials
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "your_telegram_bot_token_here",
                "TELEGRAM_CHAT_ID": "your_telegram_chat_id_here",
            },
        ):
            validation = manager.validate_environment()
            assert not validation["valid"]
            assert any("placeholder" in error for error in validation["errors"])

        # Test with valid credentials
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            validation = manager.validate_environment()
            assert validation["valid"]
            assert len(validation["errors"]) == 0

    def test_credential_loading(self):
        """Test credential loading from environment."""
        manager = CredentialManager()

        # Test successful loading
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            success = manager.load_credentials()
            assert success
            assert manager.telegram_credentials is not None
            assert manager.telegram_credentials.bot_token == "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
            assert manager.telegram_credentials.chat_id == "987654321"

        # Test failed loading
        with patch.dict(os.environ, {}, clear=True):
            success = manager.load_credentials()
            assert not success
            assert manager.telegram_credentials is None

    def test_credential_hash_generation(self):
        """Test credential hash generation for audit purposes."""
        manager = CredentialManager()

        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            manager.load_credentials()

            token_hash = manager.get_credential_hash("telegram_token")
            chat_hash = manager.get_credential_hash("telegram_chat")

            assert token_hash is not None
            assert len(token_hash) == 8  # First 8 chars of SHA256
            assert chat_hash is not None
            assert len(chat_hash) == 8


class TestMessageQueue:
    """Test the priority message queue system."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_file = Path(self.temp_dir) / "test_queue.json"

    async def test_message_queue_basic_operations(self):
        """Test basic queue operations."""
        queue = MessageQueue(
            queue_file=str(self.queue_file), max_queue_size=10, persistence_enabled=True
        )

        # Test enqueue
        success = await queue.enqueue(
            message="Test message", priority=MessagePriority.HIGH, parse_mode="HTML"
        )
        assert success

        # Test dequeue
        message = await queue.dequeue()
        assert message is not None
        assert message.message == "Test message"
        assert message.priority == MessagePriority.HIGH.value
        assert message.parse_mode == "HTML"

        # Test empty queue
        empty_message = await queue.dequeue()
        assert empty_message is None

    async def test_message_queue_priority_ordering(self):
        """Test that messages are dequeued in priority order."""
        queue = MessageQueue(
            queue_file=str(self.queue_file), max_queue_size=10, persistence_enabled=False
        )

        # Enqueue messages with different priorities
        await queue.enqueue("Low priority", MessagePriority.LOW)
        await queue.enqueue("High priority", MessagePriority.HIGH)
        await queue.enqueue("Critical priority", MessagePriority.CRITICAL)
        await queue.enqueue("Normal priority", MessagePriority.NORMAL)

        # Dequeue and check order
        msg1 = await queue.dequeue()
        assert msg1.message == "Critical priority"

        msg2 = await queue.dequeue()
        assert msg2.message == "High priority"

        msg3 = await queue.dequeue()
        assert msg3.message == "Normal priority"

        msg4 = await queue.dequeue()
        assert msg4.message == "Low priority"

    async def test_message_queue_persistence(self):
        """Test message queue persistence across restarts."""
        # Create queue and add messages
        queue1 = MessageQueue(
            queue_file=str(self.queue_file), max_queue_size=10, persistence_enabled=True
        )

        await queue1.enqueue("Persistent message 1", MessagePriority.HIGH)
        await queue1.enqueue("Persistent message 2", MessagePriority.NORMAL)

        # Simulate restart by creating new queue instance
        queue2 = MessageQueue(
            queue_file=str(self.queue_file), max_queue_size=10, persistence_enabled=True
        )

        # Check that messages were loaded
        msg1 = await queue2.dequeue()
        assert msg1.message == "Persistent message 1"

        msg2 = await queue2.dequeue()
        assert msg2.message == "Persistent message 2"

    async def test_message_queue_retry_logic(self):
        """Test message retry functionality."""
        queue = MessageQueue(
            queue_file=str(self.queue_file), max_queue_size=10, persistence_enabled=False
        )

        await queue.enqueue("Test retry", MessagePriority.NORMAL, max_retries=2)

        original_msg = await queue.dequeue()
        assert original_msg.retry_count == 0

        # Test successful retry
        success = await queue.requeue_with_retry(original_msg)
        assert success

        retry_msg = await queue.dequeue()
        assert retry_msg.message == "Test retry"
        assert retry_msg.retry_count == 1

        # Test max retries exceeded
        for _ in range(2):
            await queue.requeue_with_retry(retry_msg)
            retry_msg = await queue.dequeue()

        final_retry = await queue.requeue_with_retry(retry_msg)
        assert not final_retry  # Should move to dead letter queue

        # Check dead letters
        dead_letters = await queue.get_dead_letters()
        assert len(dead_letters) == 1
        assert dead_letters[0]["message"] == "Test retry"

    async def test_message_queue_size_limits(self):
        """Test queue size limiting."""
        queue = MessageQueue(
            queue_file=str(self.queue_file), max_queue_size=3, persistence_enabled=False
        )

        # Fill queue to limit
        for i in range(4):  # Try to add 4 messages to queue with limit of 3
            await queue.enqueue(f"Message {i}", MessagePriority.NORMAL)

        status = await queue.get_queue_status()
        assert status["queue_size"] == 3  # Should not exceed limit
        assert status["statistics"]["messages_dropped"] > 0


class TestCommandRegistry:
    """Test the command registry and authentication system."""

    def setup_method(self):
        """Setup test environment."""
        # Clear singleton
        import src.notifications.core.command_registry

        src.notifications.core.command_registry._command_registry = None

    async def test_command_registration(self):
        """Test command registration."""
        registry = CommandRegistry()

        # Mock handler
        async def test_handler(update, context):
            pass

        # Test registration
        success = registry.register_command(
            name="test",
            handler=test_handler,
            description="Test command",
            admin_only=False,
            rate_limit=10,
        )
        assert success

        # Test duplicate registration
        success = registry.register_command(
            name="test",
            handler=test_handler,
            description="Duplicate test",
            admin_only=False,
            rate_limit=5,
        )
        assert success  # Should overwrite

        # Check command info
        commands = registry.get_command_list()
        assert len(commands) == 1
        assert commands[0]["name"] == "test"

    async def test_command_authentication(self):
        """Test command authentication."""
        registry = CommandRegistry()

        # Add authorized chat
        registry.add_authorized_chat("123456789")
        registry.add_admin_chat("987654321")

        # Create mock handler
        async def admin_handler(update, context):
            await update.message.reply_text("Admin command executed")

        async def public_handler(update, context):
            await update.message.reply_text("Public command executed")

        # Register commands
        registry.register_command("admin", admin_handler, admin_only=True)
        registry.register_command("public", public_handler, admin_only=False)

        # Mock Telegram objects
        mock_user = Mock(spec=User)
        mock_chat = Mock(spec=Chat)
        mock_chat.id = 123456789  # Authorized but not admin

        mock_message = Mock(spec=Message)
        mock_message.reply_text = AsyncMock()

        mock_update = Mock(spec=Update)
        mock_update.effective_chat = mock_chat
        mock_update.message = mock_message

        mock_context = Mock(spec=ContextTypes.DEFAULT_TYPE)

        # Test public command access (authorized user)
        success = await registry.execute_command("public", mock_update, mock_context)
        assert success
        mock_message.reply_text.assert_called_with("Public command executed")

        # Test admin command access (authorized but not admin)
        mock_message.reply_text.reset_mock()
        success = await registry.execute_command("admin", mock_update, mock_context)
        assert not success
        mock_message.reply_text.assert_called_with("❌ Admin privileges required for this command")

        # Test admin command access (admin user)
        mock_chat.id = 987654321  # Admin chat
        mock_message.reply_text.reset_mock()
        success = await registry.execute_command("admin", mock_update, mock_context)
        assert success
        mock_message.reply_text.assert_called_with("Admin command executed")

    async def test_command_rate_limiting(self):
        """Test command rate limiting."""
        registry = CommandRegistry()

        # Add authorized chat
        registry.add_authorized_chat("123456789")

        # Create mock handler
        call_count = 0

        async def rate_limited_handler(update, context):
            nonlocal call_count
            call_count += 1
            await update.message.reply_text(f"Called {call_count}")

        # Register command with rate limit of 2 per minute
        registry.register_command("limited", rate_limited_handler, rate_limit=2)

        # Mock objects
        mock_chat = Mock(spec=Chat)
        mock_chat.id = 123456789

        mock_message = Mock(spec=Message)
        mock_message.reply_text = AsyncMock()

        mock_update = Mock(spec=Update)
        mock_update.effective_chat = mock_chat
        mock_update.message = mock_message

        mock_context = Mock(spec=ContextTypes.DEFAULT_TYPE)

        # First two calls should succeed
        success1 = await registry.execute_command("limited", mock_update, mock_context)
        success2 = await registry.execute_command("limited", mock_update, mock_context)
        assert success1 and success2
        assert call_count == 2

        # Third call should be rate limited
        mock_message.reply_text.reset_mock()
        success3 = await registry.execute_command("limited", mock_update, mock_context)
        assert not success3
        assert "Rate limit exceeded" in str(mock_message.reply_text.call_args)


class TestTelegramClient:
    """Test the core Telegram client."""

    def setup_method(self):
        """Setup test environment."""
        # Clear singleton
        import src.notifications.core.telegram_client

        src.notifications.core.telegram_client._telegram_client = None

    async def test_client_initialization(self):
        """Test client initialization with credentials."""
        client = TelegramClient()

        # Mock credentials
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            # Mock Bot and its methods
            with patch("src.notifications.core.telegram_client.Bot") as mock_bot_class:
                mock_bot = AsyncMock()
                mock_bot.get_me.return_value = Mock(username="test_bot")
                mock_bot.send_message = AsyncMock()
                mock_bot_class.return_value = mock_bot

                success = await client.initialize()
                assert success
                assert client.is_initialized

                # Verify bot was created with correct token
                mock_bot_class.assert_called_once_with(token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz")

    async def test_client_message_sending(self):
        """Test message sending with rate limiting."""
        client = TelegramClient()

        # Initialize client
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            with patch("src.notifications.core.telegram_client.Bot") as mock_bot_class:
                mock_bot = AsyncMock()
                mock_bot.get_me.return_value = Mock(username="test_bot")
                mock_bot.send_message = AsyncMock()
                mock_bot_class.return_value = mock_bot

                await client.initialize()

                # Test successful message send
                success = await client.send_message("Test message")
                assert success
                mock_bot.send_message.assert_called_once()

                # Verify message content and parameters
                call_args = mock_bot.send_message.call_args
                assert call_args[1]["chat_id"] == "987654321"
                assert call_args[1]["text"] == "Test message"

    async def test_client_error_handling(self):
        """Test client error handling and retry logic."""
        from telegram.error import NetworkError, RetryAfter

        client = TelegramClient()

        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            with patch("src.notifications.core.telegram_client.Bot") as mock_bot_class:
                mock_bot = AsyncMock()
                mock_bot.get_me.return_value = Mock(username="test_bot")

                # Test retry after network error
                mock_bot.send_message = AsyncMock(
                    side_effect=[NetworkError("Connection failed"), None]  # Success on retry
                )
                mock_bot_class.return_value = mock_bot

                await client.initialize()

                success = await client.send_message("Test message")
                assert success
                assert mock_bot.send_message.call_count == 2

                # Test rate limiting
                mock_bot.send_message = AsyncMock(side_effect=RetryAfter(5))

                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    success = await client.send_message("Rate limited message")
                    assert not success  # Should fail after retries
                    mock_sleep.assert_called()


class TestTradingIntegration:
    """Test trading system integration."""

    def setup_method(self):
        """Setup test environment."""
        # Clear singletons
        import src.notifications.integrations.trader_integration

        src.notifications.integrations.trader_integration._trading_integration = None

    async def test_trade_significance_assessment(self):
        """Test trade significance assessment logic."""
        integration = TradingBotIntegration()

        # High confidence trade
        trade_data = {
            "symbol": "BTCEUR",
            "side": "BUY",
            "quantity": 0.1,
            "price": 40000.0,
            "confidence": 0.95,
            "realized_pnl": 0,
        }
        significance = integration._assess_trade_significance(trade_data)
        assert significance == TradeSignificance.CRITICAL

        # Medium significance trade
        trade_data = {
            "symbol": "ETHEUR",
            "side": "SELL",
            "quantity": 1.0,
            "price": 2500.0,
            "confidence": 0.7,
            "realized_pnl": 25.0,
        }
        significance = integration._assess_trade_significance(trade_data)
        assert significance == TradeSignificance.MEDIUM

        # Low significance trade
        trade_data = {
            "symbol": "ADAEUR",
            "side": "BUY",
            "quantity": 100.0,
            "price": 0.5,
            "confidence": 0.4,
            "realized_pnl": 2.0,
        }
        significance = integration._assess_trade_significance(trade_data)
        assert significance == TradeSignificance.LOW

    async def test_notification_formatting(self):
        """Test notification message formatting."""
        integration = TradingBotIntegration()

        # Test trade message formatting
        alert = TradingAlert(
            alert_type="trade",
            symbol="BTCEUR",
            action="BUY",
            confidence=0.85,
            price=42000.0,
            quantity=0.125,
            pnl=150.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        message = integration._format_trade_message(alert)
        assert "Trade Executed" in message
        assert "BTCEUR" in message
        assert "BUY" in message
        assert "42000.0000" in message
        assert "85.0%" in message
        assert "+150.00" in message

        # Test signal message formatting
        signal_alert = TradingAlert(
            alert_type="signal",
            symbol="ETHEUR",
            action="SELL",
            confidence=0.78,
            price=2600.0,
            metadata={"target_price": 2500.0, "stop_loss": 2700.0},
        )

        signal_message = integration._format_signal_message(signal_alert)
        assert "Trading Signal" in signal_message
        assert "ETHEUR" in signal_message
        assert "SELL" in signal_message
        assert "78.0%" in signal_message
        assert "2500.0000" in signal_message  # Target price
        assert "2700.0000" in signal_message  # Stop loss

    async def test_rate_limiting(self):
        """Test notification rate limiting."""
        integration = TradingBotIntegration()

        # Configure aggressive rate limiting for testing
        integration._notification_config["rate_limit_trades"] = 2

        # Test that rate limiting works
        assert integration._check_trade_rate_limit()  # First call should pass

        integration._record_trade_notification()
        assert integration._check_trade_rate_limit()  # Second call should pass

        integration._record_trade_notification()
        assert not integration._check_trade_rate_limit()  # Third call should fail


class TestTelegramAdapter:
    """Test the trading system adapter."""

    def setup_method(self):
        """Setup test environment."""
        # Clear singletons
        import src.adapters.telegram_adapter

        src.adapters.telegram_adapter._telegram_adapter = None

    async def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = TelegramAdapter()

        # Mock the integration initialization
        with patch.object(adapter.trading_integration, "initialize", return_value=True):
            success = await adapter.initialize()
            assert success
            assert adapter.is_initialized

    async def test_convenience_functions(self):
        """Test convenience functions for backward compatibility."""
        # Mock environment
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            # Mock the adapter and integration
            with patch("src.adapters.telegram_adapter.get_telegram_adapter") as mock_get_adapter:
                mock_adapter = Mock(spec=TelegramAdapter)
                mock_adapter.is_initialized = True
                mock_adapter.initialize = AsyncMock(return_value=True)
                mock_adapter.notify_trade_executed = AsyncMock(return_value=True)
                mock_adapter.notify_trading_signal = AsyncMock(return_value=True)
                mock_get_adapter.return_value = mock_adapter

                # Test send_trade_alert convenience function
                success = await send_trade_alert(
                    symbol="BTCEUR",
                    action="BUY",
                    price=40000.0,
                    confidence=0.8,
                    quantity=0.1,
                    pnl=50.0,
                )
                assert success
                mock_adapter.notify_trade_executed.assert_called_once()

                # Test send_signal_alert convenience function
                success = await send_signal_alert(
                    symbol="ETHEUR",
                    action="SELL",
                    confidence=0.75,
                    current_price=2500.0,
                    target_price=2400.0,
                )
                assert success
                mock_adapter.notify_trading_signal.assert_called_once()


class TestSystemIntegration:
    """Integration tests for the complete system."""

    async def test_end_to_end_notification_flow(self):
        """Test complete notification flow from adapter to Telegram."""
        # This would be a full integration test that requires actual Telegram credentials
        # For now, we'll mock the entire flow

        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            # Mock all the external dependencies
            with patch("src.notifications.core.telegram_client.Bot") as mock_bot_class:
                mock_bot = AsyncMock()
                mock_bot.get_me.return_value = Mock(username="test_bot")
                mock_bot.send_message = AsyncMock()
                mock_bot_class.return_value = mock_bot

                # Initialize the full system
                adapter = get_telegram_adapter()
                success = await adapter.initialize()
                assert success

                # Send a test notification through the adapter
                success = await adapter.test_notification("Integration test")
                assert success

                # Verify the message reached the Telegram API
                mock_bot.send_message.assert_called()

    async def test_system_health_monitoring(self):
        """Test system health monitoring and status reporting."""
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                "TELEGRAM_CHAT_ID": "987654321",
            },
        ):
            adapter = get_telegram_adapter()

            # Test status reporting
            status = adapter.get_status()
            assert "initialized" in status
            assert "config" in status
            assert "telegram_service_running" in status

    async def test_error_recovery_scenarios(self):
        """Test system behavior under various error conditions."""
        client = TelegramClient()

        # Test initialization with invalid credentials
        with patch.dict(
            os.environ, {"TELEGRAM_BOT_TOKEN": "invalid_token", "TELEGRAM_CHAT_ID": "123"}
        ):
            success = await client.initialize()
            assert not success
            assert not client.is_initialized

        # Test message sending when not initialized
        success = await client.send_message("Test message")
        assert not success

        # Test health status reporting
        health = client.get_health_status()
        assert not health["initialized"]
        assert not health["healthy"]


# Test configuration for pytest
pytest_plugins = ["pytest_asyncio"]

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
