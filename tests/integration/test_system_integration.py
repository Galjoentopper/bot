"""
System Integration Tests
Tests for the complete trading bot system integration.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.lock_manager import ServiceLockManager, cleanup_all_stale_locks
from src.monitoring.service_health_monitor import ServiceHealthMonitor
from src.notifications.telegram_service import TelegramService
from src.notifier.telegram import TelegramNotifier


class TestSystemIntegration:
    """Integration tests for the complete system."""

    def setup_method(self):
        """Setup for each test method."""
        self.test_logs_dir = Path("test_logs")
        self.test_logs_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up any test locks
        cleanup_all_stale_locks()

        # Clean up test log files
        import shutil

        if self.test_logs_dir.exists():
            shutil.rmtree(self.test_logs_dir, ignore_errors=True)

    def test_lock_manager_basic_functionality(self):
        """Test basic lock manager functionality."""
        lock_file = self.test_logs_dir / "test_service.lock"
        manager = ServiceLockManager("test_service", str(self.test_logs_dir))

        # Test lock acquisition
        assert manager.acquire_service_lock(timeout=5)
        assert manager.manager.is_locked()

        # Test lock info
        lock_info = manager.manager.get_lock_info()
        assert lock_info["locked"] is True
        assert "pid" in lock_info

        # Test release
        assert manager.release_service_lock()
        assert not manager.manager.is_locked()

    def test_lock_manager_stale_detection(self):
        """Test stale lock detection and cleanup."""
        lock_file = self.test_logs_dir / "stale_test.lock"

        # Create a fake stale lock
        lock_data = {
            "pid": 99999,  # Non-existent PID
            "timestamp": time.time() - 7200,  # 2 hours ago
            "hostname": "test-host",
        }

        with open(lock_file, "w") as f:
            json.dump(lock_data, f)

        # Manager should detect and clean up stale lock
        manager = ServiceLockManager("stale_test", str(self.test_logs_dir))
        assert not manager.manager.is_locked()  # Should be cleaned up
        assert not lock_file.exists()

    def test_lock_manager_context_manager(self):
        """Test lock manager context manager."""
        manager = ServiceLockManager("context_test", str(self.test_logs_dir))

        with manager.service_lock(timeout=5):
            assert manager.manager.is_locked()

        # Should be released after context
        assert not manager.manager.is_locked()

    def test_telegram_notifier_compatibility(self):
        """Test that TelegramNotifier maintains compatibility."""
        # This should not raise ImportError
        notifier = TelegramNotifier()
        assert notifier is not None

        # Test that it has expected methods
        assert hasattr(notifier, "send_message")
        assert hasattr(notifier, "send_alert")

    @patch("src.security.get_credential_manager")
    def test_telegram_service_initialization(self, mock_cred_manager):
        """Test TelegramService can initialize without errors."""
        # Mock credentials
        mock_cred = MagicMock()
        mock_cred.get_credential.side_effect = lambda key, default=None: {
            "telegram_bot_token": "test_token",
            "telegram_chat_id": "123456789",
        }.get(key, default)
        mock_cred_manager.return_value = mock_cred

        # This should not raise any errors
        with patch("telegram.ext.Application.builder") as mock_builder:
            mock_app = MagicMock()
            mock_builder.return_value.token.return_value.build.return_value = mock_app

            service = TelegramService()
            assert service is not None
            assert service.client is not None

    @pytest.mark.asyncio
    async def test_health_monitor_service_detection(self):
        """Test health monitor can detect services."""
        monitor = ServiceHealthMonitor(check_interval=1)

        # Run one check cycle
        await monitor._check_all_services()

        # Should have status for all configured services
        statuses = monitor.get_service_status()
        assert "telegram" in statuses
        assert "trader" in statuses

        # Each status should have required fields
        for service_name, status in statuses.items():
            assert "running" in status
            assert "last_check" in status
            assert "lock_status" in status

    @pytest.mark.asyncio
    async def test_health_monitor_summary(self):
        """Test health monitor summary generation."""
        monitor = ServiceHealthMonitor(check_interval=1)
        await monitor._check_all_services()

        summary = monitor.get_health_summary()

        # Check summary structure
        assert "overall_health" in summary
        assert "total_services" in summary
        assert "running_services" in summary
        assert "services" in summary
        assert summary["overall_health"] in ["healthy", "degraded", "critical"]

    def test_import_chains(self):
        """Test critical import chains work correctly."""
        # Test notifier import chain
        from src.notifier.telegram import TelegramNotifier

        assert TelegramNotifier is not None

        # Test new telegram service imports
        from src.notifications.telegram_service import TelegramService

        assert TelegramService is not None

        # Test handlers import
        from src.notifications.handlers.admin_commands import AdminCommandHandler

        assert AdminCommandHandler is not None

        # Test core components
        from src.core.lock_manager import ServiceLockManager

        assert ServiceLockManager is not None

    def test_configuration_loading(self):
        """Test configuration files can be loaded."""
        # Test telegram config
        telegram_config_path = Path("config/telegram_config.yaml")
        if telegram_config_path.exists():
            import yaml

            with open(telegram_config_path) as f:
                config = yaml.safe_load(f)
                assert config is not None
                assert "bot" in config or "telegram" in config

    @patch.dict(
        "os.environ",
        {"TELEGRAM_BOT_TOKEN": "test_token", "TELEGRAM_CHAT_ID": "123456789"},
    )
    def test_environment_variables(self):
        """Test environment variable loading."""
        import os

        assert os.getenv("TELEGRAM_BOT_TOKEN") == "test_token"
        assert os.getenv("TELEGRAM_CHAT_ID") == "123456789"

    def test_directory_structure(self):
        """Test that required directories exist or can be created."""
        required_dirs = [
            Path("logs"),
            Path("config"),
            Path("src/notifications"),
            Path("src/core"),
            Path("src/monitoring"),
        ]

        for directory in required_dirs:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
            assert directory.exists()
            assert directory.is_dir()

    @pytest.mark.asyncio
    async def test_service_communication(self):
        """Test services can communicate through the notification system."""
        # Mock a simple message passing test
        with patch("src.notifications.core.get_telegram_client") as mock_client:
            mock_telegram = MagicMock()
            mock_client.return_value = mock_telegram

            # Create notifier and try to send a test message
            notifier = TelegramNotifier()

            # This should not raise errors
            try:
                await notifier.send_message("Test message", priority="low")
                # If we get here without exception, the interface works
                assert True
            except Exception as e:
                # Expected in test environment without real telegram setup
                assert "telegram" in str(e).lower() or "token" in str(e).lower()


class TestDevelopmentTools:
    """Tests for development tools and processes."""

    def test_pre_commit_config_exists(self):
        """Test pre-commit configuration exists and is valid."""
        config_file = Path(".pre-commit-config.yaml")
        assert config_file.exists()

        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert "repos" in config
        assert len(config["repos"]) > 0

        # Check for our custom hooks
        local_repo = None
        for repo in config["repos"]:
            if repo.get("repo") == "local":
                local_repo = repo
                break

        assert local_repo is not None
        hook_ids = [hook["id"] for hook in local_repo["hooks"]]
        assert "python-syntax-check" in hook_ids
        assert "telegram-system-validation" in hook_ids

    def test_dev_tools_script_exists(self):
        """Test development tools script exists and is executable."""
        script_path = Path("scripts/dev_tools.sh")
        assert script_path.exists()
        assert script_path.stat().st_mode & 0o111  # Check if executable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
