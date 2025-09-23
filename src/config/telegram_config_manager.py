"""
Telegram Configuration Manager
Handles loading, validation, and management of Telegram system configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.logging_manager import get_system_logger


@dataclass
class TelegramServiceConfig:
    """Configuration for the Telegram service."""

    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 5
    connection_timeout: int = 30
    health_check_interval: int = 60


@dataclass
class TelegramSecurityConfig:
    """Security configuration for Telegram system."""

    auth_required: bool = True
    admin_commands_enabled: bool = True
    command_rate_limiting: bool = True
    max_unauthorized_attempts: int = 5


@dataclass
class TelegramQueueConfig:
    """Message queue configuration."""

    enabled: bool = True
    persistence: bool = True
    max_size: int = 1000
    persistence_file: str = "logs/telegram_queue.json"
    dead_letter_max_size: int = 100
    cleanup_interval: int = 3600


@dataclass
class TelegramRateLimitConfig:
    """Rate limiting configuration."""

    messages_per_minute: int = 20
    commands_per_user_per_minute: int = 10
    admin_commands_per_minute: int = 20
    burst_messages: int = 5
    burst_window: int = 10


@dataclass
class TelegramNotificationConfig:
    """Notification filtering and behavior configuration."""

    trade_notifications: bool = True
    signal_notifications: bool = True
    position_notifications: bool = True
    risk_notifications: bool = True
    system_notifications: bool = True
    performance_reports: bool = True

    min_trade_confidence: float = 0.6
    min_trade_amount: float = 100.0
    min_pnl_threshold: float = 10.0

    max_trades_per_minute: int = 10
    max_signals_per_minute: int = 15
    max_risk_alerts_per_minute: int = 5


@dataclass
class TelegramFormattingConfig:
    """Message formatting configuration."""

    parse_mode: str = "HTML"
    include_timestamps: bool = True
    use_emojis: bool = True
    truncate_long_messages: bool = True
    max_message_length: int = 4096

    currency_format: str = "${:.4f}"
    percentage_format: str = "{:.1f}%"
    datetime_format: str = "%Y-%m-%d %H:%M:%S UTC"


@dataclass
class TelegramMonitoringConfig:
    """Health monitoring configuration."""

    enabled: bool = True
    health_check_commands: bool = True
    performance_tracking: bool = True
    error_reporting: bool = True

    max_consecutive_failures: int = 5
    max_queue_size_warning: int = 800
    max_memory_usage_warning: int = 85
    max_response_time_warning: int = 5000


@dataclass
class TelegramConfig:
    """Complete Telegram system configuration."""

    service: TelegramServiceConfig = field(default_factory=TelegramServiceConfig)
    security: TelegramSecurityConfig = field(default_factory=TelegramSecurityConfig)
    queue: TelegramQueueConfig = field(default_factory=TelegramQueueConfig)
    rate_limits: TelegramRateLimitConfig = field(default_factory=TelegramRateLimitConfig)
    notifications: TelegramNotificationConfig = field(default_factory=TelegramNotificationConfig)
    formatting: TelegramFormattingConfig = field(default_factory=TelegramFormattingConfig)
    monitoring: TelegramMonitoringConfig = field(default_factory=TelegramMonitoringConfig)

    # Additional settings as dictionaries for flexibility
    commands: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    development: Dict[str, Any] = field(default_factory=dict)
    production: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)


class TelegramConfigManager:
    """
    Manager for Telegram system configuration with validation and hot reloading.
    """

    def __init__(self, config_file: Optional[str] = None):
        self.logger = get_system_logger(__name__)

        # Default config file location
        if config_file is None:
            config_file = Path(__file__).parent.parent.parent / "config" / "telegram_config.yaml"

        self.config_file = Path(config_file)
        self.config: Optional[TelegramConfig] = None
        self.last_modified: Optional[datetime] = None

        # Validation rules
        self.validation_rules = self._setup_validation_rules()

    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup configuration validation rules."""
        return {
            "service.max_retries": {"min": 1, "max": 10, "type": int},
            "service.retry_delay": {"min": 1, "max": 300, "type": int},
            "service.connection_timeout": {"min": 5, "max": 300, "type": int},
            "service.health_check_interval": {"min": 10, "max": 3600, "type": int},
            "queue.max_size": {"min": 10, "max": 10000, "type": int},
            "queue.dead_letter_max_size": {"min": 10, "max": 1000, "type": int},
            "queue.cleanup_interval": {"min": 60, "max": 86400, "type": int},
            "rate_limits.messages_per_minute": {"min": 1, "max": 100, "type": int},
            "rate_limits.commands_per_user_per_minute": {
                "min": 1,
                "max": 50,
                "type": int,
            },
            "rate_limits.burst_messages": {"min": 1, "max": 20, "type": int},
            "rate_limits.burst_window": {"min": 1, "max": 60, "type": int},
            "notifications.min_trade_confidence": {
                "min": 0.0,
                "max": 1.0,
                "type": float,
            },
            "notifications.min_trade_amount": {
                "min": 0.0,
                "max": 1000000.0,
                "type": float,
            },
            "notifications.min_pnl_threshold": {
                "min": 0.0,
                "max": 10000.0,
                "type": float,
            },
            "formatting.max_message_length": {"min": 100, "max": 4096, "type": int},
            "monitoring.max_consecutive_failures": {"min": 1, "max": 100, "type": int},
            "monitoring.max_queue_size_warning": {"min": 10, "max": 5000, "type": int},
            "monitoring.max_memory_usage_warning": {"min": 50, "max": 100, "type": int},
            "monitoring.max_response_time_warning": {
                "min": 100,
                "max": 30000,
                "type": int,
            },
        }

    def load_config(self) -> TelegramConfig:
        """
        Load configuration from file with validation.

        Returns:
            TelegramConfig: Loaded and validated configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        try:
            self.logger.info(f"Loading Telegram configuration from {self.config_file}")

            if not self.config_file.exists():
                self.logger.warning(f"Config file not found: {self.config_file}")
                self.logger.info("Using default configuration")
                self.config = TelegramConfig()
                return self.config

            # Check if file has been modified
            current_mtime = datetime.fromtimestamp(
                self.config_file.stat().st_mtime, tz=timezone.utc
            )

            if self.config is not None and self.last_modified == current_mtime:
                self.logger.debug("Configuration file unchanged, using cached config")
                return self.config

            # Load YAML configuration
            with open(self.config_file, "r") as f:
                raw_config = yaml.safe_load(f)

            # Parse configuration
            self.config = self._parse_config(raw_config)
            self.last_modified = current_mtime

            # Validate configuration
            validation_errors = self._validate_config(self.config)
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {validation_errors}")

            self.logger.info("Telegram configuration loaded and validated successfully")
            return self.config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            if self.config is None:
                self.logger.info("Using default configuration as fallback")
                self.config = TelegramConfig()
            return self.config

    def _parse_config(self, raw_config: Dict[str, Any]) -> TelegramConfig:
        """Parse raw configuration into structured config object."""
        try:
            telegram_config = raw_config.get("telegram", {})

            config = TelegramConfig()

            # Parse service config
            service_config = telegram_config.get("service", {})
            config.service = TelegramServiceConfig(
                enabled=service_config.get("enabled", True),
                max_retries=service_config.get("max_retries", 3),
                retry_delay=service_config.get("retry_delay", 5),
                connection_timeout=service_config.get("connection_timeout", 30),
                health_check_interval=service_config.get("health_check_interval", 60),
            )

            # Parse security config
            security_config = telegram_config.get("security", {})
            config.security = TelegramSecurityConfig(
                auth_required=security_config.get("auth_required", True),
                admin_commands_enabled=security_config.get("admin_commands_enabled", True),
                command_rate_limiting=security_config.get("command_rate_limiting", True),
                max_unauthorized_attempts=security_config.get("max_unauthorized_attempts", 5),
            )

            # Parse queue config
            queue_config = telegram_config.get("queue", {})
            config.queue = TelegramQueueConfig(
                enabled=queue_config.get("enabled", True),
                persistence=queue_config.get("persistence", True),
                max_size=queue_config.get("max_size", 1000),
                persistence_file=queue_config.get("persistence_file", "logs/telegram_queue.json"),
                dead_letter_max_size=queue_config.get("dead_letter_max_size", 100),
                cleanup_interval=queue_config.get("cleanup_interval", 3600),
            )

            # Parse rate limits config
            rate_limits_config = telegram_config.get("rate_limits", {})
            config.rate_limits = TelegramRateLimitConfig(
                messages_per_minute=rate_limits_config.get("messages_per_minute", 20),
                commands_per_user_per_minute=rate_limits_config.get(
                    "commands_per_user_per_minute", 10
                ),
                admin_commands_per_minute=rate_limits_config.get("admin_commands_per_minute", 20),
                burst_messages=rate_limits_config.get("burst_messages", 5),
                burst_window=rate_limits_config.get("burst_window", 10),
            )

            # Parse notifications config
            notifications_config = telegram_config.get("notifications", {})
            config.notifications = TelegramNotificationConfig(
                trade_notifications=notifications_config.get("trade_notifications", True),
                signal_notifications=notifications_config.get("signal_notifications", True),
                position_notifications=notifications_config.get("position_notifications", True),
                risk_notifications=notifications_config.get("risk_notifications", True),
                system_notifications=notifications_config.get("system_notifications", True),
                performance_reports=notifications_config.get("performance_reports", True),
                min_trade_confidence=notifications_config.get("min_trade_confidence", 0.6),
                min_trade_amount=notifications_config.get("min_trade_amount", 100.0),
                min_pnl_threshold=notifications_config.get("min_pnl_threshold", 10.0),
                max_trades_per_minute=notifications_config.get("max_trades_per_minute", 10),
                max_signals_per_minute=notifications_config.get("max_signals_per_minute", 15),
                max_risk_alerts_per_minute=notifications_config.get(
                    "max_risk_alerts_per_minute", 5
                ),
            )

            # Parse formatting config
            formatting_config = telegram_config.get("formatting", {})
            config.formatting = TelegramFormattingConfig(
                parse_mode=formatting_config.get("parse_mode", "HTML"),
                include_timestamps=formatting_config.get("include_timestamps", True),
                use_emojis=formatting_config.get("use_emojis", True),
                truncate_long_messages=formatting_config.get("truncate_long_messages", True),
                max_message_length=formatting_config.get("max_message_length", 4096),
                currency_format=formatting_config.get("currency_format", "${:.4f}"),
                percentage_format=formatting_config.get("percentage_format", "{:.1f}%"),
                datetime_format=formatting_config.get("datetime_format", "%Y-%m-%d %H:%M:%S UTC"),
            )

            # Parse monitoring config
            monitoring_config = telegram_config.get("monitoring", {})
            config.monitoring = TelegramMonitoringConfig(
                enabled=monitoring_config.get("enabled", True),
                health_check_commands=monitoring_config.get("health_check_commands", True),
                performance_tracking=monitoring_config.get("performance_tracking", True),
                error_reporting=monitoring_config.get("error_reporting", True),
                max_consecutive_failures=monitoring_config.get("max_consecutive_failures", 5),
                max_queue_size_warning=monitoring_config.get("max_queue_size_warning", 800),
                max_memory_usage_warning=monitoring_config.get("max_memory_usage_warning", 85),
                max_response_time_warning=monitoring_config.get("max_response_time_warning", 5000),
            )

            # Store additional configurations as dictionaries
            config.commands = telegram_config.get("commands", {})
            config.logging = telegram_config.get("logging", {})
            config.development = raw_config.get("development", {})
            config.production = raw_config.get("production", {})
            config.features = raw_config.get("features", {})

            return config

        except Exception as e:
            self.logger.error(f"Error parsing configuration: {e}")
            raise ValueError(f"Invalid configuration format: {e}")

    def _validate_config(self, config: TelegramConfig) -> List[str]:
        """
        Validate configuration values against rules.

        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Validate using rules
            for rule_path, rule in self.validation_rules.items():
                value = self._get_config_value(config, rule_path)

                if value is None:
                    continue  # Skip if value not set

                # Type validation
                expected_type = rule.get("type")
                if expected_type and not isinstance(value, expected_type):
                    errors.append(
                        f"{rule_path}: Expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                    continue

                # Range validation
                min_val = rule.get("min")
                max_val = rule.get("max")

                if min_val is not None and value < min_val:
                    errors.append(f"{rule_path}: Value {value} is below minimum {min_val}")

                if max_val is not None and value > max_val:
                    errors.append(f"{rule_path}: Value {value} is above maximum {max_val}")

            # Custom validation logic
            errors.extend(self._custom_validation(config))

        except Exception as e:
            errors.append(f"Validation error: {e}")

        return errors

    def _get_config_value(self, config: TelegramConfig, path: str) -> Any:
        """Get configuration value by dot-separated path."""
        parts = path.split(".")
        value = config

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

        return value

    def _custom_validation(self, config: TelegramConfig) -> List[str]:
        """Custom validation logic for complex rules."""
        errors = []

        # Validate parse mode
        valid_parse_modes = ["HTML", "Markdown", "MarkdownV2", None]
        if config.formatting.parse_mode not in valid_parse_modes:
            errors.append(
                f"formatting.parse_mode: Invalid parse mode '{config.formatting.parse_mode}'. Must be one of {valid_parse_modes}"
            )

        # Validate file paths
        queue_file = Path(config.queue.persistence_file)
        if not queue_file.parent.exists():
            try:
                queue_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(
                    f"queue.persistence_file: Cannot create directory {queue_file.parent}: {e}"
                )

        # Validate rate limiting consistency
        if config.rate_limits.burst_messages > config.rate_limits.messages_per_minute:
            errors.append("rate_limits.burst_messages cannot be greater than messages_per_minute")

        # Validate notification thresholds
        if (
            config.notifications.min_trade_confidence < 0
            or config.notifications.min_trade_confidence > 1
        ):
            errors.append("notifications.min_trade_confidence must be between 0.0 and 1.0")

        return errors

    def get_config(self) -> TelegramConfig:
        """
        Get current configuration, loading if necessary.

        Returns:
            TelegramConfig: Current configuration
        """
        if self.config is None:
            return self.load_config()

        # Check for file changes and reload if necessary
        if self.config_file.exists():
            current_mtime = datetime.fromtimestamp(
                self.config_file.stat().st_mtime, tz=timezone.utc
            )
            if self.last_modified != current_mtime:
                self.logger.info("Configuration file changed, reloading...")
                return self.load_config()

        return self.config

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values and save to file.

        Args:
            updates: Dictionary of configuration updates using dot notation

        Returns:
            bool: True if update successful
        """
        try:
            config = self.get_config()

            # Apply updates
            for path, value in updates.items():
                self._set_config_value(config, path, value)

            # Validate updated configuration
            validation_errors = self._validate_config(config)
            if validation_errors:
                self.logger.error(f"Configuration update validation failed: {validation_errors}")
                return False

            # Save to file if it exists
            if self.config_file.exists():
                self._save_config_to_file(config)

            self.logger.info(f"Configuration updated with {len(updates)} changes")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False

    def _set_config_value(self, config: TelegramConfig, path: str, value: Any):
        """Set configuration value by dot-separated path."""
        parts = path.split(".")
        target = config

        for part in parts[:-1]:
            target = getattr(target, part)

        setattr(target, parts[-1], value)

    def _save_config_to_file(self, config: TelegramConfig):
        """Save configuration back to YAML file."""
        # This is a simplified implementation
        # In practice, you'd want to preserve comments and structure
        self.logger.info(
            "Configuration save to file not implemented - using in-memory updates only"
        )

    def get_environment_overrides(self) -> Dict[str, Any]:
        """
        Get configuration overrides from environment variables.

        Returns:
            Dict[str, Any]: Environment-based configuration overrides
        """
        overrides = {}

        # Environment variable mappings
        env_mappings = {
            "TELEGRAM_RATE_LIMIT_MESSAGES": "rate_limits.messages_per_minute",
            "TELEGRAM_MAX_QUEUE_SIZE": "queue.max_size",
            "TELEGRAM_MIN_CONFIDENCE": "notifications.min_trade_confidence",
            "TELEGRAM_DEBUG": "development.verbose_logging",
            "TELEGRAM_MOCK_MODE": "development.mock_mode",
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "." in value and value.replace(".", "").isdigit():
                    value = float(value)

                overrides[config_path] = value

        if overrides:
            self.logger.info(f"Found {len(overrides)} environment overrides")

        return overrides

    def apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        overrides = self.get_environment_overrides()
        if overrides:
            self.update_config(overrides)

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration for monitoring/debugging.

        Returns:
            Dict[str, Any]: Configuration summary
        """
        config = self.get_config()

        return {
            "config_file": str(self.config_file),
            "last_modified": (self.last_modified.isoformat() if self.last_modified else None),
            "service_enabled": config.service.enabled,
            "auth_required": config.security.auth_required,
            "queue_enabled": config.queue.enabled,
            "queue_max_size": config.queue.max_size,
            "notifications_enabled": {
                "trades": config.notifications.trade_notifications,
                "signals": config.notifications.signal_notifications,
                "positions": config.notifications.position_notifications,
                "risk": config.notifications.risk_notifications,
                "system": config.notifications.system_notifications,
            },
            "rate_limits": {
                "messages_per_minute": config.rate_limits.messages_per_minute,
                "commands_per_user_per_minute": config.rate_limits.commands_per_user_per_minute,
            },
            "monitoring_enabled": config.monitoring.enabled,
            "development_mode": config.development.get("mock_mode", False),
        }


# Global configuration manager instance
_config_manager = None


def get_telegram_config_manager() -> TelegramConfigManager:
    """Get singleton Telegram configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = TelegramConfigManager()
    return _config_manager
