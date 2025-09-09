"""Configuration adapter for legacy ConfigLoader."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.config_loader import ConfigLoader
from ..core.base_service import BaseService
from ..core.container import injectable
from ..core.interfaces import IConfigurationManager, ValidationResult


@injectable
class ConfigAdapter(BaseService, IConfigurationManager):
    """Adapter that wraps legacy ConfigLoader to implement IConfigurationManager."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the config adapter.

        Args:
            config_path: Optional path to config file. If None, auto-detection is used.
        """
        super().__init__()
        self._config_loader = ConfigLoader(config_path)
        self._config_path = config_path

    async def initialize(self) -> None:
        """Initialize the configuration adapter."""
        await super().initialize()
        self._log_info(
            f"ConfigAdapter initialized with path: {self._config_path or 'auto-detected'}"
        )

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation like 'trading.symbols')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            # Handle dot notation for nested keys
            if "." in key:
                keys = key.split(".")
                value = self._config_loader.config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
            else:
                return self._config_loader.get(key, default)
        except Exception as e:
            self._log_error(f"Error getting config key '{key}': {e}")
            return default

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config_loader.config.copy()

    def load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration by type."""
        try:
            if config_type == "trading":
                return self._config_loader.config.get("trading", {})
            elif config_type == "model":
                return self._config_loader.config.get("model", {})
            elif config_type == "data":
                return self._config_loader.config.get("data", {})
            else:
                return self._config_loader.config
        except Exception as e:
            self._log_error(f"Failed to load config type '{config_type}': {e}")
            return {}

    def get_symbols(self) -> List[str]:
        """Get list of trading symbols."""
        try:
            symbols = self._config_loader.get("symbols", [])
            if isinstance(symbols, str):
                return [symbols]
            elif isinstance(symbols, list):
                return symbols
            else:
                return []
        except Exception as e:
            self._log_error(f"Failed to get symbols from config: {e}")
            return []

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate configuration structure and values."""
        try:
            # Use provided config or current config
            config_to_validate = config or self._config_loader.config

            errors = []
            warnings = []

            # Basic validation checks
            required_keys = ["symbols", "interval", "initial_balance"]
            for key in required_keys:
                if key not in config_to_validate:
                    errors.append(f"Missing required configuration key: {key}")

            # Validate symbols
            if "symbols" in config_to_validate:
                symbols = config_to_validate["symbols"]
                if not isinstance(symbols, list) or not symbols:
                    errors.append("Symbols must be a non-empty list")

            # Validate interval
            if "interval" in config_to_validate:
                interval = config_to_validate["interval"]
                valid_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                if interval not in valid_intervals:
                    warnings.append(f"Interval '{interval}' may not be supported")

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata={"config_keys": list(config_to_validate.keys())},
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Configuration validation failed: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def reload_config(self) -> bool:
        """Reload configuration from source.

        Returns:
            True if reload was successful
        """
        try:
            # Create new ConfigLoader instance to reload
            self._config_loader = ConfigLoader(self._config_path)
            self._log_info("Configuration reloaded successfully")
            return True
        except Exception as e:
            self._log_error(f"Failed to reload configuration: {e}")
            return False

    def get_config_path(self) -> Optional[str]:
        """Get the current configuration file path.

        Returns:
            Path to configuration file or None if auto-detected
        """
        return self._config_path

    def has_config(self, key: str) -> bool:
        """Check if configuration key exists.

        Args:
            key: Configuration key to check

        Returns:
            True if key exists
        """
        try:
            if "." in key:
                keys = key.split(".")
                value = self._config_loader.config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return False
                return True
            else:
                return key in self._config_loader.config
        except Exception:
            return False
