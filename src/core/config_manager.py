"""Enhanced configuration management with validation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import yaml
from jsonschema import ValidationError, validate

from .base_service import BaseService
from .container import injectable
from .interfaces import IConfigurationManager, ILogger, ValidationResult


@injectable
class ConfigurationManager(BaseService, IConfigurationManager):
    """Enhanced configuration manager with validation."""

    def __init__(self, logger: ILogger):
        super().__init__(logger)
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._config_dir = Path("src/config")
        self._load_schemas()

    def initialize(self) -> bool:
        """Initialize configuration manager."""
        if not super().initialize():
            return False

        try:
            # Load all available configurations
            self._load_all_configs()
            return True
        except Exception as e:
            self._log_error("Failed to initialize configuration manager", exception=e)
            return False

    def load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration by type."""
        self._ensure_initialized()

        if config_type in self._configs:
            return self._configs[config_type].copy()

        # Try to load the config file
        config_file = self._config_dir / f"config_{config_type}.yaml"
        if not config_file.exists():
            config_file = self._config_dir / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found for type: {config_type}")

        config = self._load_yaml_file(config_file)

        # Validate if schema exists
        validation_result = self.validate_config(config)
        if not validation_result.is_valid:
            self._log_warning(
                f"Configuration validation failed for {config_type}",
                {
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                },
            )

        self._configs[config_type] = config
        return config.copy()

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration structure and values."""
        errors = []
        warnings = []
        metadata = {}

        try:
            # Basic structure validation
            required_sections = ["data", "models", "trading"]
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required section: {section}")

            # Data section validation
            if "data" in config:
                data_config = config["data"]
                if "symbols" not in data_config or not data_config["symbols"]:
                    errors.append("Data section must contain symbols list")

                if "timeframe" not in data_config:
                    warnings.append("No timeframe specified in data section")

            # Models section validation
            if "models" in config:
                models_config = config["models"]
                valid_model_types = ["gru", "lightgbm", "ppo"]

                for model_type in models_config:
                    if model_type not in valid_model_types:
                        warnings.append(f"Unknown model type: {model_type}")

            # Trading section validation
            if "trading" in config:
                trading_config = config["trading"]
                if "initial_balance" not in trading_config:
                    warnings.append("No initial balance specified")

                if "risk_management" not in trading_config:
                    warnings.append("No risk management configuration")

            # Environment variables validation
            env_vars = ["BINANCE_API_KEY", "BINANCE_SECRET_KEY"]
            for var in env_vars:
                if not os.getenv(var):
                    warnings.append(f"Environment variable {var} not set")

            metadata["validation_timestamp"] = self._get_full_context()["timestamp"]
            metadata["config_sections"] = list(config.keys())

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def get_symbols(self) -> List[str]:
        """Get list of trading symbols."""
        self._ensure_initialized()

        # Try to get from cached configs first
        for config in self._configs.values():
            if "data" in config and "symbols" in config["data"]:
                return config["data"]["symbols"]

        # Load trading config if not cached
        try:
            config = self.load_config("trading")
            return config.get("data", {}).get("symbols", [])
        except Exception:
            # Fallback to training config
            try:
                config = self.load_config("training")
                return config.get("data", {}).get("symbols", [])
            except Exception:
                return []

    def get_config_value(self, config_type: str, key_path: str, default: Any = None) -> Any:
        """Get a specific configuration value using dot notation."""
        config = self.load_config(config_type)

        keys = key_path.split(".")
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def reload_config(self, config_type: str) -> bool:
        """Reload a specific configuration."""
        try:
            if config_type in self._configs:
                del self._configs[config_type]

            self.load_config(config_type)
            self._log_info(f"Reloaded configuration: {config_type}")
            return True
        except Exception as e:
            self._log_error(f"Failed to reload configuration: {config_type}", exception=e)
            return False

    def _load_all_configs(self):
        """Load all available configuration files."""
        config_files = {
            "trading": "config_trading.yaml",
            "training": "config_training.yaml",
            "default": "config.yaml",
        }

        for config_type, filename in config_files.items():
            config_path = self._config_dir / filename
            if config_path.exists():
                try:
                    config = self._load_yaml_file(config_path)
                    self._configs[config_type] = config
                    self._log_info(f"Loaded configuration: {config_type}")
                except Exception as e:
                    self._log_warning(f"Failed to load {filename}", exception=e)

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file safely."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self._log_error(f"Failed to load YAML file: {file_path}", exception=e)
            raise

    def _load_schemas(self):
        """Load configuration schemas for validation."""
        # Basic schema for trading configuration
        self._schemas["trading"] = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "symbols": {"type": "array", "items": {"type": "string"}},
                        "timeframe": {"type": "string"},
                    },
                    "required": ["symbols"],
                },
                "models": {"type": "object"},
                "trading": {
                    "type": "object",
                    "properties": {
                        "initial_balance": {"type": "number"},
                        "risk_management": {"type": "object"},
                    },
                },
            },
            "required": ["data", "models", "trading"],
        }
