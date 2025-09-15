"""Secure environment variable manager for the trading bot system.

This module provides centralized, secure management of all environment variables
with validation, type conversion, and security best practices.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from ..core.logging_manager import TradingBotLogger


class SecureEnvManager:
    """Secure environment variable manager with validation and type conversion."""

    # Define all required and optional environment variables with their types and defaults
    ENV_SCHEMA = {
        # Trading Configuration
        "INITIAL_BALANCE": {"type": float, "default": 10000.0, "required": False},
        "MAX_POSITION_SIZE": {"type": float, "default": 0.1, "required": False},
        "TRANSACTION_FEE": {"type": float, "default": 0.001, "required": False},
        # API Keys - REQUIRED for production
        "TELEGRAM_BOT_TOKEN": {"type": str, "required": True, "sensitive": True},
        "TELEGRAM_CHAT_ID": {"type": str, "required": True, "sensitive": True},
        "BITVAVO_API_KEY": {"type": str, "required": False, "sensitive": True},
        "BITVAVO_API_SECRET": {"type": str, "required": False, "sensitive": True},
        "BINANCE_API_KEY": {"type": str, "required": False, "sensitive": True},
        "BINANCE_API_SECRET": {"type": str, "required": False, "sensitive": True},
        # AWS Configuration
        "AWS_ACCESS_KEY_ID": {"type": str, "required": False, "sensitive": True},
        "AWS_SECRET_ACCESS_KEY": {"type": str, "required": False, "sensitive": True},
        "AWS_DEFAULT_REGION": {"type": str, "default": "us-east-1", "required": False},
        "AWS_MODELS_BUCKET": {
            "type": str,
            "default": "paperspace-models-1757381376",
            "required": False,
        },
        # MLflow Configuration
        "MLFLOW_TRACKING_URI": {"type": str, "default": "./mlruns", "required": False},
        # Logging Configuration
        "LOG_LEVEL": {"type": str, "default": "INFO", "required": False},
        "LOG_FILE": {"type": str, "default": "logs/trading_bot.log", "required": False},
        # Data Configuration
        "DATA_DIR": {"type": str, "default": "./data", "required": False},
        "MODELS_DIR": {"type": str, "default": "./models", "required": False},
        # GPU Configuration
        "CUDA_VISIBLE_DEVICES": {"type": str, "default": "0", "required": False},
        # Production Configuration
        "PRODUCTION_API_KEY": {"type": str, "required": False, "sensitive": True},
        "ENVIRONMENT": {"type": str, "default": "production", "required": False},
    }

    # Insecure placeholder values that should never be used
    INSECURE_PLACEHOLDERS = {
        "your_aws_access_key_here",
        "your_aws_secret_key_here",
        "your_bitvavo_key_here",
        "your_bitvavo_secret_here",
        "your_bot_token_here",
        "your_chat_id_here",
        "your-access-key-here",
        "your-secret-key-here",
        "CHANGEME",
        "REDACTED",
        "test_token",
        "test_chat_id",
    }

    def __init__(self, env_file_path: Optional[str] = None):
        """Initialize the secure environment manager.

        Args:
            env_file_path: Optional path to .env file. If None, auto-detects.
        """
        from src.core.logging_manager import get_system_logger

        self.logger = get_system_logger("SecureEnvManager")
        self._env_loaded = False
        self._validated_vars: Dict[str, Any] = {}

        # Find and load .env file
        self.env_file_path = self._find_env_file(env_file_path)
        self._load_env_file()

    def _find_env_file(self, env_file_path: Optional[str] = None) -> Optional[str]:
        """Find the .env file to load.

        Args:
            env_file_path: Optional explicit path to .env file

        Returns:
            Path to .env file or None if not found
        """
        if env_file_path and Path(env_file_path).exists():
            return env_file_path

        # Try common locations
        search_paths = [
            Path.cwd() / ".env",
            Path(__file__).parent.parent.parent / ".env",
            Path("/opt/trading_bot/bot/.env"),
        ]

        for path in search_paths:
            if path.exists():
                return str(path)

        self.logger.warning("No .env file found in common locations")
        return None

    def _load_env_file(self) -> None:
        """Load the .env file if it exists."""
        if self.env_file_path:
            try:
                load_dotenv(self.env_file_path, override=True)
                self._env_loaded = True
                self.logger.info(f"Loaded environment variables from {self.env_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to load .env file {self.env_file_path}: {e}")
        else:
            self.logger.info("No .env file loaded, using system environment variables only")

    def get(self, key: str, default: Any = None, required: bool = None) -> Any:
        """Get environment variable with type conversion and validation.

        Args:
            key: Environment variable name
            default: Default value if not set
            required: Override required setting from schema

        Returns:
            Environment variable value with proper type conversion

        Raises:
            ValueError: If required variable is missing or has insecure value
        """
        # Use cached value if already validated
        if key in self._validated_vars:
            return self._validated_vars[key]

        # Get schema info
        schema = self.ENV_SCHEMA.get(key, {})
        var_type = schema.get("type", str)
        is_required = required if required is not None else schema.get("required", False)
        schema_default = schema.get("default")
        is_sensitive = schema.get("sensitive", False)

        # Get raw value from environment
        raw_value = os.getenv(key)

        # Use provided default, schema default, or None
        if raw_value is None:
            if default is not None:
                final_value = default
            elif schema_default is not None:
                final_value = schema_default
            elif is_required:
                raise ValueError(f"Required environment variable '{key}' is not set")
            else:
                final_value = None
        else:
            final_value = raw_value

        # Validate sensitive variables for insecure placeholders
        if is_sensitive and final_value and str(final_value) in self.INSECURE_PLACEHOLDERS:
            if is_required:
                raise ValueError(
                    f"Environment variable '{key}' contains insecure placeholder value"
                )
            else:
                self.logger.warning(
                    f"Environment variable '{key}' contains placeholder value, using None"
                )
                final_value = None

        # Type conversion
        if final_value is not None and var_type != str:
            try:
                if var_type == bool:
                    final_value = str(final_value).lower() in ("true", "1", "yes", "on")
                elif var_type == int:
                    final_value = int(final_value)
                elif var_type == float:
                    final_value = float(final_value)
                # str type needs no conversion
            except (ValueError, TypeError) as e:
                self.logger.error(f"Failed to convert '{key}' to {var_type.__name__}: {e}")
                if is_required:
                    raise
                final_value = schema_default

        # Cache the validated value
        self._validated_vars[key] = final_value

        # Log access (mask sensitive values)
        if is_sensitive and final_value:
            self.logger.debug(f"Retrieved sensitive environment variable '{key}': ***MASKED***")
        else:
            self.logger.debug(f"Retrieved environment variable '{key}': {final_value}")

        return final_value

    def get_telegram_config(self) -> Dict[str, str]:
        """Get Telegram configuration with validation.

        Returns:
            Dictionary with bot_token and chat_id

        Raises:
            ValueError: If Telegram credentials are missing or invalid
        """
        bot_token = self.get("TELEGRAM_BOT_TOKEN", required=True)
        chat_id = self.get("TELEGRAM_CHAT_ID", required=True)

        if not bot_token or not chat_id:
            raise ValueError("Telegram credentials are required for notifications")

        return {"bot_token": bot_token, "chat_id": chat_id}

    def get_trading_config(self) -> Dict[str, Union[float, str]]:
        """Get trading configuration.

        Returns:
            Dictionary with trading parameters
        """
        return {
            "initial_balance": self.get("INITIAL_BALANCE"),
            "max_position_size": self.get("MAX_POSITION_SIZE"),
            "transaction_fee": self.get("TRANSACTION_FEE"),
        }

    def get_aws_config(self) -> Dict[str, Optional[str]]:
        """Get AWS configuration.

        Returns:
            Dictionary with AWS credentials and settings
        """
        return {
            "access_key_id": self.get("AWS_ACCESS_KEY_ID"),
            "secret_access_key": self.get("AWS_SECRET_ACCESS_KEY"),
            "default_region": self.get("AWS_DEFAULT_REGION"),
            "models_bucket": self.get("AWS_MODELS_BUCKET"),
        }

    def get_bitvavo_config(self) -> Dict[str, Optional[str]]:
        """Get Bitvavo API configuration.

        Returns:
            Dictionary with Bitvavo API credentials
        """
        return {
            "api_key": self.get("BITVAVO_API_KEY"),
            "api_secret": self.get("BITVAVO_API_SECRET"),
        }

    def get_binance_config(self) -> Dict[str, Optional[str]]:
        """Get Binance API configuration.

        Returns:
            Dictionary with Binance API credentials
        """
        return {
            "api_key": self.get("BINANCE_API_KEY"),
            "api_secret": self.get("BINANCE_API_SECRET"),
        }

    def validate_all_required(self) -> List[str]:
        """Validate all required environment variables are set.

        Returns:
            List of missing required variables
        """
        missing_vars = []

        for var_name, schema in self.ENV_SCHEMA.items():
            if schema.get("required", False):
                try:
                    value = self.get(var_name)
                    if value is None:
                        missing_vars.append(var_name)
                except ValueError:
                    missing_vars.append(var_name)

        return missing_vars

    def get_security_status(self) -> Dict[str, Any]:
        """Get security status of all environment variables.

        Returns:
            Dictionary with security analysis
        """
        status = {
            "secure_vars": [],
            "insecure_vars": [],
            "missing_required": [],
            "recommendations": [],
        }

        for var_name, schema in self.ENV_SCHEMA.items():
            is_sensitive = schema.get("sensitive", False)
            is_required = schema.get("required", False)

            raw_value = os.getenv(var_name)

            if raw_value is None:
                if is_required:
                    status["missing_required"].append(var_name)
            elif is_sensitive:
                if raw_value in self.INSECURE_PLACEHOLDERS:
                    status["insecure_vars"].append(
                        {
                            "name": var_name,
                            "issue": "Contains placeholder value",
                            "value": raw_value,
                        }
                    )
                else:
                    status["secure_vars"].append(var_name)

        # Add recommendations
        if status["insecure_vars"]:
            status["recommendations"].append("Replace placeholder values with actual credentials")
        if status["missing_required"]:
            status["recommendations"].append("Set all required environment variables")
        if not self._env_loaded:
            status["recommendations"].append("Create .env file for centralized configuration")

        return status

    def export_secure_template(self, output_path: str = ".env.template") -> None:
        """Export secure .env template file.

        Args:
            output_path: Path where to save the template
        """
        template_lines = [
            "# Secure Environment Configuration for Trading Bot",
            "# Copy this file to .env and fill in your actual values",
            "# NEVER commit .env file with real credentials to git!",
            "",
            "# =============================================================================",
            "# REQUIRED VARIABLES (Must be set for system to work)",
            "# =============================================================================",
            "",
            "# Telegram Bot Configuration (Required for notifications)",
            "TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here",
            "TELEGRAM_CHAT_ID=your_telegram_chat_id_here",
            "",
            "# =============================================================================",
            "# OPTIONAL API KEYS (For live trading - leave blank for paper trading)",
            "# =============================================================================",
            "",
            "# Bitvavo Exchange API",
            "BITVAVO_API_KEY=",
            "BITVAVO_API_SECRET=",
            "",
            "# Binance Exchange API",
            "BINANCE_API_KEY=",
            "BINANCE_API_SECRET=",
            "",
            "# AWS Configuration (For model storage)",
            "AWS_ACCESS_KEY_ID=",
            "AWS_SECRET_ACCESS_KEY=",
            "AWS_DEFAULT_REGION=us-east-1",
            "AWS_MODELS_BUCKET=paperspace-models-1757381376",
            "",
            "# =============================================================================",
            "# SYSTEM CONFIGURATION (Optional - defaults provided)",
            "# =============================================================================",
            "",
            "# Trading Parameters",
            "INITIAL_BALANCE=10000.0",
            "MAX_POSITION_SIZE=0.1",
            "TRANSACTION_FEE=0.001",
            "",
            "# Logging Configuration",
            "LOG_LEVEL=INFO",
            "LOG_FILE=logs/trading_bot.log",
            "",
            "# Data and Model Directories",
            "DATA_DIR=./data",
            "MODELS_DIR=./models",
            "",
            "# MLflow Tracking",
            "MLFLOW_TRACKING_URI=./mlruns",
            "",
            "# GPU Configuration",
            "CUDA_VISIBLE_DEVICES=0",
            "",
            "# Production Configuration",
            "ENVIRONMENT=production",
            "PRODUCTION_API_KEY=",
            "",
            "# =============================================================================",
            "# SECURITY NOTES:",
            "# - Never commit .env file to version control",
            "# - Use strong, unique API keys",
            "# - Regularly rotate sensitive credentials",
            "# - Monitor API usage for unauthorized access",
            "# =============================================================================",
        ]

        try:
            with open(output_path, "w") as f:
                f.write("\n".join(template_lines))
            self.logger.info(f"Secure .env template exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export template: {e}")
            raise


# Global instance for easy access
_global_env_manager: Optional[SecureEnvManager] = None


def get_env_manager() -> SecureEnvManager:
    """Get the global environment manager instance.

    Returns:
        Global SecureEnvManager instance
    """
    global _global_env_manager
    if _global_env_manager is None:
        _global_env_manager = SecureEnvManager()
    return _global_env_manager


def get_env(key: str, default: Any = None, required: bool = None) -> Any:
    """Convenience function to get environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Override required setting from schema

    Returns:
        Environment variable value with proper type conversion
    """
    return get_env_manager().get(key, default, required)
