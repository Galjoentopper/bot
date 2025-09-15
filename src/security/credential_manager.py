"""
Secure Credential Manager for Trading Bot
Handles secure loading and validation of sensitive credentials.
"""

import base64
import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TelegramCredentials:
    """Secure storage for Telegram credentials."""

    bot_token: str
    chat_id: str

    def __post_init__(self):
        """Validate credentials format."""
        if not self.bot_token or self.bot_token.startswith("your_"):
            raise ValueError("Invalid Telegram bot token - appears to be placeholder")

        if not self.chat_id or not self.chat_id.isdigit():
            raise ValueError("Invalid Telegram chat ID - must be numeric")

        # Validate bot token format (should be like 123456:ABC-DEF...)
        if ":" not in self.bot_token or len(self.bot_token.split(":")[0]) < 8:
            raise ValueError("Invalid Telegram bot token format")


@dataclass
class ExchangeCredentials:
    """Secure storage for exchange API credentials."""

    api_key: str
    api_secret: str

    def __post_init__(self):
        """Validate credentials."""
        if self.api_key and (self.api_key.startswith("your_") or len(self.api_key) < 10):
            raise ValueError("Invalid API key format")
        if self.api_secret and (self.api_secret.startswith("your_") or len(self.api_secret) < 10):
            raise ValueError("Invalid API secret format")


class CredentialManager:
    """
    Secure credential manager with validation and audit logging.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._credentials_loaded = False
        self._telegram_creds: Optional[TelegramCredentials] = None
        self._bitvavo_creds: Optional[ExchangeCredentials] = None
        self._binance_creds: Optional[ExchangeCredentials] = None

    def load_credentials(self) -> bool:
        """
        Load and validate all credentials from environment.

        Returns:
            bool: True if all required credentials loaded successfully
        """
        try:
            # Load Telegram credentials (required)
            telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
            telegram_chat = os.getenv("TELEGRAM_CHAT_ID")

            if not telegram_token or not telegram_chat:
                self.logger.error("Missing required Telegram credentials")
                return False

            self._telegram_creds = TelegramCredentials(
                bot_token=telegram_token, chat_id=telegram_chat
            )

            # Load exchange credentials (optional)
            bitvavo_key = os.getenv("BITVAVO_API_KEY", "")
            bitvavo_secret = os.getenv("BITVAVO_API_SECRET", "")

            if bitvavo_key and bitvavo_secret:
                self._bitvavo_creds = ExchangeCredentials(
                    api_key=bitvavo_key, api_secret=bitvavo_secret
                )

            binance_key = os.getenv("BINANCE_API_KEY", "")
            binance_secret = os.getenv("BINANCE_API_SECRET", "")

            if binance_key and binance_secret:
                self._binance_creds = ExchangeCredentials(
                    api_key=binance_key, api_secret=binance_secret
                )

            self._credentials_loaded = True
            self.logger.info("Credentials loaded successfully")
            return True

        except ValueError as e:
            self.logger.error(f"Credential validation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
            return False

    @property
    def telegram_credentials(self) -> Optional[TelegramCredentials]:
        """Get validated Telegram credentials."""
        if not self._credentials_loaded:
            self.load_credentials()
        return self._telegram_creds

    @property
    def bitvavo_credentials(self) -> Optional[ExchangeCredentials]:
        """Get validated Bitvavo credentials."""
        if not self._credentials_loaded:
            self.load_credentials()
        return self._bitvavo_creds

    @property
    def binance_credentials(self) -> Optional[ExchangeCredentials]:
        """Get validated Binance credentials."""
        if not self._credentials_loaded:
            self.load_credentials()
        return self._binance_creds

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate current environment configuration.

        Returns:
            Dict with validation results
        """
        validation_result = {"valid": True, "errors": [], "warnings": [], "credentials_found": {}}

        # Check for required environment variables
        required_vars = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

        for var in required_vars:
            value = os.getenv(var)
            if not value:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required environment variable: {var}")
            elif value.startswith("your_"):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Environment variable {var} contains placeholder value"
                )
            else:
                validation_result["credentials_found"][var] = "✓ Found"

        # Check optional credentials
        optional_pairs = [
            ("BITVAVO_API_KEY", "BITVAVO_API_SECRET"),
            ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
            ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        ]

        for key_var, secret_var in optional_pairs:
            key_val = os.getenv(key_var, "")
            secret_val = os.getenv(secret_var, "")

            if key_val and secret_val:
                if key_val.startswith("your_") or secret_val.startswith("your_"):
                    validation_result["warnings"].append(
                        f"{key_var}/{secret_var} contain placeholder values"
                    )
                else:
                    validation_result["credentials_found"][f"{key_var}_PAIR"] = "✓ Found"
            elif key_val or secret_val:
                validation_result["warnings"].append(
                    f"Partial credentials for {key_var}/{secret_var} - need both or neither"
                )

        return validation_result

    def get_credential_hash(self, credential_type: str) -> Optional[str]:
        """
        Get a hash of credentials for audit purposes (not the actual credential).

        Args:
            credential_type: Type of credential to hash

        Returns:
            SHA256 hash of credential (first 8 chars) or None if not found
        """
        credential_value = None

        if credential_type == "telegram_token" and self._telegram_creds:
            credential_value = self._telegram_creds.bot_token
        elif credential_type == "telegram_chat" and self._telegram_creds:
            credential_value = self._telegram_creds.chat_id
        elif credential_type == "bitvavo_key" and self._bitvavo_creds:
            credential_value = self._bitvavo_creds.api_key

        if credential_value:
            hash_obj = hashlib.sha256(credential_value.encode())
            return hash_obj.hexdigest()[:8]

        return None


# Global credential manager instance
_credential_manager = None


def get_credential_manager() -> CredentialManager:
    """Get singleton credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager
