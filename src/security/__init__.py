"""
Security module for trading bot.
Handles credential management, authentication, and security validation.
"""

from .credential_manager import (
    CredentialManager,
    ExchangeCredentials,
    TelegramCredentials,
    get_credential_manager,
)

__all__ = [
    "CredentialManager",
    "TelegramCredentials",
    "ExchangeCredentials",
    "get_credential_manager",
]
