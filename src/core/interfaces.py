"""Core interfaces for the trading system architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ModelMetadata:
    """Metadata for a trading model."""

    model_type: str
    symbol: str
    version: str
    features: List[str]
    created_at: datetime
    performance_metrics: Dict[str, float]
    config: Dict[str, Any]


class IConfigurationManager(ABC):
    """Interface for configuration management."""

    @abstractmethod
    def load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration by type."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration structure and values."""
        pass

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of trading symbols."""
        pass


class IFeatureManager(ABC):
    """Interface for feature management."""

    @abstractmethod
    def load_feature_schema(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Load feature schema for a specific symbol and model type."""
        pass

    @abstractmethod
    def validate_features(self, features: pd.DataFrame, schema: Dict[str, Any]) -> ValidationResult:
        """Validate features against schema."""
        pass

    @abstractmethod
    def detect_schema_drift(
        self, current_features: pd.DataFrame, reference_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Detect schema drift in features."""
        pass


class IModelManager(ABC):
    """Interface for model management."""

    @abstractmethod
    def load_model(self, symbol: str, model_type: str) -> Any:
        """Load a model for the given symbol and type."""
        pass

    @abstractmethod
    def get_model_metadata(self, symbol: str, model_type: str) -> ModelMetadata:
        """Get metadata for a model."""
        pass

    @abstractmethod
    def validate_model_compatibility(
        self, model_metadata: ModelMetadata, feature_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate model-feature compatibility."""
        pass


class IDataProvider(ABC):
    """Interface for data provision."""

    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get market data for a symbol."""
        pass

    @abstractmethod
    def get_features(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get processed features for a symbol."""
        pass


class ITradingEngine(ABC):
    """Interface for trading execution."""

    @abstractmethod
    def execute_trade(
        self, symbol: str, action: str, amount: float, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a trade."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a symbol."""
        pass


class ILogger(ABC):
    """Interface for logging."""

    @abstractmethod
    def log_info(self, message: str, context: Dict[str, Any] = None):
        """Log info message."""
        pass

    @abstractmethod
    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """Log warning message."""
        pass

    @abstractmethod
    def log_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log error message."""
        pass

    @abstractmethod
    def log_trade(self, symbol: str, action: str, amount: float, metadata: Dict[str, Any]):
        """Log trade execution."""
        pass


class INotificationService(ABC):
    """Interface for notifications."""

    @abstractmethod
    def send_notification(self, message: str, level: str = "info", context: Dict[str, Any] = None):
        """Send a notification."""
        pass
