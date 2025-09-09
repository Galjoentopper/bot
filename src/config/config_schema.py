"""Configuration schema validation using Pydantic."""

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelType(str, Enum):
    GRU = "gru"
    LIGHTGBM = "lightgbm"
    PPO = "ppo"


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITY = "equity"


class IntervalType(str, Enum):
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class DataSourceConfig(BaseModel):
    """Data source configuration."""

    primary: str = Field(..., description="Primary data source")
    fallback: Optional[str] = Field(None, description="Fallback data source")
    timeout: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(3, ge=1, le=10, description="Number of retry attempts")


class ModelWeights(BaseModel):
    """Model ensemble weights configuration."""

    lightgbm: float = Field(0.55, ge=0.0, le=1.0)
    gru: float = Field(0.35, ge=0.0, le=1.0)
    ppo: float = Field(0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self):
        """Ensure weights sum to 1.0."""
        total = self.lightgbm + self.gru + self.ppo
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Model weights must sum to 1.0, got {total}")
        return self


class ThresholdConfig(BaseModel):
    """Trading threshold configuration."""

    base_buy_threshold: float = Field(0.6, ge=0.1, le=0.9)
    base_sell_threshold: float = Field(-0.6, ge=-0.9, le=-0.1)
    confidence_multiplier: float = Field(1.2, ge=1.0, le=2.0)
    volatility_adjustment: bool = Field(True)


class SymbolThresholds(BaseModel):
    """Per-symbol threshold configuration."""

    buy_threshold: float = Field(..., ge=0.1, le=0.9)
    sell_threshold: float = Field(..., ge=-0.9, le=-0.1)
    max_position_size: float = Field(1000.0, ge=10.0, le=100000.0)

    @field_validator("sell_threshold")
    @classmethod
    def sell_threshold_negative(cls, v):
        """Ensure sell threshold is negative."""
        if v > 0:
            raise ValueError("Sell threshold must be negative")
        return v


class RiskManagementConfig(BaseModel):
    """Risk management configuration."""

    max_drawdown_pct: float = Field(20.0, ge=5.0, le=50.0)
    position_size_pct: float = Field(2.0, ge=0.1, le=10.0)
    stop_loss_pct: float = Field(5.0, ge=1.0, le=20.0)
    take_profit_pct: float = Field(10.0, ge=2.0, le=50.0)
    max_open_positions: int = Field(5, ge=1, le=20)


class TelegramConfig(BaseModel):
    """Telegram notification configuration."""

    enabled: bool = Field(True)
    bot_token: Optional[str] = Field(None)
    chat_id: Optional[str] = Field(None)
    notification_level: LogLevel = Field(LogLevel.INFO)
    trade_notifications: bool = Field(True)
    error_notifications: bool = Field(True)


class ModelConfig(BaseModel):
    """Individual model configuration."""

    enabled: bool = Field(True)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    validation_split: float = Field(0.2, ge=0.1, le=0.5)
    early_stopping_patience: int = Field(10, ge=5, le=50)


class TradingConfig(BaseModel):
    """Main trading configuration schema."""

    # Basic settings
    symbols: List[str] = Field(..., min_items=1, description="Trading symbols")
    interval: IntervalType = Field(IntervalType.MIN_30, description="Candle interval")
    lookback_days: int = Field(365, ge=30, le=2000, description="Historical data days")

    # Data source
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)

    # Model configuration
    model_weights: ModelWeights = Field(default_factory=ModelWeights)
    models: Dict[ModelType, ModelConfig] = Field(default_factory=dict)

    # Trading thresholds
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    per_symbol: Dict[str, SymbolThresholds] = Field(default_factory=dict)

    # Risk management
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)

    # Notifications
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)

    # Feature engineering
    feature_config: Dict[str, Any] = Field(default_factory=dict)

    # Advanced settings
    paper_trading: bool = Field(True, description="Paper trading mode")
    log_level: LogLevel = Field(LogLevel.INFO)
    debug_mode: bool = Field(False)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        """Validate symbol format."""
        for symbol in v:
            if not symbol.isupper():
                raise ValueError(f"Symbol {symbol} must be uppercase")
            if len(symbol) < 3:
                raise ValueError(f"Symbol {symbol} too short")
        return v

    @model_validator(mode="after")
    def validate_per_symbol_config(self):
        """Ensure all symbols have threshold configuration."""
        missing_symbols = set(self.symbols) - set(self.per_symbol.keys())
        if missing_symbols:
            # Auto-generate default thresholds for missing symbols
            for symbol in missing_symbols:
                self.per_symbol[symbol] = SymbolThresholds(
                    buy_threshold=self.thresholds.base_buy_threshold,
                    sell_threshold=self.thresholds.base_sell_threshold,
                    max_position_size=1000.0,
                )

        return self


class EnvironmentConfig(BaseModel):
    """Environment variables validation."""

    # Required environment variables
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")

    # Optional API keys
    bitvavo_api_key: Optional[str] = Field(None, env="BITVAVO_API_KEY")
    bitvavo_api_secret: Optional[str] = Field(None, env="BITVAVO_API_SECRET")
    binance_api_key: Optional[str] = Field(None, env="BINANCE_API_KEY")
    binance_api_secret: Optional[str] = Field(None, env="BINANCE_API_SECRET")

    # MLflow configuration
    mlflow_tracking_uri: Optional[str] = Field(None, env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("trading-bot", env="MLFLOW_EXPERIMENT_NAME")

    # Database configuration
    database_url: Optional[str] = Field(None, env="DATABASE_URL")

    # Monitoring
    prometheus_port: int = Field(8000, env="PROMETHEUS_PORT")
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @field_validator("telegram_bot_token")
    @classmethod
    def validate_telegram_token(cls, v):
        """Validate Telegram bot token format."""
        if v and not v.count(":") == 1:
            raise ValueError("Invalid Telegram bot token format")
        return v

    @field_validator("telegram_chat_id")
    @classmethod
    def validate_chat_id(cls, v):
        """Validate Telegram chat ID format."""
        if v and not (v.startswith("-") or v.isdigit()):
            raise ValueError("Invalid Telegram chat ID format")
        return v


def validate_environment() -> EnvironmentConfig:
    """Validate environment variables."""
    return EnvironmentConfig()


def validate_trading_config(config_dict: Dict[str, Any]) -> TradingConfig:
    """Validate trading configuration."""
    return TradingConfig(**config_dict)


def get_config_schema() -> Dict[str, Any]:
    """Get JSON schema for configuration validation."""
    return TradingConfig.schema()
