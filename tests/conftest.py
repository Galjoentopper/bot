"""Pytest configuration and fixtures."""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.config_schema import EnvironmentConfig, TradingConfig
from src.core.structured_logger import LoggerType, get_logger


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample trading configuration for tests."""
    return {
        "symbols": ["BTCEUR", "ETHEUR"],
        "interval": "30m",
        "lookback_days": 365,
        "model_weights": {"lightgbm": 0.55, "gru": 0.35, "ppo": 0.1},
        "thresholds": {
            "base_buy_threshold": 0.6,
            "base_sell_threshold": -0.6,
            "confidence_multiplier": 1.2,
            "volatility_adjustment": True,
        },
        "per_symbol": {
            "BTCEUR": {"buy_threshold": 0.6, "sell_threshold": -0.6, "max_position_size": 1000.0},
            "ETHEUR": {"buy_threshold": 0.65, "sell_threshold": -0.65, "max_position_size": 500.0},
        },
        "risk_management": {
            "max_drawdown_pct": 20.0,
            "position_size_pct": 2.0,
            "stop_loss_pct": 5.0,
            "take_profit_pct": 10.0,
            "max_open_positions": 5,
        },
        "paper_trading": True,
        "log_level": "INFO",
    }


@pytest.fixture
def trading_config(sample_config: Dict[str, Any]) -> TradingConfig:
    """Trading configuration object for tests."""
    return TradingConfig(**sample_config)


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Sample market data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="30T")
    np.random.seed(42)

    # Generate realistic OHLCV data
    base_price = 50000
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)

    # Generate OHLCV from prices
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        close_price = price
        volume = np.random.randint(100, 10000)

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": max(open_price, high, close_price),
                "low": min(open_price, low, close_price),
                "close": close_price,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Sample feature data for testing."""
    np.random.seed(42)
    n_samples = 100

    features = {
        "rsi_14": np.random.uniform(0, 100, n_samples),
        "macd": np.random.normal(0, 0.1, n_samples),
        "bb_upper": np.random.uniform(1.01, 1.05, n_samples),
        "bb_lower": np.random.uniform(0.95, 0.99, n_samples),
        "volume_sma": np.random.uniform(1000, 10000, n_samples),
        "price_change": np.random.normal(0, 0.02, n_samples),
    }

    return pd.DataFrame(features)


@pytest.fixture
def mock_logger():
    """Mock logger for tests."""
    return Mock(spec=get_logger("test", LoggerType.APPLICATION))


@pytest.fixture
def mock_model():
    """Mock ML model for tests."""
    model = Mock()
    model.predict.return_value = np.array([0.7, 0.8, 0.2])
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.8, 0.2]])
    return model


@pytest.fixture
def mock_data_loader():
    """Mock data loader for tests."""
    loader = Mock()
    loader.load_data.return_value = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="30T"),
            "open": np.random.uniform(50000, 51000, 100),
            "high": np.random.uniform(50500, 51500, 100),
            "low": np.random.uniform(49500, 50500, 100),
            "close": np.random.uniform(50000, 51000, 100),
            "volume": np.random.randint(100, 10000, 100),
        }
    )
    return loader


@pytest.fixture
def mock_telegram_client():
    """Mock Telegram client for tests."""
    client = Mock()
    client.send_message.return_value = True
    client.is_connected.return_value = True
    return client


@pytest.fixture(autouse=True)
def patch_environment():
    """Patch environment variables for tests."""
    with patch.dict(
        "os.environ",
        {
            "TELEGRAM_BOT_TOKEN": "test_token:test_token",
            "TELEGRAM_CHAT_ID": "123456789",
            "LOG_LEVEL": "DEBUG",
        },
    ):
        yield


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        "data_loading": {"max_duration_ms": 1000, "target_duration_ms": 500},
        "feature_extraction": {"max_duration_ms": 2000, "target_duration_ms": 1000},
        "model_prediction": {"max_duration_ms": 500, "target_duration_ms": 200},
        "trade_execution": {"max_duration_ms": 100, "target_duration_ms": 50},
    }


class AsyncContextManager:
    """Helper for async context manager testing."""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Async context manager factory for tests."""
    return AsyncContextManager
