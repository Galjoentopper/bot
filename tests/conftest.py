"""Pytest configuration and fixtures for the trading bot tests."""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock

# Ensure we're in test mode to avoid any real API calls
os.environ['TESTING'] = 'true'
os.environ['BITVAVO_API_KEY'] = 'test_key'
os.environ['BITVAVO_API_SECRET'] = 'test_secret'
os.environ['TELEGRAM_BOT_TOKEN'] = 'test_token'
os.environ['TELEGRAM_CHAT_ID'] = 'test_chat_id'


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_portfolio_settings():
    """Mock portfolio settings for testing."""
    return {
        'initial_capital': 10000.0,
        'max_positions': 5,
        'max_positions_per_symbol': 1,
        'base_position_size': 0.1,
        'take_profit_pct': 0.02,
        'stop_loss_pct': 0.01,
        'trailing_stop_pct': 0.006,
    }


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing."""
    return {
        'symbol': 'BTCEUR',
        'entry_price': 50000.0,
        'exit_price': 51000.0,
        'quantity': 0.1,
        'entry_time': datetime.now() - timedelta(hours=1),
        'exit_time': datetime.now(),
        'pnl': 100.0,
        'pnl_pct': 2.0,
        'exit_reason': 'take_profit',
        'hold_time_hours': 1.0,
        'confidence': 0.85,
        'signal_strength': 'strong'
    }


@pytest.fixture
def sample_position_data():
    """Sample position data for testing."""
    return {
        'symbol': 'BTCEUR',
        'entry_price': 50000.0,
        'quantity': 0.1,
        'entry_time': datetime.now(),
        'take_profit': 51000.0,
        'stop_loss': 49500.0,
        'position_value': 5000.0,
        'confidence': 0.8,
        'signal_strength': 'strong'
    }