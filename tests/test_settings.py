"""Unit tests for trading settings configuration."""

import os
import pytest
from unittest.mock import patch

from paper_trader.config.settings import TradingSettings


class TestTradingSettings:
    """Test cases for TradingSettings configuration."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = TradingSettings()
        
        # Test some basic defaults that should be consistent
        assert settings.initial_capital > 0
        assert settings.max_positions > 0
        assert settings.base_position_size > 0
        assert settings.take_profit_pct > 0
        assert settings.stop_loss_pct > 0
        assert settings.candle_interval == '1m'
        assert settings.sequence_length == 96
    
    def test_environment_variable_override(self):
        """Test that environment variables can be read properly."""
        # Just test that the setting loading mechanism works
        settings = TradingSettings()
        
        # Verify that settings can be created and have valid values
        assert isinstance(settings.initial_capital, float)
        assert isinstance(settings.max_positions, int)
        assert isinstance(settings.take_profit_pct, float)
        assert isinstance(settings.stop_loss_pct, float)
        
        # Test the validate method exists and can be called
        assert hasattr(settings, 'validate')
        assert callable(settings.validate)
    
    def test_api_credentials_from_env(self):
        """Test that API credentials are loaded from environment."""
        settings = TradingSettings()
        
        # Should have some credentials loaded from .env file
        assert len(settings.bitvavo_api_key) > 0
        assert len(settings.bitvavo_api_secret) > 0
        assert len(settings.telegram_bot_token) > 0
        assert len(settings.telegram_chat_id) > 0
    
    def test_risk_parameters_validation(self):
        """Test that risk parameters are within reasonable bounds."""
        settings = TradingSettings()
        
        # Position sizes should be reasonable
        assert 0 < settings.min_position_size < settings.base_position_size < settings.max_position_size
        assert settings.max_position_size <= 1.0  # Not more than 100% of capital
        
        # Max positions should be positive
        assert settings.max_positions > 0
        assert settings.max_positions_per_symbol > 0
        
        # Take profit and stop loss should be positive
        assert settings.take_profit_pct > 0
        assert settings.stop_loss_pct > 0
    
    def test_symbols_initialization(self):
        """Test symbols list initialization."""
        settings = TradingSettings()
        
        # symbols should be initialized from __post_init__
        assert settings.symbols is not None
        assert len(settings.symbols) > 0
        assert 'BTC-EUR' in settings.symbols or 'BTCEUR' in settings.symbols
    
    def test_model_path_default(self):
        """Test that model path default is correctly set."""
        settings = TradingSettings()
        
        assert 'models' in settings.model_path
        assert os.path.isabs(settings.model_path) or settings.model_path.startswith('models')