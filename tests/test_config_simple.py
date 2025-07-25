"""Tests for trading configuration settings."""

import os
import tempfile
import pytest
from unittest.mock import patch


class TestTradingSettings:
    """Test cases for TradingSettings class."""
    
    def test_trading_settings_import(self):
        """Test that TradingSettings can be imported successfully."""
        # Import here to avoid .env loading issues
        from paper_trader.config.settings import TradingSettings
        assert TradingSettings is not None
    
    def test_default_values_with_clean_env(self):
        """Test default values when environment is clean."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh instance
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            # Test core defaults that should always be present
            assert settings.initial_capital >= 0
            assert settings.max_positions >= 0 
            assert 0 <= settings.base_position_size <= 1
            assert settings.candle_interval in ['1m', '5m', '15m', '1h']
            assert isinstance(settings.symbols, list)
            assert len(settings.symbols) > 0
    
    def test_position_size_calculation(self):
        """Test position size calculation method."""
        from paper_trader.config.settings import TradingSettings
        
        with patch.dict(os.environ, {'BASE_POSITION_SIZE': '0.1'}):
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            # Test calculation based on actual base_position_size
            expected_size = 10000.0 * settings.base_position_size
            assert settings.get_position_size(10000.0) == expected_size
            
            # Test with different amounts
            assert settings.get_position_size(50000.0) == 50000.0 * settings.base_position_size
            assert settings.get_position_size(0.0) == 0.0
    
    def test_validation_method_exists(self):
        """Test that validation method works."""
        from paper_trader.config.settings import TradingSettings
        
        settings = TradingSettings()
        
        # Method should exist and return boolean
        result = settings.validate()
        assert isinstance(result, bool)
    
    def test_environment_variable_override(self):
        """Test that specific environment variables can override defaults."""
        test_env = {
            'INITIAL_CAPITAL': '25000.0',
            'MAX_POSITIONS': '15',
            'CANDLE_INTERVAL': '5m'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            assert settings.initial_capital == 25000.0
            assert settings.max_positions == 15
            assert settings.candle_interval == '5m'
    
    def test_symbols_parsing(self):
        """Test symbols parsing from environment."""
        test_env = {'SYMBOLS': 'BTC-EUR,ETH-EUR,ADA-EUR'}
        
        with patch.dict(os.environ, test_env, clear=False):
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            assert 'BTC-EUR' in settings.symbols
            assert 'ETH-EUR' in settings.symbols
            assert 'ADA-EUR' in settings.symbols
            assert len(settings.symbols) == 3
    
    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing."""
        test_env = {
            'ENABLE_PREDICTION_EXITS': 'False',
            'DYNAMIC_STOP_LOSS_ADJUSTMENT': 'True'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            assert settings.enable_prediction_exits is False
            assert settings.dynamic_stop_loss_adjustment is True
    
    def test_validation_with_missing_credentials(self):
        """Test validation fails with missing credentials."""
        test_env = {
            'BITVAVO_API_KEY': '',
            'BITVAVO_API_SECRET': '',
            'TELEGRAM_BOT_TOKEN': '',
            'TELEGRAM_CHAT_ID': ''
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            # Should fail validation due to missing credentials
            assert settings.validate() is False
    
    def test_validation_with_valid_credentials(self):
        """Test validation succeeds with valid credentials."""
        test_env = {
            'BITVAVO_API_KEY': 'test_key',
            'BITVAVO_API_SECRET': 'test_secret',
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': 'test_chat_id',
            'SYMBOLS': 'BTC-EUR'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            # Should pass validation with valid credentials
            assert settings.validate() is True
    
    def test_invalid_position_size_validation(self):
        """Test validation fails with invalid position size."""
        test_env = {
            'BITVAVO_API_KEY': 'test_key',
            'BITVAVO_API_SECRET': 'test_secret', 
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': 'test_chat_id',
            'BASE_POSITION_SIZE': '1.5',  # Invalid > 1
            'SYMBOLS': 'BTC-EUR'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            import importlib
            import paper_trader.config.settings
            importlib.reload(paper_trader.config.settings)
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            # Should fail validation due to invalid position size
            assert settings.validate() is False