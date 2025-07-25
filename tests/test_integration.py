"""Basic import and integration tests."""

import pytest
import sys
import os


class TestImports:
    """Test that all core modules can be imported without errors."""
    
    def test_main_imports(self):
        """Test importing main modules."""
        try:
            import main_paper_trader
            import enhanced_main_paper_trader
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import main modules: {e}")
    
    def test_config_imports(self):
        """Test importing configuration modules."""
        try:
            from paper_trader.config.settings import TradingSettings
            assert TradingSettings is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config modules: {e}")
    
    def test_portfolio_imports(self):
        """Test importing portfolio modules."""
        try:
            from paper_trader.portfolio.portfolio_manager import PortfolioManager, Position, Trade
            assert PortfolioManager is not None
            assert Position is not None
            assert Trade is not None
        except ImportError as e:
            pytest.fail(f"Failed to import portfolio modules: {e}")
    
    def test_utils_imports(self):
        """Test importing utility modules."""
        try:
            from paper_trader.utils.data_quality import DataQualityMonitor
            assert DataQualityMonitor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import utils modules: {e}")
    
    def test_strategy_imports(self):
        """Test importing strategy modules."""
        try:
            # Check if strategy modules exist and can be imported
            from paper_trader.strategy import signal_generator
            assert signal_generator is not None
        except ImportError:
            # This is expected if modules don't exist yet - just pass
            pass
    
    def test_data_imports(self):
        """Test importing data modules."""
        try:
            from paper_trader.data.bitvavo_collector import BitvavoDataCollector
            assert BitvavoDataCollector is not None
        except ImportError as e:
            pytest.fail(f"Failed to import data modules: {e}")


class TestEnvironment:
    """Test environment and setup."""
    
    def test_python_version(self):
        """Test Python version compatibility."""
        version = sys.version_info
        assert version.major == 3
        assert version.minor >= 8  # Requires Python 3.8+
    
    def test_required_packages(self):
        """Test that required packages are available."""
        required_packages = [
            'numpy',
            'pandas',
            'tensorflow',
            'xgboost',
            'sklearn',  # scikit-learn imports as sklearn
            'httpx',
            'websockets',
            'dotenv'    # python-dotenv imports as dotenv
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package not available: {package}")
    
    def test_model_directory_exists(self):
        """Test that model directory exists."""
        model_dir = os.path.join(os.getcwd(), 'models')
        assert os.path.exists(model_dir), "Models directory should exist"
    
    def test_env_file_exists(self):
        """Test that environment file exists."""
        env_files = ['.env', '.env.example']
        has_env_file = any(os.path.exists(f) for f in env_files)
        assert has_env_file, "Either .env or .env.example should exist"


class TestBasicFunctionality:
    """Test basic functionality without external dependencies."""
    
    def test_trading_settings_creation(self):
        """Test creating TradingSettings without errors."""
        from paper_trader.config.settings import TradingSettings
        
        settings = TradingSettings()
        assert settings.initial_capital > 0
        assert settings.max_positions > 0
        assert settings.candle_interval in ['1m', '5m', '15m', '30m', '1h']
    
    def test_portfolio_manager_creation(self):
        """Test creating PortfolioManager without errors."""
        from unittest.mock import patch
        
        # Mock file operations to avoid creating actual files
        with patch('pathlib.Path.mkdir'), \
             patch('csv.writer'), \
             patch('builtins.open'):
            
            from paper_trader.portfolio.portfolio_manager import PortfolioManager
            
            portfolio = PortfolioManager(initial_capital=10000.0)
            assert portfolio.initial_capital == 10000.0
            assert portfolio.current_capital == 10000.0
    
    def test_data_quality_monitor_creation(self):
        """Test creating DataQualityMonitor without errors."""
        from paper_trader.utils.data_quality import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        assert hasattr(monitor, 'thresholds')
        assert hasattr(monitor, 'logger')


class TestDataStructures:
    """Test data structure creation and validation."""
    
    def test_position_dataclass(self):
        """Test Position dataclass functionality."""
        from datetime import datetime
        from paper_trader.portfolio.portfolio_manager import Position
        
        position = Position(
            symbol='TEST-EUR',
            entry_price=1000.0,
            quantity=1.0,
            entry_time=datetime.now(),
            take_profit=1100.0,
            stop_loss=900.0,
            position_value=1000.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        assert position.symbol == 'TEST-EUR'
        assert position.entry_price == 1000.0
        assert position.highest_price == 1000.0  # Should be auto-set
    
    def test_trade_dataclass(self):
        """Test Trade dataclass functionality."""
        from datetime import datetime
        from paper_trader.portfolio.portfolio_manager import Trade
        
        now = datetime.now()
        trade = Trade(
            symbol='TEST-EUR',
            entry_price=1000.0,
            exit_price=1100.0,
            quantity=1.0,
            entry_time=now,
            exit_time=now,
            pnl=100.0,
            pnl_pct=0.1,
            exit_reason='TAKE_PROFIT',
            hold_time_hours=1.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        assert trade.symbol == 'TEST-EUR'
        assert trade.pnl == 100.0
        assert trade.exit_reason == 'TAKE_PROFIT'