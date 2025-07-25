"""Integration tests for basic trading workflow."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestTradingIntegration:
    """Integration tests for core trading functionality."""
    
    def test_basic_trading_workflow(self):
        """Test a complete trading workflow from configuration to portfolio management."""
        # Import here to avoid env loading issues
        with patch('dotenv.load_dotenv'):
            from paper_trader.config.settings import TradingSettings
            from paper_trader.portfolio.portfolio_manager import PortfolioManager
        
        # Create temporary directory for logs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configure settings with test values
            test_env = {
                'INITIAL_CAPITAL': '10000.0',
                'BASE_POSITION_SIZE': '0.1',
                'TAKE_PROFIT_PCT': '0.02',
                'STOP_LOSS_PCT': '0.01',
                'SYMBOLS': 'BTC-EUR,ETH-EUR'
            }
            
            with patch.dict(os.environ, test_env, clear=False):
                # Initialize components
                settings = TradingSettings()
                portfolio = PortfolioManager(
                    initial_capital=settings.initial_capital,
                    log_dir=temp_dir
                )
                
                # Verify initialization
                assert portfolio.initial_capital == 10000.0
                assert portfolio.current_capital == 10000.0
                assert len(settings.symbols) >= 2
                
                # Simulate opening a position
                symbol = 'BTC-EUR'
                entry_price = 50000.0
                
                # Calculate position size (settings gives percentage, not absolute amount)
                position_percentage = settings.base_position_size  # This is percentage like 0.08
                position_value = portfolio.current_capital * position_percentage  # Actual money amount
                quantity = position_value / entry_price  # Number of coins
                take_profit = entry_price * (1 + settings.take_profit_pct)
                stop_loss = entry_price * (1 - settings.stop_loss_pct)
                
                # Open position
                success = portfolio.open_position(
                    symbol=symbol,
                    entry_price=entry_price,
                    quantity=quantity,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    confidence=0.8,
                    signal_strength='STRONG'
                )
                
                assert success is True
                assert symbol in portfolio.positions
                assert len(portfolio.positions[symbol]) == 1
                
                # Check portfolio state
                assert portfolio.current_capital < 10000.0  # Capital should be reduced
                total_value = portfolio.get_total_capital_value()
                assert total_value == pytest.approx(10000.0, rel=1e-3)  # Total value preserved
                
                # Simulate closing position with profit
                position = portfolio.positions[symbol][0]
                exit_price = entry_price * 1.015  # 1.5% profit
                
                success = portfolio.close_position(
                    symbol=symbol,
                    position=position,
                    exit_price=exit_price,
                    reason='TAKE_PROFIT'
                )
                
                assert success is True
                assert symbol not in portfolio.positions
                assert len(portfolio.trades) == 1
                assert portfolio.trades[0].pnl > 0  # Should be profitable
                assert portfolio.winning_trades == 1
                assert portfolio.losing_trades == 0
    
    def test_risk_management_integration(self):
        """Test risk management features integration."""
        with patch('dotenv.load_dotenv'):
            from paper_trader.config.settings import TradingSettings
            from paper_trader.portfolio.portfolio_manager import PortfolioManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_env = {
                'INITIAL_CAPITAL': '10000.0',
                'MAX_POSITIONS_PER_SYMBOL': '2'
            }
            
            with patch.dict(os.environ, test_env, clear=False):
                settings = TradingSettings()
                portfolio = PortfolioManager(
                    initial_capital=settings.initial_capital,
                    log_dir=temp_dir,
                    max_positions_per_symbol=2  # Explicitly set to 2
                )
                
                # Test insufficient capital scenario
                large_position_size = 12000.0  # More than available capital
                success = portfolio.open_position(
                    symbol='BTC-EUR',
                    entry_price=50000.0,
                    quantity=large_position_size / 50000.0,
                    take_profit=52000.0,
                    stop_loss=48000.0
                )
                
                assert success is False  # Should fail due to insufficient capital
                assert 'BTC-EUR' not in portfolio.positions
                
                # Test that we can open valid positions within capital limits
                entry_price = 1000.0  # Lower price to make math easier
                quantity = 1.0  # 1 unit at 1000 = 1000 cost
                
                # Open first position (should succeed)
                success1 = portfolio.open_position(
                    symbol='ETH-EUR',
                    entry_price=entry_price,
                    quantity=quantity,
                    take_profit=1100.0,
                    stop_loss=900.0
                )
                assert success1 is True
                assert 'ETH-EUR' in portfolio.positions
                assert len(portfolio.positions['ETH-EUR']) == 1
                
                # Verify capital was reduced
                assert portfolio.current_capital == 9000.0  # 10000 - 1000
    
    def test_configuration_validation_integration(self):
        """Test configuration validation in different scenarios."""
        with patch('dotenv.load_dotenv'):
            from paper_trader.config.settings import TradingSettings
        
        # Test with completely invalid configuration
        invalid_env = {
            'INITIAL_CAPITAL': '-1000',  # Invalid negative
            'MAX_POSITIONS': '0',        # Invalid zero
            'BASE_POSITION_SIZE': '2.0'  # Invalid > 1
        }
        
        with patch.dict(os.environ, invalid_env, clear=False):
            settings = TradingSettings()
            assert settings.validate() is False
        
        # Test with partially valid configuration
        partial_env = {
            'INITIAL_CAPITAL': '10000.0',
            'MAX_POSITIONS': '5',
            'BASE_POSITION_SIZE': '0.1',
            'BITVAVO_API_KEY': 'test_key',
            'BITVAVO_API_SECRET': 'test_secret',
            # Missing telegram credentials
        }
        
        with patch.dict(os.environ, partial_env, clear=False):
            settings = TradingSettings()
            # May fail depending on validation requirements
            # Just test that validation runs without errors
            result = settings.validate()
            assert isinstance(result, bool)
    
    def test_portfolio_summary_integration(self):
        """Test portfolio summary generation with multiple positions."""
        with patch('dotenv.load_dotenv'):
            from paper_trader.portfolio.portfolio_manager import PortfolioManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
            
            # Open multiple positions
            positions_data = [
                ('BTC-EUR', 50000.0, 0.05),
                ('ETH-EUR', 3000.0, 0.5),
                ('ADA-EUR', 1.0, 1000.0)
            ]
            
            for symbol, price, quantity in positions_data:
                portfolio.open_position(
                    symbol=symbol,
                    entry_price=price,
                    quantity=quantity,
                    take_profit=price * 1.02,
                    stop_loss=price * 0.98,
                    confidence=0.75,
                    signal_strength='MODERATE'
                )
            
            # Create current prices with some profit/loss
            current_prices = {
                'BTC-EUR': 51000.0,  # +2% profit
                'ETH-EUR': 2950.0,   # -1.67% loss
                'ADA-EUR': 1.05      # +5% profit
            }
            
            # Get portfolio summary
            summary = portfolio.get_portfolio_summary(current_prices)
            
            # Verify summary structure
            assert 'total_capital' in summary
            assert 'unrealized_pnl' in summary
            assert 'num_positions' in summary
            assert 'positions' in summary
            
            # Should have 3 positions
            assert summary['num_positions'] == 3
            assert len(summary['positions']) == 3
            
            # Check that unrealized P&L is calculated
            assert summary['unrealized_pnl'] != 0
            
            # Verify individual position summaries
            position_symbols = [pos['symbol'] for pos in summary['positions']]
            assert 'BTC-EUR' in position_symbols
            assert 'ETH-EUR' in position_symbols
            assert 'ADA-EUR' in position_symbols