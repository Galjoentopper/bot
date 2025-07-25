"""Tests for portfolio management functionality."""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from paper_trader.portfolio.portfolio_manager import PortfolioManager, Position, Trade


class TestPortfolioManager:
    """Test cases for PortfolioManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def portfolio_manager(self, temp_dir):
        """Create a PortfolioManager instance for testing."""
        return PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
    
    def test_initialization(self, portfolio_manager):
        """Test PortfolioManager initialization."""
        assert portfolio_manager.initial_capital == 10000.0
        assert portfolio_manager.current_capital == 10000.0
        assert portfolio_manager.positions == {}
        assert portfolio_manager.trades == []
        assert portfolio_manager.total_pnl == 0.0
        assert portfolio_manager.max_drawdown == 0.0
        assert portfolio_manager.winning_trades == 0
        assert portfolio_manager.losing_trades == 0
        assert portfolio_manager.total_trades == 0
    
    def test_open_position_success(self, portfolio_manager):
        """Test successful position opening."""
        result = portfolio_manager.open_position(
            symbol='BTC-EUR',
            entry_price=50000.0,
            quantity=0.1,
            take_profit=52000.0,
            stop_loss=48000.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        assert result is True
        assert 'BTC-EUR' in portfolio_manager.positions
        assert len(portfolio_manager.positions['BTC-EUR']) == 1
        
        position = portfolio_manager.positions['BTC-EUR'][0]
        assert position.symbol == 'BTC-EUR'
        assert position.entry_price == 50000.0
        assert position.quantity == 0.1
        assert position.position_value == 5000.0
        assert position.confidence == 0.8
        assert position.signal_strength == 'STRONG'
        
        # Check capital reduction
        assert portfolio_manager.current_capital == 5000.0
    
    def test_open_position_insufficient_capital(self, portfolio_manager):
        """Test position opening with insufficient capital."""
        result = portfolio_manager.open_position(
            symbol='BTC-EUR',
            entry_price=50000.0,
            quantity=1.0,  # Would cost 50,000, more than available capital
            take_profit=52000.0,
            stop_loss=48000.0
        )
        
        assert result is False
        assert 'BTC-EUR' not in portfolio_manager.positions
        assert portfolio_manager.current_capital == 10000.0
    
    def test_max_positions_per_symbol(self, temp_dir):
        """Test maximum positions per symbol limit."""
        portfolio_manager = PortfolioManager(
            initial_capital=10000.0, 
            log_dir=temp_dir,
            max_positions_per_symbol=2
        )
        
        # Open first position
        result1 = portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.05,
            take_profit=52000.0, stop_loss=48000.0
        )
        assert result1 is True
        
        # Open second position
        result2 = portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=51000.0, quantity=0.05,
            take_profit=53000.0, stop_loss=49000.0
        )
        assert result2 is True
        
        # Third position should fail
        result3 = portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=52000.0, quantity=0.05,
            take_profit=54000.0, stop_loss=50000.0
        )
        assert result3 is False
        
        assert len(portfolio_manager.positions['BTC-EUR']) == 2
    
    def test_close_position_profit(self, portfolio_manager):
        """Test closing a position with profit."""
        # Open position
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.1,
            take_profit=52000.0, stop_loss=48000.0
        )
        
        position = portfolio_manager.positions['BTC-EUR'][0]
        
        # Close position at higher price
        result = portfolio_manager.close_position(
            symbol='BTC-EUR',
            position=position,
            exit_price=51000.0,
            reason='TAKE_PROFIT'
        )
        
        assert result is True
        assert 'BTC-EUR' not in portfolio_manager.positions
        assert len(portfolio_manager.trades) == 1
        
        trade = portfolio_manager.trades[0]
        assert trade.symbol == 'BTC-EUR'
        assert trade.entry_price == 50000.0
        assert trade.exit_price == 51000.0
        assert trade.pnl == 100.0  # (51000 - 50000) * 0.1
        assert trade.pnl_pct == 0.02  # 2% profit
        assert trade.exit_reason == 'TAKE_PROFIT'
        
        # Check statistics
        assert portfolio_manager.total_trades == 1
        assert portfolio_manager.winning_trades == 1
        assert portfolio_manager.losing_trades == 0
        assert portfolio_manager.total_pnl == 100.0
        
        # Check capital (initial position value + profit)
        assert portfolio_manager.current_capital == 5100.0
    
    def test_close_position_loss(self, portfolio_manager):
        """Test closing a position with loss."""
        # Open position
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.1,
            take_profit=52000.0, stop_loss=48000.0
        )
        
        position = portfolio_manager.positions['BTC-EUR'][0]
        
        # Close position at lower price
        result = portfolio_manager.close_position(
            symbol='BTC-EUR',
            position=position,
            exit_price=49000.0,
            reason='STOP_LOSS'
        )
        
        assert result is True
        trade = portfolio_manager.trades[0]
        assert trade.pnl == -100.0  # (49000 - 50000) * 0.1
        assert trade.pnl_pct == -0.02  # 2% loss
        
        # Check statistics
        assert portfolio_manager.winning_trades == 0
        assert portfolio_manager.losing_trades == 1
        assert portfolio_manager.total_pnl == -100.0
        
        # Check capital (initial position value - loss)
        assert portfolio_manager.current_capital == 4900.0
    
    def test_get_position_value(self, portfolio_manager):
        """Test position value calculation."""
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.1,
            take_profit=52000.0, stop_loss=48000.0
        )
        
        # Test with current price
        value = portfolio_manager.get_position_value('BTC-EUR', 51000.0)
        assert value == 5100.0  # 0.1 * 51000
        
        # Test with non-existent symbol
        value = portfolio_manager.get_position_value('ETH-EUR', 3000.0)
        assert value == 0.0
    
    def test_get_position_pnl(self, portfolio_manager):
        """Test position P&L calculation."""
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.1,
            take_profit=52000.0, stop_loss=48000.0
        )
        
        # Test with profitable price
        pnl, pnl_pct = portfolio_manager.get_position_pnl('BTC-EUR', 51000.0)
        assert pnl == 100.0
        assert pnl_pct == 0.02
        
        # Test with losing price
        pnl, pnl_pct = portfolio_manager.get_position_pnl('BTC-EUR', 49000.0)
        assert pnl == -100.0
        assert pnl_pct == -0.02
        
        # Test with non-existent symbol
        pnl, pnl_pct = portfolio_manager.get_position_pnl('ETH-EUR', 3000.0)
        assert pnl == 0.0
        assert pnl_pct == 0.0
    
    def test_get_total_capital_value(self, portfolio_manager):
        """Test total capital value calculation."""
        # Initial state
        assert portfolio_manager.get_total_capital_value() == 10000.0
        
        # After opening position
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.1,
            take_profit=52000.0, stop_loss=48000.0
        )
        
        # Without current prices (using entry values)
        assert portfolio_manager.get_total_capital_value() == 10000.0
        
        # With current prices
        current_prices = {'BTC-EUR': 51000.0}
        total_value = portfolio_manager.get_total_capital_value(current_prices)
        assert total_value == 10100.0  # 5000 cash + 5100 position value
    
    def test_get_portfolio_summary(self, portfolio_manager):
        """Test portfolio summary generation."""
        # Open a position
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.1,
            take_profit=52000.0, stop_loss=48000.0
        )
        
        # Close a position for trade history
        position = portfolio_manager.positions['BTC-EUR'][0]
        portfolio_manager.close_position(
            symbol='BTC-EUR', position=position, exit_price=51000.0,
            reason='TAKE_PROFIT'
        )
        
        # Open another position
        portfolio_manager.open_position(
            symbol='ETH-EUR', entry_price=3000.0, quantity=1.0,
            take_profit=3200.0, stop_loss=2800.0
        )
        
        current_prices = {'ETH-EUR': 3100.0}
        summary = portfolio_manager.get_portfolio_summary(current_prices)
        
        assert summary['initial_capital'] == 10000.0
        assert summary['total_trades'] == 1
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 0
        assert summary['win_rate'] == 1.0
        assert summary['realized_pnl'] == 100.0
        assert summary['num_positions'] == 1
        assert len(summary['positions']) == 1
        
        # Check position summary
        pos_summary = summary['positions'][0]
        assert pos_summary['symbol'] == 'ETH-EUR'
        assert pos_summary['entry_price'] == 3000.0
        assert pos_summary['current_price'] == 3100.0
        assert pos_summary['pnl'] == 100.0
        assert pos_summary['pnl_pct'] == pytest.approx(0.0333, rel=1e-3)
    
    def test_drawdown_calculation(self, portfolio_manager):
        """Test maximum drawdown calculation."""
        # Open and close positions to create drawdown scenario
        
        # Profitable trade - increases peak
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.1,
            take_profit=52000.0, stop_loss=48000.0
        )
        position = portfolio_manager.positions['BTC-EUR'][0]
        portfolio_manager.close_position(
            symbol='BTC-EUR', position=position, exit_price=52000.0,
            reason='TAKE_PROFIT'
        )
        
        # Capital should be 10200 (10000 + 200 profit)
        assert portfolio_manager.current_capital == 5200.0
        assert portfolio_manager.peak_capital > 10000.0
        
        # Losing trade - creates drawdown
        portfolio_manager.open_position(
            symbol='BTC-EUR', entry_price=50000.0, quantity=0.2,
            take_profit=52000.0, stop_loss=48000.0
        )
        position = portfolio_manager.positions['BTC-EUR'][0]
        portfolio_manager.close_position(
            symbol='BTC-EUR', position=position, exit_price=47000.0,
            reason='STOP_LOSS'
        )
        
        # Should have some drawdown
        assert portfolio_manager.max_drawdown > 0
    
    def test_position_class(self):
        """Test Position dataclass functionality."""
        entry_time = datetime.now()
        position = Position(
            symbol='BTC-EUR',
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
            take_profit=52000.0,
            stop_loss=48000.0,
            position_value=5000.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        assert position.symbol == 'BTC-EUR'
        assert position.entry_price == 50000.0
        assert position.highest_price == 50000.0  # Should be set in __post_init__
    
    def test_trade_class(self):
        """Test Trade dataclass functionality."""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=1)
        
        trade = Trade(
            symbol='BTC-EUR',
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=100.0,
            pnl_pct=0.02,
            exit_reason='TAKE_PROFIT',
            hold_time_hours=1.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        assert trade.symbol == 'BTC-EUR'
        assert trade.pnl == 100.0
        assert trade.hold_time_hours == 1.0
    
    def test_close_nonexistent_position(self, portfolio_manager):
        """Test closing a position that doesn't exist."""
        fake_position = Position(
            symbol='BTC-EUR',
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
            take_profit=52000.0,
            stop_loss=48000.0,
            position_value=5000.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        result = portfolio_manager.close_position(
            symbol='BTC-EUR',
            position=fake_position,
            exit_price=51000.0
        )
        
        assert result is False
        assert len(portfolio_manager.trades) == 0