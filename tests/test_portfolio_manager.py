"""Unit tests for portfolio management functionality."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from paper_trader.portfolio.portfolio_manager import PortfolioManager, Position, Trade


class TestPosition:
    """Test cases for Position dataclass."""
    
    def test_position_creation(self, sample_position_data):
        """Test creating a position with valid data."""
        position = Position(**sample_position_data)
        
        assert position.symbol == 'BTCEUR'
        assert position.entry_price == 50000.0
        assert position.quantity == 0.1
        assert position.take_profit == 51000.0
        assert position.stop_loss == 49500.0
        assert position.confidence == 0.8
        assert position.signal_strength == 'strong'
    
    def test_position_highest_price_initialization(self, sample_position_data):
        """Test that highest_price is initialized to entry_price."""
        position = Position(**sample_position_data)
        
        assert position.highest_price == position.entry_price
    
    def test_position_with_custom_highest_price(self, sample_position_data):
        """Test position creation with custom highest_price."""
        sample_position_data['highest_price'] = 50500.0
        position = Position(**sample_position_data)
        
        assert position.highest_price == 50500.0


class TestTrade:
    """Test cases for Trade dataclass."""
    
    def test_trade_creation(self, sample_trade_data):
        """Test creating a trade with valid data."""
        trade = Trade(**sample_trade_data)
        
        assert trade.symbol == 'BTCEUR'
        assert trade.entry_price == 50000.0
        assert trade.exit_price == 51000.0
        assert trade.quantity == 0.1
        assert trade.pnl == 100.0
        assert trade.pnl_pct == 2.0
        assert trade.exit_reason == 'take_profit'
        assert trade.hold_time_hours == 1.0


class TestPortfolioManager:
    """Test cases for PortfolioManager class."""
    
    def test_portfolio_initialization(self, temp_dir):
        """Test portfolio manager initialization."""
        portfolio = PortfolioManager(
            initial_capital=10000.0,
            log_dir=temp_dir
        )
        
        assert portfolio.initial_capital == 10000.0
        assert portfolio.current_capital == 10000.0
        assert portfolio.get_available_capital() == 10000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
    
    def test_open_position(self, temp_dir):
        """Test opening a position."""
        portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
        
        # Open a position
        success = portfolio.open_position(
            symbol='BTCEUR',
            entry_price=50000.0,
            quantity=0.1,
            take_profit=51000.0,
            stop_loss=49500.0,
            confidence=0.8,
            signal_strength='strong'
        )
        
        assert success is True
        assert 'BTCEUR' in portfolio.positions
        assert len(portfolio.positions['BTCEUR']) == 1
        
        position = portfolio.positions['BTCEUR'][0]
        assert position.symbol == 'BTCEUR'
        assert position.entry_price == 50000.0
        assert position.quantity == 0.1
    
    def test_close_position_profit(self, temp_dir):
        """Test closing a position with profit."""
        portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
        
        # Open position first
        portfolio.open_position(
            symbol='BTCEUR',
            entry_price=50000.0,
            quantity=0.1,
            take_profit=51000.0,
            stop_loss=49500.0
        )
        
        # Get the position that was created
        position = portfolio.positions['BTCEUR'][0]
        
        # Close at higher price for profit
        exit_price = 51000.0
        success = portfolio.close_position('BTCEUR', position, exit_price, 'take_profit')
        
        assert success is True
        assert len(portfolio.positions.get('BTCEUR', [])) == 0  # Position should be removed
        assert len(portfolio.trades) == 1  # Trade should be recorded
        
        trade = portfolio.trades[0]
        assert trade.exit_price == exit_price
        assert trade.pnl > 0  # Should be profitable
    
    def test_close_position_loss(self, temp_dir):
        """Test closing a position with loss."""
        portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
        
        # Open position first
        portfolio.open_position(
            symbol='BTCEUR',
            entry_price=50000.0,
            quantity=0.1,
            take_profit=51000.0,
            stop_loss=49500.0
        )
        
        # Get the position that was created
        position = portfolio.positions['BTCEUR'][0]
        
        # Close at lower price for loss
        exit_price = 49000.0
        success = portfolio.close_position('BTCEUR', position, exit_price, 'stop_loss')
        
        assert success is True
        assert len(portfolio.positions.get('BTCEUR', [])) == 0  # Position should be removed
        assert len(portfolio.trades) == 1  # Trade should be recorded
        
        trade = portfolio.trades[0]
        assert trade.exit_price == exit_price
        assert trade.pnl < 0  # Should be a loss
        assert trade.exit_reason == 'stop_loss'
    
    def test_portfolio_capital_calculation(self, temp_dir):
        """Test portfolio capital calculations."""
        portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
        
        # Open a position
        portfolio.open_position(
            symbol='BTCEUR',
            entry_price=50000.0,
            quantity=0.1,
            take_profit=51000.0,
            stop_loss=49500.0
        )
        
        # Test capital calculations
        total_capital = portfolio.get_total_capital_value()
        available_capital = portfolio.get_available_capital()
        invested_capital = portfolio.get_invested_capital()
        
        assert isinstance(total_capital, float)
        assert isinstance(available_capital, float)
        assert isinstance(invested_capital, float)
        assert total_capital >= 0
        assert available_capital >= 0
        assert invested_capital >= 0
    
    def test_portfolio_summary(self, temp_dir):
        """Test portfolio summary generation."""
        portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
        
        # Open a position
        portfolio.open_position(
            symbol='BTCEUR',
            entry_price=50000.0,
            quantity=0.1,
            take_profit=51000.0,
            stop_loss=49500.0
        )
        
        summary = portfolio.get_portfolio_summary()
        
        assert isinstance(summary, dict)
        assert 'total_capital' in summary
        assert 'available_capital' in summary
        assert 'invested_capital' in summary
        assert 'num_positions' in summary
        
        # Verify basic values
        assert summary['num_positions'] == 1
        assert summary['total_capital'] > 0
    
    def test_risk_management_max_positions(self, temp_dir):
        """Test max positions per symbol validation."""
        portfolio = PortfolioManager(
            initial_capital=10000.0, 
            log_dir=temp_dir,
            max_positions_per_symbol=1
        )
        
        # Open first position - should succeed
        success1 = portfolio.open_position(
            symbol='BTCEUR',
            entry_price=50000.0,
            quantity=0.1,
            take_profit=51000.0,
            stop_loss=49500.0
        )
        assert success1 is True
        
        # Try to open second position for same symbol - should fail
        success2 = portfolio.open_position(
            symbol='BTCEUR',
            entry_price=50100.0,
            quantity=0.1,
            take_profit=51100.0,
            stop_loss=49600.0
        )
        assert success2 is False
    
    def test_csv_files_creation(self, temp_dir):
        """Test that CSV files are created during initialization."""
        portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
        
        # Check that files are created
        trades_file = Path(temp_dir) / 'trades.csv'
        portfolio_file = Path(temp_dir) / 'portfolio.csv'
        
        assert trades_file.exists()
        assert portfolio_file.exists()
        
        # Check headers exist
        with open(trades_file, 'r') as f:
            header = f.readline().strip()
            assert 'symbol' in header
            assert 'entry_price' in header
            assert 'exit_price' in header
    
    def test_recent_trades(self, temp_dir):
        """Test getting recent trades."""
        portfolio = PortfolioManager(initial_capital=10000.0, log_dir=temp_dir)
        
        # Start with no trades
        recent_trades = portfolio.get_recent_trades(limit=5)
        assert len(recent_trades) == 0
        
        # Add a trade by opening and closing position
        portfolio.open_position('BTCEUR', 50000.0, 0.1, 51000.0, 49500.0)
        position = portfolio.positions['BTCEUR'][0]
        portfolio.close_position('BTCEUR', position, 51000.0, 'take_profit')
        
        # Should have one trade now
        recent_trades = portfolio.get_recent_trades(limit=5)
        assert len(recent_trades) == 1
        assert recent_trades[0]['symbol'] == 'BTCEUR'
        assert recent_trades[0]['entry_price'] == 50000.0
        assert recent_trades[0]['exit_price'] == 51000.0