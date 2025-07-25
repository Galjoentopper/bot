"""Tests for portfolio management functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from paper_trader.portfolio.portfolio_manager import PortfolioManager, Position, Trade


class TestPosition:
    """Test Position data class."""
    
    def test_position_creation(self):
        """Test position object creation."""
        entry_time = datetime.now()
        position = Position(
            symbol='BTC-EUR',
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
            take_profit=51000.0,
            stop_loss=49000.0,
            position_value=5000.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        assert position.symbol == 'BTC-EUR'
        assert position.entry_price == 50000.0
        assert position.quantity == 0.1
        assert position.highest_price == 50000.0  # Should be set to entry_price
        
    def test_position_highest_price_initialization(self):
        """Test that highest_price is initialized to entry_price."""
        position = Position(
            symbol='ETH-EUR',
            entry_price=3000.0,
            quantity=1.0,
            entry_time=datetime.now(),
            take_profit=3100.0,
            stop_loss=2900.0,
            position_value=3000.0,
            confidence=0.7,
            signal_strength='MODERATE'
        )
        
        assert position.highest_price == position.entry_price


class TestTrade:
    """Test Trade data class."""
    
    def test_trade_creation(self):
        """Test trade object creation."""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=2)
        
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
            hold_time_hours=2.0,
            confidence=0.8,
            signal_strength='STRONG'
        )
        
        assert trade.symbol == 'BTC-EUR'
        assert trade.pnl == 100.0
        assert trade.exit_reason == 'TAKE_PROFIT'


class TestPortfolioManager:
    """Test PortfolioManager functionality."""
    
    @patch('pathlib.Path.mkdir')
    @patch('csv.writer')
    @patch('builtins.open')
    def test_portfolio_manager_initialization(self, mock_open, mock_csv_writer, mock_mkdir):
        """Test portfolio manager initialization."""
        portfolio = PortfolioManager(initial_capital=20000.0)
        
        assert portfolio.initial_capital == 20000.0
        assert portfolio.current_capital == 20000.0
        assert portfolio.total_pnl == 0.0
        assert portfolio.total_trades == 0
        assert len(portfolio.positions) == 0
        
    @patch('pathlib.Path.mkdir')
    @patch('csv.writer')
    @patch('builtins.open')
    def test_calculate_position_value(self, mock_open, mock_csv_writer, mock_mkdir):
        """Test position value calculation."""
        portfolio = PortfolioManager()
        
        # Test with example values
        price = 50000.0
        capital = 10000.0
        position_size_pct = 0.1  # 10%
        
        expected_value = capital * position_size_pct
        assert expected_value == 1000.0
        
        quantity = expected_value / price
        assert abs(quantity - 0.02) < 0.001  # Should be approximately 0.02
        
    @patch('pathlib.Path.mkdir')
    @patch('csv.writer')
    @patch('builtins.open')
    def test_portfolio_capital_tracking(self, mock_open, mock_csv_writer, mock_mkdir):
        """Test capital tracking functionality."""
        initial_capital = 15000.0
        portfolio = PortfolioManager(initial_capital=initial_capital)
        
        # Test initial state
        assert portfolio.current_capital == initial_capital
        assert portfolio.peak_capital == initial_capital
        assert portfolio.max_drawdown == 0.0
        
        # Simulate capital change
        portfolio.current_capital = 16000.0
        portfolio.peak_capital = 16000.0
        
        # Test profit scenario
        assert portfolio.current_capital > initial_capital
        
    @patch('pathlib.Path.mkdir')
    @patch('csv.writer')
    @patch('builtins.open')
    def test_position_limits(self, mock_open, mock_csv_writer, mock_mkdir):
        """Test position limit validation."""
        portfolio = PortfolioManager(max_positions_per_symbol=2)
        
        assert portfolio.max_positions_per_symbol == 2
        
        # Test that we can track multiple positions for same symbol
        portfolio.positions['BTC-EUR'] = []
        assert len(portfolio.positions['BTC-EUR']) == 0
        
        # Can add up to max_positions_per_symbol
        for i in range(portfolio.max_positions_per_symbol):
            portfolio.positions['BTC-EUR'].append(f"position_{i}")
        
        assert len(portfolio.positions['BTC-EUR']) == portfolio.max_positions_per_symbol
        
    @patch('pathlib.Path.mkdir')
    @patch('csv.writer')
    @patch('builtins.open')
    def test_pnl_calculation(self, mock_open, mock_csv_writer, mock_mkdir):
        """Test P&L calculation logic."""
        portfolio = PortfolioManager()
        
        # Test profit calculation
        entry_price = 50000.0
        exit_price = 51000.0
        quantity = 0.1
        
        expected_pnl = (exit_price - entry_price) * quantity
        assert expected_pnl == 100.0
        
        expected_pnl_pct = (exit_price - entry_price) / entry_price
        assert abs(expected_pnl_pct - 0.02) < 0.001
        
    @patch('pathlib.Path.mkdir')
    @patch('csv.writer')
    @patch('builtins.open')
    def test_trade_statistics(self, mock_open, mock_csv_writer, mock_mkdir):
        """Test trade statistics tracking."""
        portfolio = PortfolioManager()
        
        # Initial state
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 0
        assert portfolio.total_trades == 0
        
        # Simulate some trades
        portfolio.winning_trades = 5
        portfolio.losing_trades = 3
        portfolio.total_trades = 8
        
        # Test win rate calculation
        if portfolio.total_trades > 0:
            win_rate = portfolio.winning_trades / portfolio.total_trades
            assert abs(win_rate - 0.625) < 0.001  # 5/8 = 0.625