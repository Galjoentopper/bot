import pandas as pd
import numpy as np
import time
import datetime
from typing import Dict, List, Optional
import os
import logging

# Import our custom modules
from feature_factory import FeatureFactory
from model_manager import ModelManager
from data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trader_factory.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperTrader:
    """
    Paper trading system that uses multiple models with different time windows
    to make trading decisions using the Feature Factory pattern.
    """
    
    def __init__(
        self, 
        symbol: str, 
        initial_balance: float = 10000.0,
        window_sizes: List[int] = [30, 60, 90],
        model_dir: str = './models'
    ):
        """
        Initialize the paper trader.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            initial_balance: Starting balance for paper trading
            window_sizes: List of window sizes for models
            model_dir: Directory containing trained models
        """
        self.symbol = symbol
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.window_sizes = window_sizes
        
        logger.info(f"Initializing Paper Trader for {symbol}")
        
        # Initialize data fetcher with 30m intervals
        self.data_fetcher = DataFetcher(symbol, interval="30m")
        
        # Load historical data - fetch 300 30-minute candles from API
        logger.info("Loading historical data (300 30-minute candles from Binance API)...")
        self.historical_data = self.data_fetcher.get_historical_data(
            limit=300,  # Fetch exactly 300 30-minute candles
            force_api=True  # Force fetch from API, not database
        )
        
        # Validate data
        if not self.data_fetcher.validate_data(self.historical_data):
            raise ValueError("Invalid historical data")
        
        logger.info(f"Loaded {len(self.historical_data)} historical records")
        
        # Initialize feature factory
        logger.info("Initializing Feature Factory...")
        self.feature_factory = FeatureFactory(self.historical_data)
        
        # Initialize model manager
        logger.info("Initializing Model Manager...")
        self.model_manager = ModelManager(model_dir, window_sizes)
        
        # Display model information
        model_info = self.model_manager.get_model_info()
        logger.info(f"Available LSTM models: {model_info['lstm_windows']}")
        logger.info(f"Available XGBoost models: {model_info['xgboost_windows']}")
        
        # Trading settings
        self.trade_threshold = 0.55  # Threshold for taking a trade (above = buy, below 1-threshold = sell)
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
        
        # Trading log
        self.trade_log = []
        
        logger.info("Paper Trader initialized successfully")
    
    def update_data(self) -> None:
        """Fetch fresh 300 30-minute candles from Binance API for each trading iteration."""
        try:
            # Fetch fresh 300 30-minute candles from API each iteration
            logger.info("Fetching fresh 300 30-minute candles from Binance API...")
            self.historical_data = self.data_fetcher.get_historical_data(
                limit=300,  # Fetch exactly 300 30-minute candles
                force_api=True  # Force fetch from API, not database
            )
            
            # Validate the fresh data
            if not self.data_fetcher.validate_data(self.historical_data):
                raise ValueError("Invalid fresh historical data")
            
            # Update feature factory with fresh data
            self.feature_factory = FeatureFactory(self.historical_data)
            
            logger.debug(f"Fresh data updated successfully - {len(self.historical_data)} candles")
            
        except Exception as e:
            logger.error(f"Failed to update data: {e}")
            raise
    
    def make_trading_decision(self) -> Dict:
        """
        Make a trading decision based on model predictions.
        
        Returns:
            Dictionary with trading decision and supporting data
        """
        try:
            # Get predictions from all models
            predictions = self.model_manager.predict(self.feature_factory)
            
            # Get current price
            current_price = float(self.historical_data.iloc[-1]['close'])
            
            # Initialize decision
            decision = {
                'action': 'HOLD',
                'price': current_price,
                'predictions': predictions,
                'timestamp': datetime.datetime.now().isoformat(),
                'reason': ''
            }
            
            # Get combined prediction
            combined_pred = predictions['combined']
            
            # Check if we should open a position
            if self.position == 0:
                if combined_pred > self.trade_threshold:
                    decision['action'] = 'BUY'
                    decision['reason'] = f'Combined prediction ({combined_pred:.4f}) above threshold ({self.trade_threshold})'
                elif combined_pred < (1 - self.trade_threshold):
                    decision['action'] = 'SELL'
                    decision['reason'] = f'Combined prediction ({combined_pred:.4f}) below inverse threshold ({1-self.trade_threshold})'
            
            # Check if we should close a position
            elif self.position > 0:  # Long position
                # Check stop loss
                if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                    decision['action'] = 'CLOSE'
                    decision['reason'] = f'Stop loss triggered: {current_price:.2f} <= {self.entry_price * (1 - self.stop_loss_pct):.2f}'
                # Check take profit
                elif current_price >= self.entry_price * (1 + self.take_profit_pct):
                    decision['action'] = 'CLOSE'
                    decision['reason'] = f'Take profit triggered: {current_price:.2f} >= {self.entry_price * (1 + self.take_profit_pct):.2f}'
                # Check for reversal signal
                elif combined_pred < (1 - self.trade_threshold):
                    decision['action'] = 'CLOSE'
                    decision['reason'] = f'Reversal signal: prediction ({combined_pred:.4f}) below inverse threshold ({1-self.trade_threshold})'
            
            elif self.position < 0:  # Short position
                # Check stop loss
                if current_price >= self.entry_price * (1 + self.stop_loss_pct):
                    decision['action'] = 'CLOSE'
                    decision['reason'] = f'Stop loss triggered: {current_price:.2f} >= {self.entry_price * (1 + self.stop_loss_pct):.2f}'
                # Check take profit
                elif current_price <= self.entry_price * (1 - self.take_profit_pct):
                    decision['action'] = 'CLOSE'
                    decision['reason'] = f'Take profit triggered: {current_price:.2f} <= {self.entry_price * (1 - self.take_profit_pct):.2f}'
                # Check for reversal signal
                elif combined_pred > self.trade_threshold:
                    decision['action'] = 'CLOSE'
                    decision['reason'] = f'Reversal signal: prediction ({combined_pred:.4f}) above threshold ({self.trade_threshold})'
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to make trading decision: {e}")
            # Return a safe default decision
            current_price = float(self.historical_data.iloc[-1]['close']) if not self.historical_data.empty else 0
            return {
                'action': 'HOLD',
                'price': current_price,
                'predictions': {'combined': 0.5},
                'timestamp': datetime.datetime.now().isoformat(),
                'reason': f'Error in decision making: {e}'
            }
    
    def execute_trade(self, decision: Dict) -> None:
        """
        Execute a paper trade based on the decision.
        
        Args:
            decision: Trading decision dictionary
        """
        current_price = decision['price']
        action = decision['action']
        
        try:
            if action == 'BUY' and self.position == 0:
                # Calculate position size (use 95% of balance to account for fees)
                position_size = (self.balance * 0.95) / current_price
                self.position = position_size
                self.entry_price = current_price
                
                # Log trade
                trade = {
                    'action': 'BUY',
                    'timestamp': decision['timestamp'],
                    'price': current_price,
                    'position_size': position_size,
                    'balance': self.balance,
                    'reason': decision['reason'],
                    'predictions': decision['predictions']
                }
                self.trade_log.append(trade)
                
                logger.info(f"BUY: {position_size:.6f} {self.symbol} at {current_price:.2f}")
                logger.info(f"Reason: {decision['reason']}")
            
            elif action == 'SELL' and self.position == 0:
                # For simplicity, assume we can short with the same amount
                position_size = (self.balance * 0.95) / current_price
                self.position = -position_size  # Negative for short
                self.entry_price = current_price
                
                # Log trade
                trade = {
                    'action': 'SELL',
                    'timestamp': decision['timestamp'],
                    'price': current_price,
                    'position_size': position_size,
                    'balance': self.balance,
                    'reason': decision['reason'],
                    'predictions': decision['predictions']
                }
                self.trade_log.append(trade)
                
                logger.info(f"SELL: {position_size:.6f} {self.symbol} at {current_price:.2f}")
                logger.info(f"Reason: {decision['reason']}")
            
            elif action == 'CLOSE':
                if self.position > 0:  # Close long position
                    self.balance = self.balance + (self.position * current_price * 0.995)  # Account for fees
                    
                    profit_loss = ((current_price / self.entry_price) - 1) * 100
                    
                    # Log trade
                    trade = {
                        'action': 'CLOSE_LONG',
                        'timestamp': decision['timestamp'],
                        'price': current_price,
                        'position_size': self.position,
                        'balance': self.balance,
                        'profit_loss_pct': profit_loss,
                        'reason': decision['reason'],
                        'predictions': decision['predictions']
                    }
                    self.trade_log.append(trade)
                    
                    logger.info(f"CLOSE LONG: {self.position:.6f} {self.symbol} at {current_price:.2f} (P/L: {profit_loss:.2f}%)")
                    logger.info(f"Reason: {decision['reason']}")
                    
                    self.position = 0
                    self.entry_price = 0
                
                elif self.position < 0:  # Close short position
                    self.balance = self.balance + (abs(self.position) * (2 * self.entry_price - current_price) * 0.995)
                    
                    profit_loss = ((self.entry_price / current_price) - 1) * 100
                    
                    # Log trade
                    trade = {
                        'action': 'CLOSE_SHORT',
                        'timestamp': decision['timestamp'],
                        'price': current_price,
                        'position_size': abs(self.position),
                        'balance': self.balance,
                        'profit_loss_pct': profit_loss,
                        'reason': decision['reason'],
                        'predictions': decision['predictions']
                    }
                    self.trade_log.append(trade)
                    
                    logger.info(f"CLOSE SHORT: {abs(self.position):.6f} {self.symbol} at {current_price:.2f} (P/L: {profit_loss:.2f}%)")
                    logger.info(f"Reason: {decision['reason']}")
                    
                    self.position = 0
                    self.entry_price = 0
        
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of trading performance."""
        if not self.trade_log:
            return {'message': 'No trades executed yet'}
        
        total_return = ((self.balance / 10000.0) - 1) * 100
        
        # Count trades
        buy_trades = len([t for t in self.trade_log if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.trade_log if t['action'] == 'SELL'])
        close_trades = len([t for t in self.trade_log if 'CLOSE' in t['action']])
        
        # Calculate win rate for closed trades
        closed_with_pnl = [t for t in self.trade_log if 'profit_loss_pct' in t]
        if closed_with_pnl:
            winning_trades = len([t for t in closed_with_pnl if t['profit_loss_pct'] > 0])
            win_rate = (winning_trades / len(closed_with_pnl)) * 100
        else:
            win_rate = 0
        
        return {
            'initial_balance': 10000.0,
            'current_balance': self.balance,
            'total_return_pct': total_return,
            'total_trades': len(self.trade_log),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'close_trades': close_trades,
            'win_rate_pct': win_rate,
            'current_position': self.position
        }
    
    def run(self, iterations: int = 100, interval_seconds: int = 60) -> None:
        """
        Run the paper trader for a specified number of iterations.
        
        Args:
            iterations: Number of trading iterations to run
            interval_seconds: Seconds to wait between iterations
        """
        logger.info(f"Starting paper trader for {self.symbol} with {iterations} iterations")
        logger.info(f"Initial balance: ${self.balance:.2f}")
        
        for i in range(iterations):
            logger.info(f"\n--- Iteration {i+1}/{iterations} ---")
            
            try:
                # Update data - fetch fresh 300 30-minute candles each iteration
                logger.info("Fetching fresh market data for this iteration...")
                self.update_data()
                
                # Make trading decision
                decision = self.make_trading_decision()
                
                # Log prediction details
                predictions = decision.get('predictions', {})
                logger.info(f"Predictions - Combined: {predictions.get('combined', 'N/A'):.4f}")
                if 'lstm' in predictions:
                    logger.info(f"LSTM: {predictions['lstm']}")
                if 'xgboost' in predictions:
                    logger.info(f"XGBoost: {predictions['xgboost']}")
                
                # Execute trade
                self.execute_trade(decision)
                
                # Display current status
                current_price = self.historical_data.iloc[-1]['close']
                position_value = self.position * current_price if self.position > 0 else abs(self.position) * current_price
                total_value = self.balance + position_value
                
                logger.info(f"Current price: ${current_price:.2f}")
                logger.info(f"Balance: ${self.balance:.2f}")
                logger.info(f"Position: {self.position:.6f}")
                logger.info(f"Total value: ${total_value:.2f}")
                
                # Performance summary every 10 iterations
                if (i + 1) % 10 == 0:
                    summary = self.get_performance_summary()
                    logger.info(f"Performance Summary: {summary}")
                
                # Wait for next iteration
                if i < iterations - 1:
                    time.sleep(interval_seconds)
                    
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                time.sleep(interval_seconds)  # Still wait before next iteration
        
        # Final summary
        final_summary = self.get_performance_summary()
        logger.info(f"\n=== Final Performance Summary ===")
        for key, value in final_summary.items():
            logger.info(f"{key}: {value}")
        
        logger.info("Paper trading session completed")


def main():
    """Main function to run the paper trader with Feature Factory."""
    
    # Configuration
    SYMBOL = "BTCEUR"  # Change this to your preferred symbol
    INITIAL_BALANCE = 10000.0
    WINDOW_SIZES = [30, 60, 90]  # Days
    MODEL_DIR = "./models"
    ITERATIONS = 50
    INTERVAL_SECONDS = 30  # Check every 30 seconds for demo
    
    try:
        # Create and run the paper trader
        trader = PaperTrader(
            symbol=SYMBOL,
            initial_balance=INITIAL_BALANCE,
            window_sizes=WINDOW_SIZES,
            model_dir=MODEL_DIR
        )
        
        # Run the trader
        trader.run(iterations=ITERATIONS, interval_seconds=INTERVAL_SECONDS)
        
    except Exception as e:
        logger.error(f"Failed to run paper trader: {e}")
        raise


if __name__ == "__main__":
    main()