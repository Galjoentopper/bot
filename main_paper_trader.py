"""Main paper trader script - orchestrates the entire paper trading system."""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import contextlib

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent))

from paper_trader.config.settings import TradingSettings
from paper_trader.data.bitvavo_collector import BitvavoDataCollector
from paper_trader.models.feature_engineer import FeatureEngineer
from paper_trader.models.model_loader import WindowBasedModelLoader, WindowBasedEnsemblePredictor, directional_loss
from paper_trader.strategy.signal_generator import SignalGenerator
from paper_trader.strategy.exit_manager import ExitManager
from paper_trader.portfolio.portfolio_manager import PortfolioManager
from paper_trader.notifications.telegram_notifier import TelegramNotifier

class PaperTrader:
    """Main paper trading orchestrator."""
    
    def __init__(self):
        # Load settings
        self.settings = TradingSettings()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.data_collector = None
        self.feature_engineer = None
        self.model_loader = None
        self.predictor = None
        self.signal_generator = None
        self.exit_manager = None
        self.portfolio_manager = None
        self.telegram_notifier = None

        # Background tasks
        self.data_feed_task = None
        self.api_update_task = None
        
        # State tracking
        self.last_hourly_update = datetime.now()
        self.is_running = False
        
        self.logger.info("Paper Trader initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path('paper_trader/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'paper_trader.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Reduce noise from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('telegram').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
    
    async def initialize(self):
        """Initialize all components."""
        try:
            self.logger.info("Initializing Paper Trader components...")
            
            # Initialize data collector
            self.data_collector = BitvavoDataCollector(
                api_key=self.settings.bitvavo_api_key,
                api_secret=self.settings.bitvavo_api_secret
            )
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Initialize window-based model loader
            self.model_loader = WindowBasedModelLoader(self.settings.model_path)
            
            # Initialize enhanced ensemble predictor with confidence thresholds
            self.predictor = WindowBasedEnsemblePredictor(
                model_loader=self.model_loader,
                min_confidence_threshold=self.settings.min_confidence_threshold,
                min_signal_strength=self.settings.min_signal_strength
            )
            
            # Set ensemble weights from settings
            self.predictor.lstm_weight = self.settings.lstm_weight
            self.predictor.xgb_weight = self.settings.xgb_weight
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator(
                max_positions=self.settings.max_positions,
                position_size_pct=self.settings.position_size_pct,
                take_profit_pct=self.settings.take_profit_pct,
                stop_loss_pct=self.settings.stop_loss_pct
            )
            
            # Initialize exit manager
            self.exit_manager = ExitManager(
                trailing_stop_pct=self.settings.trailing_stop_pct,
                max_hold_hours=self.settings.max_hold_hours
            )
            
            # Initialize portfolio manager
            self.portfolio_manager = PortfolioManager(
                initial_capital=self.settings.initial_capital
            )
            
            # Initialize Telegram notifier
            self.telegram_notifier = TelegramNotifier(
                bot_token=self.settings.telegram_bot_token,
                chat_id=self.settings.telegram_chat_id
            )
            
            # Test Telegram connection
            if self.telegram_notifier.is_enabled():
                await self.telegram_notifier.test_connection()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def load_models(self):
        """Load ML models for all symbols."""
        import tensorflow as tf
        from paper_trader.models.model_loader import directional_loss
        try:
            self.logger.info("Loading ML models...")
            
            loaded_count = 0
            for symbol in self.settings.symbols:
                try:
                    # Load models for symbol with explicit custom_objects for directional_loss
                    success = await self.model_loader.load_symbol_models(symbol, custom_objects={"directional_loss": directional_loss})
                    
                    if success:
                        loaded_count += 1
                        self.logger.info(f"Models loaded for {symbol}")
                    else:
                        self.logger.warning(f"No models found for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading models for {symbol}: {e}")
            
            self.logger.info(f"Models loaded for {loaded_count} symbols")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    async def process_symbol(self, symbol: str) -> bool:
        """Process trading signals for a single symbol."""
        try:
            # Skip if no models for this symbol
            if (symbol not in self.model_loader.lstm_models and 
                symbol not in self.model_loader.xgb_models):
                return False
            
            # Get latest data
            data = await self.data_collector.get_latest_data(symbol, limit=200)
            if data is None or len(data) < 50:
                self.logger.warning(f"Insufficient data for {symbol}")
                return False
            
            # Engineer features
            features_df = self.feature_engineer.engineer_features(data)
            if features_df is None or len(features_df) < self.settings.sequence_length:
                self.logger.warning(f"Insufficient features for {symbol}")
                return False
            
            # Get current price
            current_price = await self.data_collector.get_current_price(symbol)
            if current_price is None:
                self.logger.warning(f"Could not get current price for {symbol}")
                return False
            
            # Calculate market volatility for enhanced prediction
            price_data = features_df['close'].tail(20)  # Use last 20 periods
            market_volatility = price_data.std() / price_data.mean() if len(price_data) > 1 else 0.5
            
            # Generate enhanced prediction with market volatility
            prediction_result = await self.predictor.predict(symbol, features_df, market_volatility)
            if prediction_result is None:
                return False
            
            # Check if prediction meets trading thresholds
            if not self.predictor._meets_trading_threshold(
                prediction_result['confidence'], 
                prediction_result['signal_strength']
            ):
                self.logger.debug(f"Prediction for {symbol} doesn't meet trading thresholds")
                return False
            
            # Check if we already have a position
            if symbol in self.portfolio_manager.positions:
                return False  # Skip if already have position
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                symbol=symbol,
                prediction=prediction_result,
                current_price=current_price,
                portfolio=self.portfolio_manager
            )
            
            if signal and signal['action'] == 'BUY':
                # Execute buy signal
                success = self.portfolio_manager.open_position(
                    symbol=symbol,
                    entry_price=current_price,
                    quantity=signal['quantity'],
                    take_profit=signal['take_profit'],
                    stop_loss=signal['stop_loss'],
                    confidence=prediction_result['confidence'],
                    signal_strength=prediction_result['signal_strength']
                )
                
                if success:
                    # Send Telegram notification
                    await self.telegram_notifier.send_position_opened(
                        symbol=symbol,
                        entry_price=current_price,
                        quantity=signal['quantity'],
                        take_profit=signal['take_profit'],
                        stop_loss=signal['stop_loss'],
                        confidence=prediction_result['confidence'],
                        signal_strength=prediction_result['signal_strength'],
                        position_value=signal['position_value']
                    )
                    
                    self.logger.info(f"Opened position for {symbol}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return False
    
    async def check_exit_conditions(self):
        """Check exit conditions for all open positions."""
        try:
            positions_to_close = []
            
            for symbol in list(self.portfolio_manager.positions.keys()):
                try:
                    # Get current price
                    current_price = await self.data_collector.get_current_price(symbol)
                    if current_price is None:
                        continue
                    
                    position = self.portfolio_manager.positions[symbol]
                    
                    # Check exit conditions
                    exit_result = self.exit_manager.check_exit_conditions(position, current_price)
                    
                    if exit_result:
                        exit_price = exit_result.get('exit_price', current_price)
                        exit_reason = exit_result['reason']
                        positions_to_close.append((symbol, exit_price, exit_reason))
                
                except Exception as e:
                    self.logger.error(f"Error checking exit for {symbol}: {e}")
            
            # Close positions that need to be closed
            for symbol, exit_price, exit_reason in positions_to_close:
                try:
                    position = self.portfolio_manager.positions[symbol]
                    
                    success = self.portfolio_manager.close_position(
                        symbol=symbol,
                        exit_price=exit_price,
                        reason=exit_reason
                    )
                    
                    if success:
                        # Calculate P&L for notification
                        pnl = (exit_price - position.entry_price) * position.quantity
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                        hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600
                        
                        # Send Telegram notification
                        await self.telegram_notifier.send_position_closed(
                            symbol=symbol,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            quantity=position.quantity,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            exit_reason=exit_reason,
                            hold_time_hours=hold_time
                        )
                        
                        self.logger.info(f"Closed position for {symbol} - {exit_reason}")
                
                except Exception as e:
                    self.logger.error(f"Error closing position for {symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    async def send_hourly_update(self):
        """Send hourly portfolio update."""
        try:
            # Get current prices for all positions
            current_prices = {}
            for symbol in self.portfolio_manager.positions.keys():
                price = await self.data_collector.get_current_price(symbol)
                if price:
                    current_prices[symbol] = price
            
            # Get portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary(current_prices)
            
            # Send Telegram update
            await self.telegram_notifier.send_portfolio_update(portfolio_summary)
            
            self.logger.info("Sent hourly portfolio update")
            
        except Exception as e:
            self.logger.error(f"Error sending hourly update: {e}")
    
    async def run(self):
        """Main trading loop."""
        try:
            self.logger.info("Starting Paper Trader...")
            
            # Initialize components
            await self.initialize()
            
            # Load models
            await self.load_models()
            
            # Send startup notification
            await self.telegram_notifier.send_system_status(
                "STARTED",
                f"Paper Trader started with {len(self.settings.symbols)} symbols"
            )
            
            self.is_running = True
            
            # Initialize data buffers
            self.logger.info("Initializing data buffers...")
            await self.data_collector.initialize_buffers(self.settings.symbols)
            
            # Start WebSocket data feed and periodic API updates in background
            self.logger.info("Starting real-time data feed...")
            self.data_feed_task = asyncio.create_task(
                self.data_collector.start_websocket_feed(self.settings.symbols)
            )
            self.api_update_task = asyncio.create_task(
                self.data_collector.update_data_periodically(self.settings.symbols)
            )
            
            self.logger.info("Paper Trader is now running...")
            
            # Main trading loop
            while self.is_running:
                try:
                    # Process each symbol
                    for symbol in self.settings.symbols:
                        if not self.is_running:
                            break
                        await self.process_symbol(symbol)
                        await asyncio.sleep(1)  # Small delay between symbols
                    
                    # Check exit conditions for open positions
                    await self.check_exit_conditions()
                    
                    # Send hourly updates
                    now = datetime.now()
                    if now - self.last_hourly_update >= timedelta(hours=1):
                        await self.send_hourly_update()
                        self.last_hourly_update = now
                    
                    # Wait before next cycle (15 minutes to align with data intervals)
                    if self.is_running:
                        self.logger.debug("Waiting for next cycle...")
                        await asyncio.sleep(900)  # 15 minutes
                
                except KeyboardInterrupt:
                    self.logger.info("Received shutdown signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await self.telegram_notifier.send_error_alert(str(e), "Main Loop")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        except Exception as e:
            self.logger.error(f"Critical error in Paper Trader: {e}")
            await self.telegram_notifier.send_error_alert(str(e), "System")
            raise
        
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown."""
        try:
            self.logger.info("Shutting down Paper Trader...")
            self.is_running = False

            # Cancel background tasks
            tasks = [t for t in [self.data_feed_task, self.api_update_task] if t]
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            
            # Send final portfolio update
            await self.send_hourly_update()
            
            # Send shutdown notification
            await self.telegram_notifier.send_system_status(
                "STOPPED",
                "Paper Trader has been stopped"
            )
            
            self.logger.info("Paper Trader shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

async def main():
    """Main entry point."""
    trader = PaperTrader()
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the paper trader
    asyncio.run(main())