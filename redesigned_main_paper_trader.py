"""Redesigned main paper trader with WebSocket-based real-time pricing and separate historical data."""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import contextlib

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent))

from paper_trader.config.settings import TradingSettings
from paper_trader.data.realtime_price_collector import RealtimePriceCollector, PriceUpdate
from paper_trader.data.historical_data_collector import HistoricalDataCollector
from paper_trader.models.feature_engineer import FeatureEngineer
from paper_trader.models.model_loader import WindowBasedModelLoader, WindowBasedEnsemblePredictor, directional_loss
from paper_trader.strategy.signal_generator import SignalGenerator
from paper_trader.strategy.exit_manager import ExitManager
from paper_trader.portfolio.portfolio_manager import PortfolioManager
from paper_trader.notifications.telegram_notifier import TelegramNotifier
from paper_trader.notifications.websocket_server import PredictionWebSocketServer


class RedesignedPaperTrader:
    """Redesigned paper trader with WebSocket-based real-time pricing."""
    
    def __init__(self):
        # Load settings
        self.settings = TradingSettings()
        
        # Setup logging
        self._setup_logging()
        
        # Core data collectors
        self.realtime_price_collector = None
        self.historical_data_collector = None
        
        # Trading components
        self.feature_engineer = None
        self.model_loader = None
        self.predictor = None
        self.signal_generator = None
        self.exit_manager = None
        self.portfolio_manager = None
        
        # Notification components
        self.telegram_notifier = None
        self.ws_server = None
        
        # Prediction tracking
        self.last_prediction_time: Dict[str, datetime] = {}
        self.prediction_triggered: Dict[str, bool] = {}
        
        # State tracking
        self.last_hourly_update = datetime.now()
        self.is_running = False
        self.prediction_cycle_task = None
        
        self.logger.info("Redesigned Paper Trader initialized")
        self.trading_logger.info("=== REDESIGNED PAPER TRADER STARTING ===")
        self.trading_logger.info(f"Settings: MIN_CONFIDENCE={self.settings.min_confidence_threshold}")
        self.trading_logger.info(f"Symbols: {self.settings.symbols}")
        self.trading_logger.info(f"Capital: €{self.settings.initial_capital}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        log_dir = Path('paper_trader/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use dedicated log files
        debug_log_file = log_dir / 'redesigned_debug.log'
        trading_decisions_log_file = log_dir / 'redesigned_trading_decisions.log'

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(debug_log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create a separate logger for trading decisions
        self.trading_logger = logging.getLogger('redesigned_trading_decisions')
        self.trading_logger.setLevel(logging.INFO)
        trading_handler = logging.FileHandler(trading_decisions_log_file, mode='a')
        trading_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.trading_logger.addHandler(trading_handler)
        
        self.logger = logging.getLogger(__name__)
        
        # Reduce noise from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('telegram').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('websockets').setLevel(logging.WARNING)
        
    async def initialize(self):
        """Initialize all components."""
        try:
            self.logger.info("Initializing Redesigned Paper Trader components...")
            
            # Initialize real-time price collector
            self.realtime_price_collector = RealtimePriceCollector(self.settings)
            await self.realtime_price_collector.initialize()
            
            # Register price update callback
            self.realtime_price_collector.add_price_update_callback(self._on_price_update)
            
            # Initialize historical data collector
            self.historical_data_collector = HistoricalDataCollector(self.settings)
            await self.historical_data_collector.initialize()
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Initialize model loader
            self.model_loader = WindowBasedModelLoader(self.settings.model_path)
            
            # Initialize ensemble predictor
            self.predictor = WindowBasedEnsemblePredictor(
                model_loader=self.model_loader,
                min_confidence_threshold=self.settings.min_confidence_threshold,
                min_signal_strength=self.settings.min_signal_strength,
                settings=self.settings
            )
            
            # Set ensemble weights
            self.predictor.lstm_weight = self.settings.lstm_weight
            self.predictor.xgb_weight = self.settings.xgb_weight
            self.predictor.caboose_weight = self.settings.caboose_weight
            
            # Initialize trading components
            self.signal_generator = SignalGenerator(
                max_positions=self.settings.max_positions,
                max_positions_per_symbol=self.settings.max_positions_per_symbol,
                base_position_size=self.settings.base_position_size,
                max_position_size=self.settings.max_position_size,
                min_position_size=self.settings.min_position_size,
                take_profit_pct=self.settings.take_profit_pct,
                stop_loss_pct=self.settings.stop_loss_pct,
                min_confidence=self.settings.min_confidence_threshold,
                min_signal_strength=self.settings.min_signal_strength,
                min_expected_gain_pct=self.settings.min_expected_gain_pct,
                position_cooldown_minutes=self.settings.position_cooldown_minutes,
                data_collector=None,  # Will use real-time price collector instead
                max_daily_trades_per_symbol=self.settings.max_daily_trades_per_symbol,
                settings=self.settings
            )
            
            self.exit_manager = ExitManager(
                trailing_stop_pct=self.settings.trailing_stop_pct,
                max_hold_hours=self.settings.max_hold_hours,
                enable_prediction_exits=self.settings.enable_prediction_exits,
                prediction_exit_min_confidence=self.settings.prediction_exit_min_confidence,
                prediction_exit_min_strength=self.settings.prediction_exit_min_strength,
                dynamic_stop_loss_adjustment=self.settings.dynamic_stop_loss_adjustment,
                settings=self.settings
            )
            
            self.portfolio_manager = PortfolioManager(
                initial_capital=self.settings.initial_capital,
                max_positions_per_symbol=self.settings.max_positions_per_symbol
            )
            
            # Initialize notification components
            self.telegram_notifier = TelegramNotifier(
                bot_token=self.settings.telegram_bot_token,
                chat_id=self.settings.telegram_chat_id
            )
            
            self.ws_server = PredictionWebSocketServer(
                host=self.settings.websocket_server_host,
                port=self.settings.websocket_server_port
            )
            await self.ws_server.start()
            
            # Test Telegram connection
            if self.telegram_notifier.is_enabled():
                await self.telegram_notifier.test_connection()
                
            # Initialize prediction tracking
            for symbol in self.settings.symbols:
                self.last_prediction_time[symbol] = datetime.now()
                self.prediction_triggered[symbol] = False
            
            self.logger.info("All redesigned components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing redesigned components: {e}")
            raise
            
    async def load_models(self):
        """Load ML models for all symbols."""
        import tensorflow as tf
        
        try:
            self.logger.info("Loading ML models...")
            
            loaded_count = 0
            for symbol in self.settings.symbols:
                try:
                    success = await self.model_loader.load_symbol_models(
                        symbol, 
                        custom_objects={"directional_loss": directional_loss}
                    )
                    
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
            
    async def _on_price_update(self, price_update: PriceUpdate):
        """Handle real-time price updates from WebSocket."""
        try:
            symbol = price_update.symbol
            
            # Check if we should trigger a prediction (every minute)
            last_prediction = self.last_prediction_time.get(symbol)
            time_since_prediction = (
                (datetime.now() - last_prediction).total_seconds() 
                if last_prediction else float('inf')
            )
            
            # Trigger prediction every minute (60 seconds)
            if time_since_prediction >= 60:
                self.prediction_triggered[symbol] = True
                self.logger.debug(f"Price update triggered prediction cycle for {symbol}")
                
            # Log significant price movements
            volatility = self.realtime_price_collector.calculate_volatility(symbol)
            if volatility > 0.02:  # Log if volatility > 2%
                self.trading_logger.info(
                    f"📊 HIGH VOLATILITY {symbol}: {price_update.price:.4f} "
                    f"(volatility: {volatility*100:.1f}%)"
                )
                
        except Exception as e:
            self.logger.error(f"Error handling price update for {symbol}: {e}")
            
    async def process_symbol_prediction(self, symbol: str) -> bool:
        """Process trading prediction for a single symbol using real-time price."""
        try:
            self.trading_logger.info(f"🔍 PROCESSING PREDICTION: {symbol}")
            
            # Check if models are available
            if (symbol not in getattr(self.model_loader, 'lstm_models', {}) and
                symbol not in getattr(self.model_loader, 'xgb_models', {})):
                self.trading_logger.warning(f"❌ NO MODELS AVAILABLE for {symbol}")
                return False
            
            # Get current real-time price
            current_price = self.realtime_price_collector.get_current_price(symbol)
            if current_price is None:
                self.trading_logger.warning(f"❌ NO REAL-TIME PRICE for {symbol}")
                return False
                
            # Check price age
            price_age = self.realtime_price_collector.get_price_age_seconds(symbol)
            if price_age and price_age > 120:  # Price older than 2 minutes
                self.trading_logger.warning(
                    f"⚠️ STALE PRICE for {symbol}: {price_age:.0f} seconds old"
                )
                return False
            
            # Get historical data for features
            historical_data = await self.historical_data_collector.get_historical_data_for_features(
                symbol, min_length=500
            )
            if historical_data is None or len(historical_data) < 500:
                self.trading_logger.warning(f"❌ INSUFFICIENT HISTORICAL DATA for {symbol}")
                return False
            
            # Feature engineering
            features_df = self.feature_engineer.engineer_features(historical_data)
            if features_df is None or len(features_df) < self.settings.sequence_length:
                actual_features = len(features_df) if features_df is not None else 0
                self.trading_logger.warning(
                    f"❌ INSUFFICIENT FEATURES for {symbol}: {actual_features}/{self.settings.sequence_length}"
                )
                return False
            
            # Update the latest price in features for prediction
            # (Replace the last close price with real-time price)
            features_df.iloc[-1, features_df.columns.get_loc('close')] = current_price
            
            self.trading_logger.info(
                f"📊 DATA READY for {symbol}: {len(historical_data)} historical candles, "
                f"{len(features_df)} features, real-time price: €{current_price:.4f}"
            )
            
            # Calculate market volatility
            market_volatility = self.realtime_price_collector.calculate_volatility(symbol)
            
            # Generate prediction with real-time price
            prediction_result = await self.predictor.predict(
                symbol, features_df, market_volatility, current_price
            )
            if prediction_result is None:
                self.trading_logger.warning(f"❌ PREDICTION FAILED for {symbol}")
                return False
            
            # Log prediction result
            self.trading_logger.info(f"🧠 PREDICTION for {symbol}: {prediction_result}")
            
            # Broadcast prediction via WebSocket
            if self.ws_server:
                await self.ws_server.broadcast({
                    'type': 'prediction',
                    'symbol': symbol,
                    'data': prediction_result,
                    'real_time_price': current_price,
                    'price_age_seconds': price_age,
                    'market_volatility': market_volatility
                })
            
            # Check if prediction meets trading thresholds
            if not self.predictor._meets_trading_threshold(
                prediction_result['confidence'], 
                prediction_result['signal_strength'],
                prediction_result
            ):
                self.trading_logger.warning(f"❌ PREDICTION THRESHOLD NOT MET for {symbol}")
                return False
            
            # Check position limits
            current_positions = self.portfolio_manager.positions.get(symbol, [])
            if len(current_positions) >= self.settings.max_positions_per_symbol:
                self.trading_logger.warning(
                    f"❌ MAX POSITIONS REACHED for {symbol}: "
                    f"{len(current_positions)}/{self.settings.max_positions_per_symbol}"
                )
                return False
            
            # Generate signal with real-time price
            signal = self.signal_generator.generate_signal(
                symbol=symbol,
                prediction=prediction_result,
                current_price=current_price,
                portfolio=self.portfolio_manager
            )
            
            if signal and signal['action'] == 'BUY':
                return await self._execute_buy_signal(symbol, signal, current_price, prediction_result)
            else:
                self.trading_logger.info(f"🚫 NO SIGNAL GENERATED for {symbol}")
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing prediction for {symbol}: {e}")
            self.trading_logger.error(f"❌ ERROR PROCESSING {symbol}: {e}")
            return False
            
    async def _execute_buy_signal(self, symbol: str, signal: dict, current_price: float, prediction_result: dict) -> bool:
        """Execute a buy signal."""
        try:
            self.trading_logger.info(f"💰 EXECUTING BUY SIGNAL for {symbol}")
            
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
                self.trading_logger.info(f"✅ POSITION OPENED for {symbol}: {signal}")
                
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
                
                return True
            else:
                self.trading_logger.error(f"❌ FAILED TO OPEN POSITION for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing buy signal for {symbol}: {e}")
            
        return False
        
    async def check_exit_conditions(self):
        """Check exit conditions for all open positions using real-time prices."""
        try:
            positions_to_close = []
            
            for symbol, pos_list in list(self.portfolio_manager.positions.items()):
                try:
                    # Get current real-time price
                    current_price = self.realtime_price_collector.get_current_price(symbol)
                    if current_price is None:
                        continue

                    for position in list(pos_list):
                        # Check exit conditions
                        exit_result = self.exit_manager.check_exit_conditions(position, current_price)

                        if exit_result:
                            exit_price = exit_result.get('exit_price', current_price)
                            exit_reason = exit_result['reason']
                            positions_to_close.append((symbol, position, exit_price, exit_reason))
                
                except Exception as e:
                    self.logger.error(f"Error checking exit for {symbol}: {e}")
            
            # Close positions that need to be closed
            for symbol, position, exit_price, exit_reason in positions_to_close:
                try:
                    success = self.portfolio_manager.close_position(
                        symbol=symbol,
                        position=position,
                        exit_price=exit_price,
                        reason=exit_reason
                    )
                    
                    if success:
                        # Calculate P&L for notification
                        pnl = (exit_price - position.entry_price) * position.quantity
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                        hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600
                        
                        self.trading_logger.info(
                            f"✅ POSITION CLOSED for {symbol}: {exit_reason}, "
                            f"P&L: €{pnl:.2f} ({pnl_pct*100:+.2f}%)"
                        )
                        
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
                
                except Exception as e:
                    self.logger.error(f"Error closing position for {symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            
    async def prediction_cycle_loop(self):
        """Main prediction cycle loop - runs every minute."""
        self.logger.info("Starting prediction cycle loop")
        
        while self.is_running:
            try:
                cycle_start = datetime.now()
                predictions_made = 0
                successful_trades = 0
                
                self.trading_logger.info(f"🔄 STARTING PREDICTION CYCLE: {cycle_start.strftime('%H:%M:%S')}")
                
                # Process symbols that have triggered predictions or need periodic updates
                for symbol in self.settings.symbols:
                    try:
                        # Check if prediction was triggered by price update or periodic timer
                        should_predict = self.prediction_triggered.get(symbol, False)
                        
                        # Also trigger prediction every minute regardless
                        last_prediction = self.last_prediction_time.get(symbol)
                        time_since_prediction = (
                            (datetime.now() - last_prediction).total_seconds() 
                            if last_prediction else float('inf')
                        )
                        
                        if should_predict or time_since_prediction >= 60:
                            success = await self.process_symbol_prediction(symbol)
                            predictions_made += 1
                            
                            if success:
                                successful_trades += 1
                                
                            # Reset trigger and update timing
                            self.prediction_triggered[symbol] = False
                            self.last_prediction_time[symbol] = datetime.now()
                            
                    except Exception as e:
                        self.logger.error(f"Error in prediction cycle for {symbol}: {e}")
                
                # Check exit conditions for all positions
                await self.check_exit_conditions()
                
                # Log cycle summary
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                
                # Get current portfolio status
                total_positions = sum(len(positions) for positions in self.portfolio_manager.positions.values())
                available_capital = self.portfolio_manager.get_available_capital()
                
                self.trading_logger.info(
                    f"🔄 CYCLE COMPLETE: {cycle_time:.1f}s, "
                    f"predictions: {predictions_made}, trades: {successful_trades}, "
                    f"positions: {total_positions}, capital: €{available_capital:.2f}"
                )
                
                # Wait for next cycle (1 minute)
                if self.is_running:
                    await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in prediction cycle: {e}")
                self.trading_logger.error(f"❌ PREDICTION CYCLE ERROR: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def send_hourly_update(self):
        """Send hourly portfolio update."""
        try:
            # Get current prices for all positions
            current_prices = {}
            for symbol in self.portfolio_manager.positions.keys():
                price = self.realtime_price_collector.get_current_price(symbol)
                if price:
                    current_prices[symbol] = price
            
            # Get portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary(current_prices)
            
            # Send Telegram update
            await self.telegram_notifier.send_portfolio_update(portfolio_summary)
            
            # Log connection status
            connection_status = self.realtime_price_collector.get_connection_status()
            healthy_connections = sum(
                1 for status in connection_status.values() 
                if status.get('connected', False) and status.get('seconds_since_update', 999) < 120
            )
            
            self.logger.info(
                f"Sent hourly portfolio update - WebSocket connections: "
                f"{healthy_connections}/{len(self.settings.symbols)} healthy"
            )
            
        except Exception as e:
            self.logger.error(f"Error sending hourly update: {e}")
            
    async def run(self):
        """Main trading loop."""
        try:
            self.logger.info("Starting Redesigned Paper Trader...")
            self.trading_logger.info("🚀 REDESIGNED PAPER TRADER STARTING UP")
            
            # Log settings
            self.trading_logger.info("⚙️ CURRENT SETTINGS:")
            self.trading_logger.info(f"   📊 Min Confidence: {self.settings.min_confidence_threshold}")
            self.trading_logger.info(f"   💪 Min Signal Strength: {self.settings.min_signal_strength}")
            self.trading_logger.info(f"   💰 Initial Capital: €{self.settings.initial_capital}")
            self.trading_logger.info(f"   📋 Symbols: {self.settings.symbols}")
            
            # Initialize components
            await self.initialize()
            
            # Load models
            await self.load_models()
            
            # Send startup notification
            await self.telegram_notifier.send_system_status(
                "STARTED",
                f"Redesigned Paper Trader started with WebSocket pricing for {len(self.settings.symbols)} symbols"
            )
            
            self.is_running = True
            
            # Start real-time price collection
            self.logger.info("Starting real-time price collection...")
            await self.realtime_price_collector.start()
            
            # Start historical data updates
            self.logger.info("Starting historical data collection...")
            await self.historical_data_collector.start_periodic_updates()
            
            # Wait for initial data
            self.logger.info("Waiting for initial price data...")
            await asyncio.sleep(10)
            
            # Start prediction cycle
            self.logger.info("Starting prediction cycle...")
            self.prediction_cycle_task = asyncio.create_task(self.prediction_cycle_loop())
            
            self.logger.info("Redesigned Paper Trader is now running...")
            self.trading_logger.info("✅ REDESIGNED PAPER TRADER READY FOR TRADING")
            
            # Main loop for housekeeping
            while self.is_running:
                try:
                    # Send hourly updates
                    now = datetime.now()
                    if now - self.last_hourly_update >= timedelta(hours=1):
                        await self.send_hourly_update()
                        self.last_hourly_update = now
                    
                    # Wait before next housekeeping cycle
                    await asyncio.sleep(300)  # 5 minutes
                    
                except KeyboardInterrupt:
                    self.logger.info("Received shutdown signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(60)
        
        except Exception as e:
            self.logger.error(f"Critical error in Redesigned Paper Trader: {e}")
            self.trading_logger.error(f"🚨 CRITICAL ERROR: {e}")
            await self.telegram_notifier.send_error_alert(str(e), "System")
            raise
        
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Graceful shutdown."""
        try:
            self.logger.info("Shutting down Redesigned Paper Trader...")
            self.is_running = False

            # Cancel prediction cycle
            if self.prediction_cycle_task and not self.prediction_cycle_task.done():
                self.prediction_cycle_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.prediction_cycle_task
            
            # Stop data collectors
            if self.realtime_price_collector:
                await self.realtime_price_collector.stop()
                
            if self.historical_data_collector:
                await self.historical_data_collector.stop()
            
            # Send final portfolio update
            await self.send_hourly_update()
            
            # Send shutdown notification
            await self.telegram_notifier.send_system_status(
                "STOPPED",
                "Redesigned Paper Trader has been stopped"
            )

            # Stop WebSocket server
            if self.ws_server:
                await self.ws_server.stop()

            self.logger.info("Redesigned Paper Trader shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point."""
    trader = RedesignedPaperTrader()
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.basicConfig()
        log = logging.getLogger(__name__)
        log.error("A fatal error occurred in Redesigned Paper Trader.", exc_info=True)
        sys.exit(1)