"""Main paper trader script - orchestrates the entire paper trading system."""

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
from paper_trader.data.bitvavo_collector import BitvavoDataCollector
from paper_trader.models.feature_engineer import FeatureEngineer
from paper_trader.models.model_loader import WindowBasedModelLoader, WindowBasedEnsemblePredictor, directional_loss
from paper_trader.strategy.signal_generator import SignalGenerator
from paper_trader.strategy.exit_manager import ExitManager
from paper_trader.portfolio.portfolio_manager import PortfolioManager
from paper_trader.notifications.telegram_notifier import TelegramNotifier
from paper_trader.notifications.websocket_server import PredictionWebSocketServer
import json
from paper_trader.models.feature_engineer import TRAINING_FEATURES

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
        self.ws_server = None

        # Background tasks
        self.data_feed_task = None
        self.api_update_task = None

        # Track when a symbol was last processed
        self.last_prediction_time: Dict[str, datetime] = {}
        
        # State tracking
        self.last_hourly_update = datetime.now()
        self.is_running = False
        
        self.logger.info("Paper Trader initialized")
        self.trading_logger.info("=== PAPER TRADER STARTING ===")
        self.trading_logger.info(f"Settings: MIN_CONFIDENCE={self.settings.min_confidence_threshold}, MIN_SIGNAL={self.settings.min_signal_strength}")
        self.trading_logger.info(f"Symbols: {self.settings.symbols}")
        self.trading_logger.info(f"Capital: €{self.settings.initial_capital}, Max positions: {self.settings.max_positions}")

    async def debug_data_pipeline(self, symbol: str): 
        """Debug the data pipeline to identify issues.""" 
        self.logger.info(f"=== Debugging data pipeline for {symbol} ===") 
        
        try: 
            # 1. Check raw data availability 
            raw_data = await self.data_collector.get_historical_data(symbol, limit=300) 
            self.logger.info(f"Raw data shape: {raw_data.shape if raw_data is not None else 'None'}") 
            
            if raw_data is None or raw_data.empty: 
                self.logger.error(f"No raw data available for {symbol}") 
                return False 
                
            if len(raw_data) < 250: 
                self.logger.warning(f"Insufficient raw data: {len(raw_data)} rows (need 250+)") 
                return False 
                
            # 2. Check for data quality issues 
            null_counts = raw_data.isnull().sum() 
            self.logger.info(f"Null counts in raw data: {null_counts.to_dict()}") 
            
            # 3. Check for required columns 
            required_cols = ['open', 'high', 'low', 'close', 'volume'] 
            missing_cols = [col for col in required_cols if col not in raw_data.columns] 
            if missing_cols: 
                self.logger.error(f"Missing required columns: {missing_cols}") 
                return False 
                
            # 4. Test feature engineering 
            self.logger.info("Testing feature engineering...") 
            features_df = self.feature_engineer.engineer_features(raw_data) 
            
            if features_df is None: 
                self.logger.error("Feature engineering returned None") 
                return False 
                
            self.logger.info(f"Features shape: {features_df.shape}") 
            self.logger.info(f"Features columns: {len(features_df.columns)}") 
            
            # 5. Check for training features 
            missing_training_features = [] 
            for feature in TRAINING_FEATURES: 
                if feature not in features_df.columns: 
                    missing_training_features.append(feature) 
                    
            if missing_training_features: 
                self.logger.error(f"Missing training features: {missing_training_features}") 
                return False 
                
            # 6. Check for excessive NaN values 
            nan_counts = features_df.isnull().sum() 
            high_nan_features = nan_counts[nan_counts > len(features_df) * 0.1] 
            if not high_nan_features.empty: 
                self.logger.warning(f"Features with >10% NaN values: {high_nan_features.to_dict()}") 
                
            self.logger.info(f"✅ Data pipeline check passed for {symbol}") 
            return True 
            
        except Exception as e: 
            self.logger.error(f"Error in data pipeline debug: {e}", exc_info=True) 
            return False

    def run_debug_sync(self, symbol: str):
        """Synchronous wrapper for async debug function."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.debug_data_pipeline(symbol))
            loop.close()
            return result
        except Exception as e:
            self.logger.error(f"Error running debug: {e}")
            return False

    async def run_full_diagnostics(self):
        """Run comprehensive diagnostics on the trading system."""
        self.logger.info("=== RUNNING FULL SYSTEM DIAGNOSTICS ===")
        
        try:
            # Test data collector
            self.logger.info("1. Testing Data Collector...")
            buffer_status = self.data_collector.get_detailed_buffer_status()
            
            for symbol, status in buffer_status.items():
                self.logger.info(f"  {symbol}: {status}")
            
            # Test each symbol individually
            self.logger.info("2. Testing individual symbols...")
            for symbol in self.settings.symbols:
                self.logger.info(f"  Testing {symbol}...")
                
                # Test historical data fetch
                try:
                    test_data = await self.data_collector.get_historical_data(
                        symbol, self.settings.candle_interval, 10
                    )
                    if test_data is not None and len(test_data) > 0:
                        self.logger.info(f"    ✅ Can fetch historical data: {len(test_data)} candles")
                    else:
                        self.logger.error(f"    ❌ Cannot fetch historical data")
                except Exception as e:
                    self.logger.error(f"    ❌ Error fetching data: {e}")
                

                
                # Test feature engineering
                try:
                    buffer_data = self.data_collector.get_buffer_data(symbol, 100)
                    if buffer_data is not None and len(buffer_data) >= 100:
                        features = self.feature_engineer.engineer_features(buffer_data)
                        if features is not None:
                            self.logger.info(f"    ✅ Feature engineering: {features.shape}")
                        else:
                            self.logger.error(f"    ❌ Feature engineering returned None")
                    else:
                        self.logger.error(f"    ❌ Insufficient buffer data for features")
                except Exception as e:
                    self.logger.error(f"    ❌ Error in feature engineering: {e}")
            
            self.logger.info("=== DIAGNOSTICS COMPLETE ===")
            
        except Exception as e:
            self.logger.error(f"Error running diagnostics: {e}")

    async def health_check(self) -> Dict[str, bool]:
        """Quick health check of all system components."""
        health = {}
        
        try:
            # Check data collector
            health['data_collector'] = self.data_collector is not None
            
            # Check buffers
            buffer_status = self.data_collector.get_detailed_buffer_status()
            health['buffers_healthy'] = all(
                status.get('status') == 'healthy' 
                for status in buffer_status.values()
            )
            
            # Check models
            health['models_loaded'] = (
                len(getattr(self.model_loader, 'lstm_models', {})) > 0 or
                len(getattr(self.model_loader, 'xgb_models', {})) > 0
            )
            
            # Check WebSocket connection
            health['websocket_connected'] = (
                hasattr(self, 'data_feed_task') and 
                self.data_feed_task is not None and 
                not self.data_feed_task.done()
            )
            
            # Check periodic updates
            health['periodic_updates'] = (
                hasattr(self, 'api_update_task') and 
                self.api_update_task is not None and 
                not self.api_update_task.done()
            )
            
            # Overall health
            health['overall'] = all(health.values())
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            health['error'] = str(e)
            health['overall'] = False
        
        return health

    async def process_symbol_with_debugging(self, symbol: str) -> bool:
        """Enhanced version of process_symbol with detailed debugging."""
        try:
            self.logger.debug(f"Processing {symbol}...")
            
            # Check if models are available
            if (symbol not in getattr(self.model_loader, 'lstm_models', {}) and 
                symbol not in getattr(self.model_loader, 'xgb_models', {})):
                self.logger.debug(f"No models available for {symbol}")
                return False
            
            # Check data availability with detailed logging
            self.logger.debug(f"Checking data availability for {symbol}...")
            
            if not await self.data_collector.ensure_sufficient_data(symbol, min_length=500):
                self.logger.warning(f"Could not ensure sufficient data for {symbol}")
                
                # Get detailed buffer status for debugging
                buffer_status = self.data_collector.get_detailed_buffer_status()
                symbol_status = buffer_status.get(symbol, {})
                self.logger.debug(f"Buffer status for {symbol}: {symbol_status}")
                return False

            # Get buffer data with validation
            self.logger.debug(f"Getting buffer data for {symbol}...")
            data = self.data_collector.get_buffer_data(symbol, min_length=500)

            if data is None or len(data) < 500:
                actual_length = len(data) if data is not None else 0
                self.logger.warning(f"Insufficient buffer data for {symbol}: {actual_length}/500")
                return False
            
            self.logger.debug(f"Got {len(data)} candles for {symbol}, proceeding with analysis...")
            
            # Continue with the rest of your processing logic here...
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            return False
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        log_dir = Path('paper_trader/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use dedicated log files
        debug_log_file = log_dir / 'debug.log'
        trading_decisions_log_file = log_dir / 'trading_decisions.log'

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(debug_log_file, mode='w'),  # Overwrite log file each run
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create a separate logger for trading decisions
        self.trading_logger = logging.getLogger('trading_decisions')
        self.trading_logger.setLevel(logging.INFO)
        trading_handler = logging.FileHandler(trading_decisions_log_file, mode='a')  # Append mode
        trading_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.trading_logger.addHandler(trading_handler)
        
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
                api_secret=self.settings.bitvavo_api_secret,
                interval=self.settings.candle_interval,
                settings=self.settings
            )
            
            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Initialize window-based model loader
            self.model_loader = WindowBasedModelLoader(self.settings.model_path)
            
            # Initialize enhanced ensemble predictor with confidence thresholds
            self.predictor = WindowBasedEnsemblePredictor(
                model_loader=self.model_loader,
                min_confidence_threshold=self.settings.min_confidence_threshold,
                min_signal_strength=self.settings.min_signal_strength,
                settings=self.settings
            )
            
            # Set ensemble weights from settings
            self.predictor.lstm_weight = self.settings.lstm_weight
            self.predictor.xgb_weight = self.settings.xgb_weight
            self.predictor.caboose_weight = self.settings.caboose_weight
            
            # Initialize signal generator
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
                data_collector=self.data_collector,
                max_daily_trades_per_symbol=self.settings.max_daily_trades_per_symbol,
                settings=self.settings
            )
            
            # Initialize exit manager
            self.exit_manager = ExitManager(
                trailing_stop_pct=self.settings.trailing_stop_pct,
                max_hold_hours=self.settings.max_hold_hours,
                enable_prediction_exits=self.settings.enable_prediction_exits,
                prediction_exit_min_confidence=self.settings.prediction_exit_min_confidence,
                prediction_exit_min_strength=self.settings.prediction_exit_min_strength,
                dynamic_stop_loss_adjustment=self.settings.dynamic_stop_loss_adjustment,
                settings=self.settings
            )
            
            # Initialize portfolio manager
            self.portfolio_manager = PortfolioManager(
                initial_capital=self.settings.initial_capital,
                max_positions_per_symbol=self.settings.max_positions_per_symbol
            )
            
            # Initialize Telegram notifier
            self.telegram_notifier = TelegramNotifier(
                bot_token=self.settings.telegram_bot_token,
                chat_id=self.settings.telegram_chat_id
            )

            # Initialize WebSocket server for live prediction updates
            self.ws_server = PredictionWebSocketServer(
                host=self.settings.websocket_server_host,
                port=self.settings.websocket_server_port
            )
            await self.ws_server.start()

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

            # Debug data pipeline for all symbols
            for symbol in self.settings.symbols:
                await self.debug_data_pipeline(symbol)
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    async def process_symbol(self, symbol: str) -> bool:
        """Process trading signals for a single symbol."""
        try:
            self.trading_logger.info(f"🔍 PROCESSING SYMBOL: {symbol}")
            
            # Skip if no models for this symbol
            if (symbol not in self.model_loader.lstm_models and
                symbol not in self.model_loader.xgb_models):
                self.trading_logger.warning(f"❌ NO MODELS AVAILABLE for {symbol}")
                return False

            # Refresh latest market price to avoid stale data
            await self.data_collector.refresh_latest_price(symbol)
            
            # Ensure we have sufficient data before proceeding
            if not await self.data_collector.ensure_sufficient_data(symbol, min_length=500):
                self.trading_logger.warning(f"❌ INSUFFICIENT DATA for {symbol} (need 500+ candles)")
                return False

            # Fetch latest data from buffer for feature engineering
            data = self.data_collector.get_buffer_data(symbol, min_length=500)
            if data is None or len(data) < 500:
                self.trading_logger.warning(f"❌ INSUFFICIENT BUFFER DATA for {symbol}: {len(data) if data is not None else 0}/500")
                return False
            
            # Feature engineering
            features_df = self.feature_engineer.engineer_features(data)
            if features_df is None or len(features_df) < self.settings.sequence_length:
                actual_features = len(features_df) if features_df is not None else 0
                self.trading_logger.warning(f"❌ INSUFFICIENT FEATURES for {symbol}: {actual_features}/{self.settings.sequence_length}")
                return False
            
            # Refresh buffer price first, then get most current price for trading
            await self.data_collector.refresh_latest_price(symbol)
            current_price = await self.data_collector.get_current_price_for_trading(symbol)
            if current_price is None:
                self.trading_logger.warning(f"❌ COULD NOT GET CURRENT PRICE for {symbol}")
                return False
            
            self.trading_logger.info(f"📊 DATA READY for {symbol}: {len(data)} candles, {len(features_df)} features, price: €{current_price}")
            
            # Calculate market volatility for enhanced prediction
            price_data = features_df['close'].tail(20)  # Use last 20 periods
            market_volatility = price_data.std() / price_data.mean() if len(price_data) > 1 else 0.5
            
            # Generate enhanced prediction with market volatility and current price
            prediction_result = await self.predictor.predict(
                symbol, features_df, market_volatility, current_price
            )
            if prediction_result is None:
                self.trading_logger.warning(f"❌ PREDICTION FAILED for {symbol}")
                return False

            # Log prediction result for debugging
            self.trading_logger.info(f"🧠 PREDICTION for {symbol}: {prediction_result}")

            # Broadcast prediction via WebSocket
            if self.ws_server:
                await self.ws_server.broadcast({
                    'type': 'prediction',
                    'symbol': symbol,
                    'data': prediction_result
                })
            
            # Check if prediction meets trading thresholds
            if not self.predictor._meets_trading_threshold(
                prediction_result['confidence'], 
                prediction_result['signal_strength'],
                prediction_result
            ):
                self.trading_logger.warning(f"❌ PREDICTION THRESHOLD NOT MET for {symbol}")
                return False
            
            # Check if we already have too many positions for this symbol
            current_positions = self.portfolio_manager.positions.get(symbol, [])
            if len(current_positions) >= self.settings.max_positions_per_symbol:
                self.trading_logger.warning(f"❌ MAX POSITIONS REACHED for {symbol}: {len(current_positions)}/{self.settings.max_positions_per_symbol}")
                return False  # Skip if limit reached
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                symbol=symbol,
                prediction=prediction_result,
                current_price=current_price,
                portfolio=self.portfolio_manager
            )
            
            if signal and signal['action'] == 'BUY':
                self.trading_logger.info(f"💰 EXECUTING BUY SIGNAL for {symbol}")
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
                    
                    self.logger.info(f"Opened position for {symbol}")
                    return True
                else:
                    self.trading_logger.error(f"❌ FAILED TO OPEN POSITION for {symbol}")
            else:
                self.trading_logger.info(f"🚫 NO SIGNAL GENERATED for {symbol}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            self.trading_logger.error(f"❌ ERROR PROCESSING {symbol}: {e}")
            return False
    
    async def check_exit_conditions(self):
        """Check exit conditions for all open positions."""
        try:
            positions_to_close = []
            
            for symbol, pos_list in list(self.portfolio_manager.positions.items()):
                try:
                    # Get current price for exit checks
                    current_price = await self.data_collector.get_current_price_for_trading(symbol)
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

    async def get_prediction_for_symbol(self, symbol: str) -> Optional[dict]:
        """Fetch data and return model prediction for a symbol."""
        try:
            if (symbol not in self.model_loader.lstm_models and
                symbol not in self.model_loader.xgb_models):
                return None

            await self.data_collector.refresh_latest_price(symbol)

            if not await self.data_collector.ensure_sufficient_data(symbol, min_length=500):
                return None

            data = self.data_collector.get_buffer_data(symbol, min_length=500)
            if data is None or len(data) < 500:
                return None

            features_df = self.feature_engineer.engineer_features(data)
            if features_df is None or len(features_df) < self.settings.sequence_length:
                return None

            await self.data_collector.refresh_latest_price(symbol)
            current_price = await self.data_collector.get_current_price_for_trading(symbol)
            if current_price is None:
                return None

            price_data = features_df['close'].tail(20)
            market_volatility = price_data.std() / price_data.mean() if len(price_data) > 1 else 0.5

            prediction_result = await self.predictor.predict(
                symbol, features_df, market_volatility, current_price
            )

            if prediction_result is not None and self.ws_server:
                await self.ws_server.broadcast({'type': 'prediction', 'symbol': symbol, 'data': prediction_result})

            return prediction_result

        except Exception as e:
            self.logger.error(f"Error getting prediction for {symbol}: {e}")
            return None

    async def check_existing_positions(self, symbol: str, prediction_result: dict):
        """Check existing positions for prediction-based exits."""
        positions = self.portfolio_manager.positions.get(symbol, [])

        for position in list(positions):
            current_price = await self.data_collector.get_current_price_for_trading(symbol)
            if current_price is None:
                continue

            exit_signal = self.exit_manager.check_exit_conditions(
                position, current_price, prediction_result
            )

            if exit_signal:
                exit_price = exit_signal.get('exit_price', current_price)
                reason = exit_signal['reason']
                success = self.portfolio_manager.close_position(
                    symbol=symbol,
                    position=position,
                    exit_price=exit_price,
                    reason=reason
                )

                if success:
                    pnl = (exit_price - position.entry_price) * position.quantity
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price
                    hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600

                    if reason == 'PREDICTION_REVERSAL':
                        await self.telegram_notifier.send_prediction_exit_alert(
                            symbol=symbol,
                            confidence=exit_signal.get('reversal_confidence', 0),
                            signal_strength=exit_signal.get('reversal_signal', ''),
                            pnl=pnl
                        )
                    else:
                        await self.telegram_notifier.send_position_closed(
                            symbol=symbol,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            quantity=position.quantity,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            exit_reason=reason,
                            hold_time_hours=hold_time
                        )

    async def check_entry_opportunities(self, symbol: str, prediction_result: dict) -> bool:
        """Check for new entry opportunities based on prediction."""
        if not prediction_result:
            self.trading_logger.warning(f"❌ No prediction for {symbol} - skipping entry check")
            return False

        if not self.predictor._meets_trading_threshold(
            prediction_result['confidence'], prediction_result['signal_strength'], prediction_result
        ):
            self.trading_logger.warning(f"❌ Prediction threshold not met for {symbol}")
            return False

        current_positions = self.portfolio_manager.positions.get(symbol, [])
        if len(current_positions) >= self.settings.max_positions_per_symbol:
            self.trading_logger.warning(f"❌ Max positions reached for {symbol}: {len(current_positions)}/{self.settings.max_positions_per_symbol}")
            return False

        current_price = await self.data_collector.get_current_price_for_trading(symbol)
        if current_price is None:
            self.trading_logger.warning(f"❌ Could not get current price for {symbol}")
            return False

        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            prediction=prediction_result,
            current_price=current_price,
            portfolio=self.portfolio_manager
        )

        if signal and signal['action'] == 'BUY':
            self.trading_logger.info(f"💰 Executing BUY signal for {symbol}")
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
                self.trading_logger.error(f"❌ Failed to open position for {symbol}")
        else:
            self.trading_logger.info(f"🚫 No signal generated for {symbol}")
        
        return False

    async def run_trading_cycle(self):
        """Enhanced trading cycle with prediction monitoring."""
        self.trading_logger.info("🔄 STARTING TRADING CYCLE")
        total_symbols = 0
        successful_trades = 0
        failed_signals = 0
        
        for symbol in self.settings.symbols:
            total_symbols += 1
            try:
                self.trading_logger.info(f"🔍 Analyzing {symbol}...")
                prediction_result = await self.get_prediction_for_symbol(symbol)
                
                if prediction_result:
                    self.trading_logger.info(f"📊 Got prediction for {symbol}: {prediction_result}")
                else:
                    self.trading_logger.warning(f"❌ No prediction available for {symbol}")
                    failed_signals += 1
                    continue
                
                await self.check_existing_positions(symbol, prediction_result)
                
                entry_result = await self.check_entry_opportunities(symbol, prediction_result)
                if entry_result:
                    successful_trades += 1
                else:
                    failed_signals += 1
                    
            except Exception as e:
                self.logger.error(f"Error in trading cycle for {symbol}: {e}")
                self.trading_logger.error(f"❌ ERROR in trading cycle for {symbol}: {e}")
                failed_signals += 1
        
        # Get current portfolio status
        total_positions = sum(len(positions) for positions in self.portfolio_manager.positions.values())
        available_capital = self.portfolio_manager.get_available_capital()
        
        self.trading_logger.info(f"🔄 TRADING CYCLE COMPLETE:")
        self.trading_logger.info(f"   📊 Symbols analyzed: {total_symbols}")
        self.trading_logger.info(f"   ✅ Successful trades: {successful_trades}")
        self.trading_logger.info(f"   ❌ Failed signals: {failed_signals}")
        self.trading_logger.info(f"   💼 Active positions: {total_positions}")
        self.trading_logger.info(f"   💰 Available capital: €{available_capital:.2f}")
    
    async def send_hourly_update(self):
        """Send hourly portfolio update."""
        try:
            # Get current prices for all positions
            current_prices = {}
            for symbol in self.portfolio_manager.positions.keys():
                price = await self.data_collector.get_current_price_for_trading(symbol)
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
            self.trading_logger.info("🚀 PAPER TRADER STARTING UP")
            
            # Log current settings for verification
            self.trading_logger.info("⚙️ CURRENT SETTINGS:")
            self.trading_logger.info(f"   📊 Min Confidence: {self.settings.min_confidence_threshold}")
            self.trading_logger.info(f"   💪 Min Signal Strength: {self.settings.min_signal_strength}")
            self.trading_logger.info(f"   📈 Min Expected Gain: {self.settings.min_expected_gain_pct}")
            self.trading_logger.info(f"   🔒 Strict Entry Conditions: {self.settings.enable_strict_entry_conditions}")
            self.trading_logger.info(f"   🎯 Max Prediction Uncertainty: {self.settings.max_prediction_uncertainty}")
            self.trading_logger.info(f"   🤝 Min Ensemble Agreement: {self.settings.min_ensemble_agreement_count}")
            self.trading_logger.info(f"   📊 Trend Strength Threshold: {self.settings.trend_strength_threshold}")
            self.trading_logger.info(f"   💰 Initial Capital: €{self.settings.initial_capital}")
            self.trading_logger.info(f"   📋 Symbols: {self.settings.symbols}")
            
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

            # Ensure each buffer has enough historical data **before**
            # starting the real-time feed to avoid overwriting fresh updates
            for symbol in self.settings.symbols:
                await self.data_collector.ensure_sufficient_data(symbol, min_length=500)

            # Start WebSocket data feed and periodic API updates in background
            self.logger.info("Starting real-time data feed...")
            self.data_feed_task = asyncio.create_task(
                self.data_collector.start_websocket_feed(self.settings.symbols)
            )
            self.api_update_task = asyncio.create_task(
                self.data_collector.update_data_periodically(
                    self.settings.symbols,
                    interval_minutes=self.settings.api_update_interval_minutes  # Use settings value (15 min)
                )
            )

            # Initialize last prediction timestamps
            for symbol in self.settings.symbols:
                self.last_prediction_time[symbol] = self.data_collector.last_update.get(symbol)

            self.logger.info("Paper Trader is now running...")
            self.trading_logger.info("✅ PAPER TRADER READY FOR TRADING")

            # Main trading loop
            cycle_count = 0
            while self.is_running:
                try:
                    cycle_count += 1
                    self.trading_logger.info(f"🔄 CYCLE #{cycle_count}")
                    
                    await self.run_trading_cycle()

                    # Send hourly updates
                    now = datetime.now()
                    if now - self.last_hourly_update >= timedelta(hours=1):
                        await self.send_hourly_update()
                        self.last_hourly_update = now
                    
                    # Wait before next cycle (1 minute to align with data interval)
                    if self.is_running:
                        self.logger.debug("Waiting for next cycle...")
                        await asyncio.sleep(60)  # 1 minute for debugging
                
                except KeyboardInterrupt:
                    self.logger.info("Received shutdown signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.trading_logger.error(f"❌ MAIN LOOP ERROR: {e}")
                    await self.telegram_notifier.send_error_alert(str(e), "Main Loop")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        except Exception as e:
            self.logger.error(f"Critical error in Paper Trader: {e}")
            self.trading_logger.error(f"🚨 CRITICAL ERROR: {e}")
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

            # Stop WebSocket server
            if self.ws_server:
                await self.ws_server.stop()

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

if __name__ == "__main__":
    # Run the paper trader
    try:
        asyncio.run(main())
    except Exception as e:
        # The logger may not be initialized, so we print to stderr
        logging.basicConfig()
        log = logging.getLogger(__name__)
        log.error("A fatal error occurred.", exc_info=True)
        sys.exit(1)