"""Enhanced main paper trader script with comprehensive improvements."""

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

# Enhanced utilities
from paper_trader.utils import (
    TradingCircuitBreaker,
    CircuitBreakerConfig,
    SystemRecoveryManager,
    HealthMonitor,
    DataQualityMonitor,
    PerformanceMonitor,
    EnhancedFeatureCache,
    SmartNotificationManager,
    NotificationPriority,
    NotificationType
)


class EnhancedPaperTrader:
    """Enhanced paper trading orchestrator with comprehensive improvements."""
    
    def __init__(self):
        # Load settings
        self.settings = TradingSettings()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize core components
        self.data_collector = None
        self.feature_engineer = None
        self.model_loader = None
        self.predictor = None
        self.signal_generator = None
        self.exit_manager = None
        self.portfolio_manager = None
        self.telegram_notifier = None
        self.ws_server = None

        # Enhanced components
        self.performance_monitor = PerformanceMonitor()
        self.feature_cache = None
        self.data_quality_monitor = DataQualityMonitor()
        self.health_monitor = HealthMonitor()
        self.notification_manager = None
        self.recovery_manager = None
        
        # Circuit breakers for critical operations
        self.circuit_breakers = {
            'prediction': TradingCircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=180)),
            'data_collection': TradingCircuitBreaker(CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120)),
            'trading': TradingCircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=300)),
            'notifications': TradingCircuitBreaker(CircuitBreakerConfig(failure_threshold=10, recovery_timeout=60))
        }

        # Background tasks
        self.data_feed_task = None
        self.api_update_task = None
        self.health_check_task = None
        self.performance_report_task = None

        # Track when a symbol was last processed
        self.last_prediction_time: Dict[str, datetime] = {}
        
        # State tracking
        self.last_hourly_update = datetime.now()
        self.last_health_check = datetime.now()
        self.is_running = False
        
        # Skip symbols temporarily on errors
        self.skip_symbols: Dict[str, datetime] = {}
        
        self.logger.info("Enhanced Paper Trader initialized")

    def _setup_logging(self):
        """Setup enhanced logging configuration."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        log_dir = Path('paper_trader/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Multiple log files for different purposes
        main_log = log_dir / 'enhanced_paper_trader.log'
        error_log = log_dir / 'errors.log'
        performance_log = log_dir / 'performance.log'

        # Configure main logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(main_log, mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Error logger
        error_handler = logging.FileHandler(error_log, mode='a')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Performance logger
        perf_handler = logging.FileHandler(performance_log, mode='a')
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        ))
        
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(error_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger('performance')
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
        
        # Reduce noise from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('telegram').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
    
    async def initialize(self):
        """Initialize all components with enhanced error handling."""
        try:
            self.logger.info("Initializing Enhanced Paper Trader components...")
            
            # Initialize data collector with circuit breaker
            self.data_collector = await self.circuit_breakers['data_collection'].call(
                self._initialize_data_collector
            )
            
            # Initialize feature engineer and cache
            self.feature_engineer = FeatureEngineer()
            self.feature_cache = EnhancedFeatureCache(self.performance_monitor)
            
            # Initialize model loader
            self.model_loader = WindowBasedModelLoader(self.settings.model_path)
            
            # Initialize enhanced ensemble predictor
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
                data_collector=self.data_collector,
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
            
            self.notification_manager = SmartNotificationManager(self.telegram_notifier)
            
            # Initialize recovery manager
            self.recovery_manager = SystemRecoveryManager(self)

            # Initialize WebSocket server
            self.ws_server = PredictionWebSocketServer()
            await self.ws_server.start()

            # Test connections
            if self.telegram_notifier.is_enabled():
                await self.telegram_notifier.test_connection()
            
            self.logger.info("All enhanced components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced components: {e}")
            await self.notification_manager.send_critical_alert(
                "Initialization Failed",
                f"Failed to initialize paper trader: {str(e)}"
            )
            raise
    
    async def _initialize_data_collector(self):
        """Initialize data collector separately for circuit breaker."""
        return BitvavoDataCollector(
            api_key=self.settings.bitvavo_api_key,
            api_secret=self.settings.bitvavo_api_secret,
            interval=self.settings.candle_interval,
            settings=self.settings
        )
    
    async def load_models(self):
        """Load ML models with enhanced error handling."""
        import tensorflow as tf
        
        try:
            self.logger.info("Loading ML models with enhanced monitoring...")
            
            loaded_count = 0
            failed_symbols = []
            
            for symbol in self.settings.symbols:
                try:
                    success = await self.performance_monitor.monitor_operation(
                        f"load_model_{symbol}",
                        self.model_loader.load_symbol_models,
                        symbol,
                        custom_objects={"directional_loss": directional_loss}
                    )
                    
                    if success:
                        loaded_count += 1
                        self.logger.info(f"Models loaded for {symbol}")
                        
                        # Validate data pipeline for this symbol
                        await self._validate_symbol_pipeline(symbol)
                    else:
                        failed_symbols.append(symbol)
                        self.logger.warning(f"No models found for {symbol}")
                        
                except Exception as e:
                    failed_symbols.append(symbol)
                    self.logger.error(f"Error loading models for {symbol}: {e}")
                    await self.recovery_manager.handle_system_failure(
                        'model_loading_failed', 
                        {'symbol': symbol, 'error': str(e)}
                    )
            
            # Send notification about model loading
            await self.notification_manager.send_notification(
                NotificationType.SYSTEM_STATUS,
                "Model Loading Complete",
                f"Successfully loaded models for {loaded_count}/{len(self.settings.symbols)} symbols",
                NotificationPriority.HIGH if failed_symbols else NotificationPriority.NORMAL,
                {'loaded_count': loaded_count, 'failed_symbols': failed_symbols}
            )
            
            self.logger.info(f"Model loading complete: {loaded_count} successful, {len(failed_symbols)} failed")
            
        except Exception as e:
            self.logger.error(f"Critical error loading models: {e}")
            await self.notification_manager.send_critical_alert(
                "Model Loading Failed",
                f"Critical error during model loading: {str(e)}"
            )
            raise
    
    async def _validate_symbol_pipeline(self, symbol: str):
        """Validate the complete data pipeline for a symbol."""
        try:
            # Get sample data
            data = self.data_collector.get_buffer_data(symbol, min_length=250)
            if data is None or len(data) < 250:
                self.logger.warning(f"Insufficient data for {symbol} pipeline validation")
                return False
            
            # Validate data quality
            quality_report = self.data_quality_monitor.validate_data_quality(symbol, data)
            
            if not quality_report.is_tradeable:
                self.logger.warning(f"Data quality issues for {symbol}: {quality_report.issues}")
                await self.notification_manager.send_notification(
                    NotificationType.ERROR,
                    f"Data Quality Warning - {symbol}",
                    f"Quality score: {quality_report.quality_score:.1f}/100",
                    NotificationPriority.NORMAL
                )
            
            # Test feature engineering
            features = await self.performance_monitor.monitor_operation(
                f"feature_engineering_{symbol}",
                self.feature_engineer.engineer_features,
                data
            )
            
            if features is None:
                self.logger.error(f"Feature engineering failed for {symbol}")
                return False
            
            self.logger.info(f"Pipeline validation successful for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline validation failed for {symbol}: {e}")
            return False
    
    async def process_symbol_enhanced(self, symbol: str) -> bool:
        """Enhanced symbol processing with comprehensive monitoring."""
        
        # Check if symbol should be skipped
        if symbol in self.skip_symbols:
            if datetime.now() < self.skip_symbols[symbol]:
                return False
            else:
                del self.skip_symbols[symbol]
        
        try:
            return await self.circuit_breakers['prediction'].call(
                self._process_symbol_core,
                symbol
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            self.health_monitor.record_error()
            
            # Skip symbol temporarily on repeated errors
            self.skip_symbols[symbol] = datetime.now() + timedelta(minutes=5)
            
            await self.recovery_manager.handle_system_failure(
                'prediction_failure',
                {'symbol': symbol, 'error': str(e)}
            )
            
            return False
    
    async def _process_symbol_core(self, symbol: str) -> bool:
        """Core symbol processing logic."""
        # Check if models are available
        if (symbol not in getattr(self.model_loader, 'lstm_models', {}) and 
            symbol not in getattr(self.model_loader, 'xgb_models', {})):
            return False

        # Ensure sufficient data
        if not await self.data_collector.ensure_sufficient_data(symbol, min_length=500):
            return False

        # Get buffer data
        data = self.data_collector.get_buffer_data(symbol, min_length=500)
        if data is None or len(data) < 500:
            return False

        # Check data quality
        quality_report = self.data_quality_monitor.validate_data_quality(symbol, data)
        if not quality_report.is_tradeable:
            self.logger.warning(f"Data quality insufficient for trading {symbol}")
            return False

        # Feature engineering with caching
        data_hash = self.feature_cache.calculate_data_hash(data)
        features_df = await self.feature_cache.get_cached_features(symbol, data_hash)
        
        if features_df is None:
            features_df = await self.performance_monitor.monitor_operation(
                f"feature_engineering_{symbol}",
                self.feature_engineer.engineer_features,
                data
            )
            
            if features_df is not None:
                await self.feature_cache.cache_features(symbol, data_hash, features_df)
        else:
            self.logger.debug(f"Using cached features for {symbol}")

        if features_df is None or len(features_df) < self.settings.sequence_length:
            return False

        # Get current price
        await self.data_collector.refresh_latest_price(symbol)
        current_price = await self.data_collector.get_current_price_for_trading(symbol)
        if current_price is None:
            return False

        # Calculate market volatility
        price_data = features_df['close'].tail(20)
        market_volatility = price_data.std() / price_data.mean() if len(price_data) > 1 else 0.5

        # Generate prediction with monitoring
        prediction_result = await self.performance_monitor.monitor_operation(
            f"prediction_{symbol}",
            self.predictor.predict,
            symbol, features_df, market_volatility, current_price
        )
        
        if prediction_result is None:
            return False

        # Record successful prediction
        self.health_monitor.record_prediction()
        self.last_prediction_time[symbol] = datetime.now()

        # Broadcast prediction
        if self.ws_server:
            await self.ws_server.broadcast({
                'type': 'prediction',
                'symbol': symbol,
                'data': prediction_result,
                'data_quality': quality_report.quality_score
            })

        # Check trading thresholds
        if not self.predictor._meets_trading_threshold(
            prediction_result['confidence'], 
            prediction_result['signal_strength']
        ):
            return False

        # Check position limits
        current_positions = self.portfolio_manager.positions.get(symbol, [])
        if len(current_positions) >= self.settings.max_positions_per_symbol:
            return False

        # Generate and execute signal
        return await self._execute_trading_signal(symbol, prediction_result, current_price)
    
    async def _execute_trading_signal(self, symbol: str, prediction_result: dict, current_price: float) -> bool:
        """Execute trading signal with enhanced monitoring."""
        try:
            return await self.circuit_breakers['trading'].call(
                self._execute_trading_signal_core,
                symbol, prediction_result, current_price
            )
        except Exception as e:
            self.logger.error(f"Error executing trading signal for {symbol}: {e}")
            return False
    
    async def _execute_trading_signal_core(self, symbol: str, prediction_result: dict, current_price: float) -> bool:
        """Core trading signal execution logic."""
        signal = self.signal_generator.generate_signal(
            symbol=symbol,
            prediction=prediction_result,
            current_price=current_price,
            portfolio=self.portfolio_manager
        )
        
        if signal and signal['action'] == 'BUY':
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
                # Send enhanced trade notification
                await self.notification_manager.send_trade_notification(
                    action="BUY",
                    symbol=symbol,
                    price=current_price,
                    quantity=signal['quantity']
                )
                
                self.logger.info(f"Opened position for {symbol} - Value: ${signal['position_value']:.2f}")
                return True
        
        return False
    
    async def check_exit_conditions_enhanced(self):
        """Enhanced exit condition checking with comprehensive monitoring."""
        try:
            positions_to_close = []
            
            for symbol, pos_list in list(self.portfolio_manager.positions.items()):
                try:
                    current_price = await self.data_collector.get_current_price_for_trading(symbol)
                    if current_price is None:
                        continue

                    for position in list(pos_list):
                        exit_result = self.exit_manager.check_exit_conditions(position, current_price)

                        if exit_result:
                            exit_price = exit_result.get('exit_price', current_price)
                            exit_reason = exit_result['reason']
                            positions_to_close.append((symbol, position, exit_price, exit_reason))
                
                except Exception as e:
                    self.logger.error(f"Error checking exit for {symbol}: {e}")
            
            # Close positions
            for symbol, position, exit_price, exit_reason in positions_to_close:
                try:
                    success = self.portfolio_manager.close_position(
                        symbol=symbol,
                        position=position,
                        exit_price=exit_price,
                        reason=exit_reason
                    )
                    
                    if success:
                        pnl = (exit_price - position.entry_price) * position.quantity
                        
                        # Send enhanced exit notification
                        await self.notification_manager.send_trade_notification(
                            action="SELL",
                            symbol=symbol,
                            price=exit_price,
                            quantity=position.quantity,
                            pnl=pnl,
                            reason=exit_reason
                        )
                        
                        self.logger.info(f"Closed position for {symbol} - P&L: ${pnl:.2f}")
                
                except Exception as e:
                    self.logger.error(f"Error closing position for {symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in enhanced exit condition checking: {e}")
    
    async def run_health_checks(self):
        """Run comprehensive health checks."""
        try:
            await self.health_monitor.collect_metrics(self)
            health_status = self.health_monitor.get_health_status()
            
            # Send health alert if needed
            if health_status['health_score'] < 70:
                await self.notification_manager.send_health_alert(
                    health_status['overall_status'],
                    health_status['health_score'],
                    health_status['issues']
                )
            
            self.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error in health checks: {e}")
    
    async def send_enhanced_hourly_update(self):
        """Send enhanced hourly portfolio update."""
        try:
            # Get current prices
            current_prices = {}
            for symbol in self.portfolio_manager.positions.keys():
                price = await self.data_collector.get_current_price_for_trading(symbol)
                if price:
                    current_prices[symbol] = price
            
            # Get portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary(current_prices)
            
            # Get performance metrics
            perf_summary = self.performance_monitor.get_recent_performance(60)  # Last hour
            
            # Get top performers
            top_performers = []
            for symbol, positions in self.portfolio_manager.positions.items():
                if positions and symbol in current_prices:
                    total_pnl = sum(
                        (current_prices[symbol] - pos.entry_price) * pos.quantity 
                        for pos in positions
                    )
                    total_value = sum(pos.entry_price * pos.quantity for pos in positions)
                    pnl_pct = (total_pnl / total_value) * 100 if total_value > 0 else 0
                    
                    top_performers.append({
                        'symbol': symbol,
                        'pnl': total_pnl,
                        'pnl_pct': pnl_pct
                    })
            
            top_performers.sort(key=lambda x: x['pnl_pct'], reverse=True)
            
            # Send enhanced notification
            await self.notification_manager.send_portfolio_update(
                portfolio_summary['total_value'],
                portfolio_summary['daily_pnl'],
                len([pos for positions in self.portfolio_manager.positions.values() for pos in positions]),
                top_performers[:3]
            )
            
            # Log performance summary
            self.perf_logger.info(f"Hourly Performance: {perf_summary}")
            
            self.logger.info("Sent enhanced hourly portfolio update")
            
        except Exception as e:
            self.logger.error(f"Error sending enhanced hourly update: {e}")
    
    async def run_enhanced_trading_cycle(self):
        """Enhanced trading cycle with comprehensive monitoring."""
        cycle_start = datetime.now()
        
        try:
            processed_symbols = 0
            successful_predictions = 0
            
            for symbol in self.settings.symbols:
                try:
                    success = await self.process_symbol_enhanced(symbol)
                    processed_symbols += 1
                    if success:
                        successful_predictions += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in enhanced trading cycle for {symbol}: {e}")
            
            # Check exit conditions
            await self.check_exit_conditions_enhanced()
            
            # Log cycle performance
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            self.perf_logger.info(
                f"Trading cycle: {cycle_time:.2f}s, "
                f"processed: {processed_symbols}, "
                f"predictions: {successful_predictions}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trading cycle: {e}")
            await self.notification_manager.send_notification(
                NotificationType.ERROR,
                "Trading Cycle Error",
                f"Error in trading cycle: {str(e)}",
                NotificationPriority.HIGH
            )
    
    async def start_background_tasks(self):
        """Start all background monitoring tasks."""
        # Health check task
        async def health_check_loop():
            while self.is_running:
                try:
                    await self.run_health_checks()
                    await asyncio.sleep(300)  # Every 5 minutes
                except Exception as e:
                    self.logger.error(f"Health check task error: {e}")
                    await asyncio.sleep(60)
        
        # Performance reporting task
        async def performance_report_loop():
            while self.is_running:
                try:
                    await asyncio.sleep(3600)  # Every hour
                    perf_report = self.performance_monitor.generate_performance_report()
                    self.logger.info(f"Performance Report:\n{perf_report}")
                except Exception as e:
                    self.logger.error(f"Performance report task error: {e}")
        
        self.health_check_task = asyncio.create_task(health_check_loop())
        self.performance_report_task = asyncio.create_task(performance_report_loop())
    
    async def run(self):
        """Enhanced main trading loop."""
        try:
            self.logger.info("Starting Enhanced Paper Trader...")
            
            # Initialize components
            await self.initialize()
            
            # Load models
            await self.load_models()
            
            # Send enhanced startup notification
            startup_data = {
                'symbols': len(self.settings.symbols),
                'initial_capital': self.settings.initial_capital,
                'max_positions': self.settings.max_positions
            }
            
            await self.notification_manager.send_notification(
                NotificationType.SYSTEM_STATUS,
                "🚀 Enhanced Paper Trader Started",
                f"Started with {len(self.settings.symbols)} symbols and ${self.settings.initial_capital:,.0f} initial capital",
                NotificationPriority.HIGH,
                startup_data
            )
            
            self.is_running = True
            
            # Initialize data buffers
            self.logger.info("Initializing enhanced data buffers...")
            await self.data_collector.initialize_buffers(self.settings.symbols)

            # Ensure sufficient data for all symbols
            for symbol in self.settings.symbols:
                await self.data_collector.ensure_sufficient_data(symbol, min_length=500)

            # Start background tasks
            await self.start_background_tasks()
            
            # Start data feed tasks
            self.data_feed_task = asyncio.create_task(
                self.data_collector.start_websocket_feed(self.settings.symbols)
            )
            self.api_update_task = asyncio.create_task(
                self.data_collector.update_data_periodically(
                    self.settings.symbols,
                    interval_minutes=self.settings.api_update_interval_minutes  # Use settings value (15 min)
                )
            )

            # Initialize prediction timestamps
            for symbol in self.settings.symbols:
                self.last_prediction_time[symbol] = self.data_collector.last_update.get(symbol)

            self.logger.info("Enhanced Paper Trader is now running...")

            # Main trading loop
            while self.is_running:
                try:
                    await self.run_enhanced_trading_cycle()

                    # Send hourly updates
                    now = datetime.now()
                    if now - self.last_hourly_update >= timedelta(hours=1):
                        await self.send_enhanced_hourly_update()
                        self.last_hourly_update = now
                    
                    # Wait for next cycle
                    if self.is_running:
                        await asyncio.sleep(60)  # 1 minute cycles
                
                except KeyboardInterrupt:
                    self.logger.info("Received shutdown signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in enhanced main loop: {e}")
                    await self.notification_manager.send_critical_alert(
                        "Main Loop Error",
                        f"Critical error in main loop: {str(e)}"
                    )
                    await asyncio.sleep(60)
        
        except Exception as e:
            self.logger.error(f"Critical error in Enhanced Paper Trader: {e}")
            await self.notification_manager.send_critical_alert(
                "System Critical Error",
                f"Critical system error: {str(e)}"
            )
            raise
        
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Enhanced graceful shutdown."""
        try:
            self.logger.info("Shutting down Enhanced Paper Trader...")
            self.is_running = False

            # Cancel background tasks
            tasks = [
                self.data_feed_task, 
                self.api_update_task,
                self.health_check_task,
                self.performance_report_task
            ]
            
            for task in [t for t in tasks if t]:
                task.cancel()
            
            for task in [t for t in tasks if t]:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            
            # Send final reports
            await self.send_enhanced_hourly_update()
            
            # Generate final performance report
            final_report = self.performance_monitor.generate_performance_report()
            self.logger.info(f"Final Performance Report:\n{final_report}")
            
            # Send shutdown notification
            await self.notification_manager.send_notification(
                NotificationType.SYSTEM_STATUS,
                "🛑 Enhanced Paper Trader Stopped",
                "System has been gracefully shut down",
                NotificationPriority.HIGH
            )
            
            # Shutdown notification manager
            await self.notification_manager.shutdown()

            # Stop WebSocket server
            if self.ws_server:
                await self.ws_server.stop()

            self.logger.info("Enhanced Paper Trader shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during enhanced shutdown: {e}")


async def main():
    """Enhanced main entry point."""
    trader = EnhancedPaperTrader()
    
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
        log.error("A fatal error occurred in Enhanced Paper Trader.", exc_info=True)
        sys.exit(1)