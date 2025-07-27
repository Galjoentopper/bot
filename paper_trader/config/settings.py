"""Trading settings and configuration management."""

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelSettings:
    """Model-specific configuration settings."""
    
    # Model Configuration  
    model_path: str = os.getenv('MODEL_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'))
    # Sequence length: 96 candles = 24 hours of 15-minute data (matches training)
    sequence_length: int = int(os.getenv('SEQUENCE_LENGTH', '96'))
    
    # Window-based Model Configuration
    min_window: int = int(os.getenv('MIN_WINDOW', '3'))
    max_window: int = int(os.getenv('MAX_WINDOW', '41'))
    default_window: int = int(os.getenv('DEFAULT_WINDOW', '15'))
    
    # Prediction Confidence Thresholds
    min_confidence_threshold: float = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7'))
    min_signal_strength: str = os.getenv('MIN_SIGNAL_STRENGTH', 'MODERATE')
    
    # Ensemble Model Weights
    lstm_weight: float = float(os.getenv('LSTM_WEIGHT', '0.6'))
    xgb_weight: float = float(os.getenv('XGB_WEIGHT', '0.4'))
    caboose_weight: float = float(os.getenv('CABOOSE_WEIGHT', '0.3'))

@dataclass  
class TradingSettingsConfig:
    """Trading-specific configuration settings."""
    
    # Trading Parameters
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '10000.0'))
    max_positions: int = int(os.getenv('MAX_POSITIONS', '10'))
    # Maximum simultaneous positions allowed per symbol
    max_positions_per_symbol: int = int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '1'))
    base_position_size: float = float(os.getenv('BASE_POSITION_SIZE', '0.08'))
    max_position_size: float = float(os.getenv('MAX_POSITION_SIZE', '0.15'))
    min_position_size: float = float(os.getenv('MIN_POSITION_SIZE', '0.02'))
    take_profit_pct: float = float(os.getenv('TAKE_PROFIT_PCT', '0.015'))
    stop_loss_pct: float = float(os.getenv('STOP_LOSS_PCT', '0.008'))
    trailing_stop_pct: float = float(os.getenv('TRAILING_STOP_PCT', '0.006'))
    min_profit_for_trailing: float = float(os.getenv('MIN_PROFIT_FOR_TRAILING', '0.005'))
    max_hold_hours: int = int(os.getenv('MAX_HOLD_HOURS', '2'))
    min_hold_time_minutes: int = int(os.getenv('MIN_HOLD_TIME_MINUTES', '10'))
    position_cooldown_minutes: int = int(os.getenv('POSITION_COOLDOWN_MINUTES', '5'))
    max_daily_trades_per_symbol: int = int(os.getenv('MAX_DAILY_TRADES_PER_SYMBOL', '50'))

    # Data interval for candles - using 15m to match ML model training data
    candle_interval: str = os.getenv('CANDLE_INTERVAL', '15m')

@dataclass
class TradingSettings:
    """Configuration settings for the paper trader."""
    
    # API Configuration
    bitvavo_api_key: str = os.getenv('BITVAVO_API_KEY', '')
    bitvavo_api_secret: str = os.getenv('BITVAVO_API_SECRET', '')
    
    # Telegram Configuration
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Symbols
    symbols: List[str] = None
    
    # Configuration objects
    model_settings: ModelSettings = None
    trading_settings: TradingSettingsConfig = None

    # Risk Management
    max_daily_loss_pct: float = float(os.getenv('MAX_DAILY_LOSS_PCT', '0.05'))
    max_drawdown_pct: float = float(os.getenv('MAX_DRAWDOWN_PCT', '0.10'))
    # Minimum expected gain (prediction threshold)
    min_expected_gain_pct: float = float(os.getenv('MIN_EXPECTED_GAIN_PCT', '0.001'))

    # Prediction-based exit parameters
    enable_prediction_exits: bool = os.getenv('ENABLE_PREDICTION_EXITS', 'True') == 'True'
    prediction_exit_min_confidence: float = float(os.getenv('PREDICTION_EXIT_MIN_CONFIDENCE', '0.8'))
    prediction_exit_min_strength: str = os.getenv('PREDICTION_EXIT_MIN_STRENGTH', 'STRONG')
    prediction_lookback_periods: int = int(os.getenv('PREDICTION_LOOKBACK_PERIODS', '5'))
    dynamic_stop_loss_adjustment: bool = os.getenv('DYNAMIC_STOP_LOSS_ADJUSTMENT', 'True') == 'True'
    
    # API Configuration
    bitvavo_base_url: str = os.getenv('BITVAVO_BASE_URL', 'https://api.bitvavo.com/v2')
    bitvavo_ws_url: str = os.getenv('BITVAVO_WS_URL', 'wss://ws.bitvavo.com/v2')
    
    # Data Collection Configuration (optimized for 15-minute candles)
    data_cache_max_size: int = int(os.getenv('DATA_CACHE_MAX_SIZE', '1000'))
    buffer_initialization_limit: int = int(os.getenv('BUFFER_INITIALIZATION_LIMIT', '300'))
    min_data_length: int = int(os.getenv('MIN_DATA_LENGTH', '250'))
    max_buffer_size: int = int(os.getenv('MAX_BUFFER_SIZE', '500'))
    # Update API every 15 minutes to align with candle intervals
    api_update_interval_minutes: int = int(os.getenv('API_UPDATE_INTERVAL_MINUTES', '15'))
    sufficient_data_multiplier: int = int(os.getenv('SUFFICIENT_DATA_MULTIPLIER', '2'))
    healthy_buffer_threshold: int = int(os.getenv('HEALTHY_BUFFER_THRESHOLD', '100'))
    
    # Network Configuration (adjusted for 15-minute data intervals)
    api_timeout_seconds: int = int(os.getenv('API_TIMEOUT_SECONDS', '20'))
    price_api_timeout_seconds: int = int(os.getenv('PRICE_API_TIMEOUT_SECONDS', '5'))
    # WebSocket sleep adjusted for 15-minute intervals - check more frequently than candle updates
    websocket_sleep_seconds: int = int(os.getenv('WEBSOCKET_SLEEP_SECONDS', '300'))  # 5 minutes
    api_retry_delay_min: float = float(os.getenv('API_RETRY_DELAY_MIN', '0.2'))
    api_retry_delay_max: float = float(os.getenv('API_RETRY_DELAY_MAX', '0.6'))
    
    # WebSocket Server Configuration
    websocket_server_host: str = os.getenv('WEBSOCKET_SERVER_HOST', '0.0.0.0')
    websocket_server_port: int = int(os.getenv('WEBSOCKET_SERVER_PORT', '8765'))
    
    # Signal Generation Configuration
    confidence_multiplier_min: float = float(os.getenv('CONFIDENCE_MULTIPLIER_MIN', '0.7'))
    confidence_multiplier_max: float = float(os.getenv('CONFIDENCE_MULTIPLIER_MAX', '1.0'))
    very_strong_signal_multiplier: float = float(os.getenv('VERY_STRONG_SIGNAL_MULTIPLIER', '1.2'))
    strong_signal_multiplier: float = float(os.getenv('STRONG_SIGNAL_MULTIPLIER', '1.0'))
    moderate_signal_multiplier: float = float(os.getenv('MODERATE_SIGNAL_MULTIPLIER', '0.8'))
    trend_strength_threshold: float = float(os.getenv('TREND_STRENGTH_THRESHOLD', '0.005'))
    
    # Model Configuration
    price_prediction_multiplier: float = float(os.getenv('PRICE_PREDICTION_MULTIPLIER', '0.002'))
    
    # Trade Entry Threshold Configuration
    enable_strict_entry_conditions: bool = os.getenv('ENABLE_STRICT_ENTRY_CONDITIONS', 'True') == 'True'
    max_prediction_uncertainty: float = float(os.getenv('MAX_PREDICTION_UNCERTAINTY', '0.3'))
    min_ensemble_agreement_count: int = int(os.getenv('MIN_ENSEMBLE_AGREEMENT_COUNT', '2'))
    min_volume_ratio_threshold: float = float(os.getenv('MIN_VOLUME_RATIO_THRESHOLD', '0.8'))
    strong_signal_confidence_boost: float = float(os.getenv('STRONG_SIGNAL_CONFIDENCE_BOOST', '0.85'))
    
    def __post_init__(self):
        """Initialize symbols list and nested configuration objects."""
        if self.symbols is None:
            symbols_str = os.getenv('SYMBOLS', 'BTC-EUR,ETH-EUR,ADA-EUR,SOL-EUR,XRP-EUR')
            self.symbols = [s.strip() for s in symbols_str.split(',')]
        
        # Initialize nested configuration objects
        if self.model_settings is None:
            self.model_settings = ModelSettings()
        if self.trading_settings is None:
            self.trading_settings = TradingSettingsConfig()
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        required_fields = [
            self.bitvavo_api_key,
            self.bitvavo_api_secret,
            self.telegram_bot_token,
            self.telegram_chat_id
        ]
        
        if not all(required_fields):
            return False
            
        if self.trading_settings.initial_capital <= 0:
            return False
            
        if self.trading_settings.max_positions <= 0:
            return False

        if self.trading_settings.max_positions_per_symbol <= 0:
            return False
            
        if not (0 < self.trading_settings.base_position_size <= 1):
            return False
            
        if not self.symbols:
            return False
            
        return True
    
    def get_position_size(self, current_capital: float) -> float:
        """Calculate position size based on current capital."""
        return current_capital * self.trading_settings.base_position_size
