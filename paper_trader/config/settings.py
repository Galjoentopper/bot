"""Trading settings and configuration management."""

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingSettings:
    """Configuration settings for the paper trader."""
    
    # API Configuration
    bitvavo_api_key: str = os.getenv('BITVAVO_API_KEY', '')
    bitvavo_api_secret: str = os.getenv('BITVAVO_API_SECRET', '')
    
    # Telegram Configuration
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Trading Parameters
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '10000.0'))
    max_positions: int = int(os.getenv('MAX_POSITIONS', '10'))
    position_size_pct: float = float(os.getenv('POSITION_SIZE_PCT', '0.10'))
    take_profit_pct: float = float(os.getenv('TAKE_PROFIT_PCT', '0.01'))
    stop_loss_pct: float = float(os.getenv('STOP_LOSS_PCT', '0.01'))
    trailing_stop_pct: float = float(os.getenv('TRAILING_STOP_PCT', '0.005'))
    max_hold_hours: int = int(os.getenv('MAX_HOLD_HOURS', '2'))
    
    # Symbols
    symbols: List[str] = None
    
    # Model Configuration
    model_path: str = os.getenv('MODEL_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'))
    sequence_length: int = int(os.getenv('SEQUENCE_LENGTH', '60'))
    
    # Window-based Model Configuration
    min_window: int = int(os.getenv('MIN_WINDOW', '3'))
    max_window: int = int(os.getenv('MAX_WINDOW', '41'))
    default_window: int = int(os.getenv('DEFAULT_WINDOW', '15'))
    
    # Prediction Confidence Thresholds
    min_confidence_threshold: float = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
    min_signal_strength: str = os.getenv('MIN_SIGNAL_STRENGTH', 'MODERATE')
    
    # Ensemble Model Weights
    lstm_weight: float = float(os.getenv('LSTM_WEIGHT', '0.6'))
    xgb_weight: float = float(os.getenv('XGB_WEIGHT', '0.4'))
    
    # Risk Management
    max_daily_loss_pct: float = float(os.getenv('MAX_DAILY_LOSS_PCT', '0.05'))
    max_drawdown_pct: float = float(os.getenv('MAX_DRAWDOWN_PCT', '0.10'))
    
    def __post_init__(self):
        """Initialize symbols list from environment variable."""
        if self.symbols is None:
            symbols_str = os.getenv('SYMBOLS', 'BTCEUR,ETHEUR,ADAEUR,SOLEUR,XRPEUR')
            self.symbols = [s.strip() for s in symbols_str.split(',')]
    
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
            
        if self.initial_capital <= 0:
            return False
            
        if self.max_positions <= 0:
            return False
            
        if not (0 < self.position_size_pct <= 1):
            return False
            
        if not self.symbols:
            return False
            
        return True
    
    def get_position_size(self, current_capital: float) -> float:
        """Calculate position size based on current capital."""
        return current_capital * self.position_size_pct