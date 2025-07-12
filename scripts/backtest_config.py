from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BacktestConfig:
    """Configuration class for backtesting parameters"""
    
    # Portfolio Management
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 10
    max_trades_per_hour: int = 3
    max_capital_per_trade: float = 0.1  # Max 10% of capital per trade
    
    # Trading Costs
    trading_fee: float = 0.002  # 0.2% per trade
    slippage: float = 0.001  # 0.1% slippage
    
    # Risk Management
    stop_loss_pct: float = 0.03  # 3% stop loss
    atr_multiplier: float = 2.0  # ATR multiplier for stop loss
    
    # Signal Thresholds
    buy_threshold: float = 0.6  # Lowered from 0.7 for more trades
    sell_threshold: float = 0.4  # Raised from 0.3 for more trades
    lstm_delta_threshold: float = 0.02  # Realistic for % changes (was 0.5)
    
    # Alternative thresholds for sensitivity analysis
    conservative_buy_threshold: float = 0.8
    conservative_sell_threshold: float = 0.2
    aggressive_buy_threshold: float = 0.6
    aggressive_sell_threshold: float = 0.4
    
    # Walk-Forward Parameters
    train_months: int = 4
    test_months: int = 1
    slide_months: int = 1
    
    # Data Parameters
    sequence_length: int = 60
    price_change_threshold: float = 0.002
    
    # Feature Engineering
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    sma_period: int = 200
    ema_short: int = 9
    ema_long: int = 21
    
    # Backtesting Options
    enable_stop_loss: bool = True
    enable_take_profit: bool = False
    take_profit_pct: float = 0.06  # 6% take profit
    enable_trailing_stop: bool = False
    trailing_stop_pct: float = 0.02  # 2% trailing stop
    
    # Analysis Options
    bootstrap_runs: int = 10
    noise_level: float = 0.001  # 0.1% noise for bootstrap
    confidence_intervals: List[float] = None
    
    def __post_init__(self):
        if self.confidence_intervals is None:
            self.confidence_intervals = [0.5, 0.6, 0.7, 0.8]
    
    @classmethod
    def conservative_config(cls):
        """Conservative trading configuration"""
        return cls(
            risk_per_trade=0.01,  # 1% risk
            max_positions=5,
            buy_threshold=0.8,
            sell_threshold=0.2,
            lstm_delta_threshold=0.6,
            stop_loss_pct=0.02,  # 2% stop loss
            max_trades_per_hour=2
        )
    
    @classmethod
    def aggressive_config(cls):
        """Aggressive trading configuration"""
        return cls(
            risk_per_trade=0.03,  # 3% risk
            max_positions=15,
            buy_threshold=0.6,
            sell_threshold=0.4,
            lstm_delta_threshold=0.3,
            stop_loss_pct=0.05,  # 5% stop loss
            max_trades_per_hour=5
        )
    
    @classmethod
    def high_frequency_config(cls):
        """High frequency trading configuration"""
        return cls(
            risk_per_trade=0.005,  # 0.5% risk per trade
            max_positions=20,
            buy_threshold=0.55,
            sell_threshold=0.45,
            lstm_delta_threshold=0.1,
            stop_loss_pct=0.01,  # 1% stop loss
            max_trades_per_hour=10,
            trading_fee=0.001,  # Lower fees for HF
            slippage=0.0005
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving"""
        return {
            'initial_capital': self.initial_capital,
            'risk_per_trade': self.risk_per_trade,
            'max_positions': self.max_positions,
            'max_trades_per_hour': self.max_trades_per_hour,
            'trading_fee': self.trading_fee,
            'slippage': self.slippage,
            'stop_loss_pct': self.stop_loss_pct,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'lstm_delta_threshold': self.lstm_delta_threshold,
            'train_months': self.train_months,
            'test_months': self.test_months,
            'slide_months': self.slide_months,
            'sequence_length': self.sequence_length,
            'price_change_threshold': self.price_change_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        return cls(**config_dict)

# Predefined configurations for different trading styles
CONFIG_PRESETS = {
    'conservative': BacktestConfig.conservative_config(),
    'balanced': BacktestConfig(),
    'aggressive': BacktestConfig.aggressive_config(),
    'high_frequency': BacktestConfig.high_frequency_config()
}

# Symbol-specific configurations (if needed)
SYMBOL_CONFIGS = {
    'BTCEUR': {
        'buy_threshold': 0.75,  # BTC showed best performance, be more selective
        'sell_threshold': 0.25,
        'risk_per_trade': 0.025  # Slightly higher risk for best performer
    },
    'ETHEUR': {
        'buy_threshold': 0.7,
        'sell_threshold': 0.3,
        'risk_per_trade': 0.02
    },
    'ADAEUR': {
        'buy_threshold': 0.65,  # More aggressive for smaller cap
        'sell_threshold': 0.35,
        'risk_per_trade': 0.015
    },
    'SOLEUR': {
        'buy_threshold': 0.65,
        'sell_threshold': 0.35,
        'risk_per_trade': 0.015
    }
}

def get_symbol_config(symbol: str, base_config: BacktestConfig = None) -> BacktestConfig:
    """Get symbol-specific configuration"""
    if base_config is None:
        base_config = BacktestConfig()
    
    if symbol in SYMBOL_CONFIGS:
        symbol_overrides = SYMBOL_CONFIGS[symbol]
        # Create a copy and update with symbol-specific values
        config_dict = base_config.to_dict()
        config_dict.update(symbol_overrides)
        return BacktestConfig.from_dict(config_dict)
    
    return base_config