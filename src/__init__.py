"""
Bot Kilo - Cryptocurrency Trading Bot
====================================

A sophisticated cryptocurrency trading bot with machine learning and reinforcement learning capabilities.

Modules:
- data_pipeline: Data collection, preprocessing, and feature engineering
- models: Machine learning models (GRU, LightGBM, PPO)
- rl_env: Reinforcement learning trading environment
- backtesting: Backtesting framework
- notifier: Notification system (Telegram)
- utils: Utility functions and logging
- config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "Bot Kilo Team"
__email__ = "contact@botkilo.com"

# Import main components for easy access
try:
    # Core components
    # Backtesting
    from .backtesting.backtest import Backtester

    # Configuration
    from .config.config_loader import ConfigLoader
    from .core.config_manager import ConfigurationManager
    from .core.container import DIContainer
    from .core.enhanced_logger import EnhancedLogger
    from .data_pipeline.features import FeatureEngine

    # Data pipeline
    from .data_pipeline.loader import DataLoader
    from .data_pipeline.preprocess import DataPreprocessor

    # Models
    from .models.gru_trainer import GRUTrainer
    from .models.lgbm_trainer import LightGBMTrainer
    from .models.model_manager import ModelManager
    from .models.ppo_trainer import PPOTrainer
    from .notifier.enhanced_telegram import EnhancedTelegramNotifier

    # Notifications
    from .notifier.telegram import TelegramNotifier

    # RL Environment
    from .rl_env.trading_env import TradingEnvironment
    from .trading.position_tracker import PositionTracker

    # Trading components
    from .trading.trading_metrics import TradingMetrics

    # Utilities
    from .utils.logger import TradingBotLogger, setup_logging

except ImportError as e:
    # Handle import errors gracefully during development
    import warnings

    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)

__all__ = [
    # Core
    "ConfigurationManager",
    "EnhancedLogger",
    "DIContainer",
    # Data pipeline
    "DataLoader",
    "FeatureEngine",
    "DataPreprocessor",
    # Models
    "GRUTrainer",
    "LightGBMTrainer",
    "PPOTrainer",
    "ModelManager",
    # RL Environment
    "TradingEnvironment",
    # Backtesting
    "Backtester",
    # Trading
    "TradingMetrics",
    "PositionTracker",
    # Notifications
    "TelegramNotifier",
    "EnhancedTelegramNotifier",
    # Utilities
    "setup_logging",
    "TradingBotLogger",
    # Configuration
    "ConfigLoader",
]
