# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is an advanced cryptocurrency trading bot system with machine learning capabilities, designed for automated paper trading with multiple model types including neural networks (GRU), gradient boosting (LightGBM), and reinforcement learning (PPO). The system features comprehensive risk management, Telegram notifications, and enterprise-grade monitoring capabilities.

## Key Commands

### Development and Testing
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest
pytest -v  # verbose output
pytest tests/test_specific_module.py  # run specific test file

# Code quality
black .  # format code
flake8 .  # lint code
```

### Model Training
```bash
# Train all models using unified configuration
python scripts/enhanced_trainer.py --config training_config.yaml

# Train specific model type
python scripts/enhanced_trainer.py --config training_config.yaml --model-type gru
python scripts/enhanced_trainer.py --config training_config.yaml --model-type lightgbm
python scripts/enhanced_trainer.py --config training_config.yaml --model-type ppo
```

### Trading System
```bash
# Start paper trading (main trader script)
python scripts/trader.py --config training_config.yaml --models-dir ./models

# Run specific number of iterations for testing
python scripts/trader.py --config training_config.yaml --iterations 5

# Start comprehensive test system
python comprehensive_test_system.py

# Quick system validation
python quick_test_system.py
```

### Telegram Bot
```bash
# Start Telegram bot listener
python telegram_bot_listener.py

# Test Telegram integration
python test_telegram_commands.py

# Debug Telegram connectivity
bash debug_telegram.sh
```

### System Management (Production)
```bash
# System service management
sudo systemctl start trading-bot
sudo systemctl stop trading-bot
sudo systemctl status trading-bot

# Tmux session management
./scripts/tmux_manager.sh start
./scripts/tmux_manager.sh status
./scripts/tmux_manager.sh attach
./scripts/tmux_manager.sh logs

# Health monitoring
./scripts/health_check.sh
./scripts/start_monitoring.sh
```

### Deployment
```bash
# Deploy full system to production
./deploy_full_system.sh

# System validation after deployment
./validate_fixes.py
./verify_clean_structure.sh
```

## Architecture Overview

### Core Components

**Data Pipeline (`src/data_pipeline/`)**
- `features.py`: Advanced feature engineering with 200+ technical indicators
- `preprocess.py`: Data preprocessing and normalization
- Supports multiple data sources: Binance, YFinance with failover

**Model Ensemble (`src/models/`)**
- `gru_trainer.py`: Recurrent neural network for time series prediction
- `lgbm_trainer.py`: Gradient boosting model for structured features  
- `ppo_trainer.py`: Reinforcement learning agent for dynamic trading
- Models are trained per-symbol with walk-forward validation

**Trading Engine (`scripts/trader.py`)**
- Unified paper trading system combining all model predictions
- Dynamic threshold adjustment based on market volatility
- Risk management with position sizing and stop-losses
- Real-time data fetching from multiple exchanges

**Configuration System (`src/config/`)**
- Hierarchical configuration with `training_config.yaml` as master config
- Environment-specific overrides for development/production
- Auto-detection of config files and validation

**Notification System (`src/notifier/`)**
- `enhanced_telegram.py`: Rich Telegram integration with trading alerts
- `telegram_notifier.py`: Core notification infrastructure
- Real-time trade execution and performance updates

**Risk Management & Analytics (`src/trading/`)**
- `profit_optimizer.py`: Advanced profit optimization strategies
- `enhanced_signal_generator.py`: Multi-model signal aggregation
- `performance_analytics.py`: Real-time performance monitoring
- Dynamic position sizing using Kelly criterion

### Key Architecture Patterns

**Per-Symbol Model Architecture**: Each cryptocurrency symbol has dedicated trained models (GRU, LightGBM, PPO) stored in `models/{model_type}/{SYMBOL}/` directories.

**Feature Consistency**: The `ModelMetadata` class ensures feature alignment between training and inference by persisting feature names and order in `models/metadata/features_{SYMBOL}.json`.

**Ensemble Prediction**: The system combines predictions from multiple models using configurable weights in `training_config.yaml` (`model_weights` section).

**Dynamic Thresholding**: Trading thresholds adapt to market conditions using volatility-based scaling defined in the `thresholds` configuration section.

## Configuration

### Master Configuration File: `training_config.yaml`

This single file controls the entire system pipeline:

**Data Configuration**
- `symbols`: List of trading pairs (e.g., ['BTCEUR', 'ETHEUR'])
- `interval`: Candle timeframe ('30m' recommended for optimal performance)
- `lookback_days`: Historical data range for training

**Model Parameters**
- Individual parameter sections for each model type
- `model_weights`: Ensemble weighting (lightgbm: 0.55, gru: 0.35, ppo: 0.1)
- Hyperparameter optimization settings via Optuna

**Trading Configuration**
- `thresholds.per_symbol`: Symbol-specific buy/sell thresholds
- `profit_optimization`: Risk management parameters
- `adaptive_weights`: Enable dynamic model weight adjustment

**Environment Variables** (`.env` file required)
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
BITVAVO_API_KEY=your_api_key (if using Bitvavo)
BITVAVO_API_SECRET=your_api_secret
```

## Testing Strategy

### Test Hierarchy
1. `quick_test_system.py`: Fast validation of core functionality
2. `comprehensive_test_system.py`: Full system integration tests
3. `final_test_system.py`: Production readiness validation

### Model Testing
- Each model trainer includes validation methods for prediction consistency
- Walk-forward validation ensures temporal robustness
- Feature alignment tests prevent train/test data leakage

## Development Workflow

### Adding New Features
1. Implement in appropriate `src/` module following existing patterns
2. Update `training_config.yaml` if new parameters are needed
3. Add tests in corresponding `test_*.py` file
4. Run full test suite before committing

### Model Development
- New models should inherit from base trainer classes
- Implement `train()`, `predict()`, `save_model()`, `load_model()` methods
- Add model metadata persistence for feature consistency
- Update ensemble weights in configuration

### Debugging Common Issues

**Feature Mismatch Errors**: Check `models/metadata/features_{SYMBOL}.json` files match between training and inference.

**PPO Prediction Issues**: Ensure observation shape matches training environment (sequence_length, num_features).

**Telegram Connectivity**: Run `debug_telegram.sh` to validate bot token and permissions.

**Data Freshness**: System validates data is within 2 hours of current time; older data triggers warnings.

## Production Deployment

The system includes complete production deployment infrastructure:

**Systemd Service**: `trading-bot.service` for automatic startup and monitoring
**Tmux Management**: Isolated sessions for trading and Telegram components  
**Log Rotation**: Automated log management with size limits and retention
**Health Monitoring**: Continuous system health checks and alerts
**Backup Systems**: Automated configuration and model backups

## Important Notes

- **No Live Trading**: This system is designed for paper trading only. All trades are simulated.
- **Model Persistence**: Trained models are saved in `models/` directory with metadata for reproducible deployment.
- **Resource Requirements**: GPU recommended for neural network training; system supports CPU fallback.
- **Security**: Never commit API keys or tokens to the repository. Use `.env` files for sensitive configuration.