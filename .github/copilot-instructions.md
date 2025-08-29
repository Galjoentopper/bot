# Streamlined Automated Trading System

**ALWAYS follow these instructions first and only search for additional context if the information below is incomplete or incorrect.**

This is a minimal yet comprehensive automated trading system with cross-platform compatibility between Linux training and Windows deployment environments. The system uses machine learning (GRU, LightGBM) and reinforcement learning (PPO) models for cryptocurrency trading.

## Working Effectively

### Bootstrap and Dependencies
Run these commands to set up the development environment:

```bash
# Install Python package and dependencies - NEVER CANCEL: Takes 3 minutes
cd /home/runner/work/bot/bot
pip3 install -e . --timeout 300
```

**TIMEOUT REQUIREMENT**: Always set timeout to 300+ seconds (5+ minutes) for package installation.

### Core Workflow Commands

#### 1. Data Collection
```bash
# Fetch training data for all symbols - Takes 4 seconds
chmod +x fetch_training_data.sh
./fetch_training_data.sh

# Fetch data for specific symbol only
./fetch_training_data.sh --symbol BTCEUR
```

#### 2. Model Training
```bash
# Direct training command (RECOMMENDED) - NEVER CANCEL: Takes 10+ minutes for full training
python3 scripts/enhanced_trainer.py --config training_config.yaml --models gru lightgbm ppo --symbols BTCEUR --verbose --n-splits 2

# Training single model - Takes 15 seconds
python3 scripts/enhanced_trainer.py --config training_config.yaml --models lightgbm --symbols BTCEUR --verbose --n-splits 2

# Training multiple models - NEVER CANCEL: Takes 9.5 minutes
python3 scripts/enhanced_trainer.py --config training_config.yaml --models gru lightgbm ppo --symbols BTCEUR --verbose --n-splits 2
```

**CRITICAL TIMING**: 
- Single model training: 15 seconds - set timeout to 60+ seconds
- Multiple model training: 9.5 minutes - set timeout to 15+ minutes  
- Full symbol set training: 30+ minutes - set timeout to 45+ minutes
- **NEVER CANCEL** these training operations

#### 3. Trading Bot Execution
```bash
# Run trading bot (will be blocked by Binance geolocation restrictions in sandboxed environments)
python3 scripts/enhanced_trader.py
```

Note: Live trading requires API access and will be blocked in restricted environments.

## Validation and Testing

### Manual Validation Steps
After making changes, ALWAYS run these validation steps:

```bash
# 1. Validate package installation works
pip3 install -e . --timeout 300

# 2. Validate data collection works
./fetch_training_data.sh --symbol BTCEUR

# 3. Validate training works with minimal scope
python3 scripts/enhanced_trainer.py --config training_config.yaml --models lightgbm --symbols BTCEUR --verbose --n-splits 2

# 4. Validate model loading works
python3 -c "
from src.backtesting.backtest import Backtester
bt = Backtester(initial_capital=10000, transaction_fee=0.001, slippage=0.0005)
print('Backtesting validation: PASS')
"
```

### End-to-End Validation Scenarios
Test these complete workflows after making changes:

1. **Training Pipeline**: Data fetch → Model training → Model validation
2. **Model Loading**: Train models → Load in trader → Validate predictions
3. **Configuration Changes**: Modify training_config.yaml → Retrain → Verify outputs

## Known Issues and Workarounds

### train_models_linux.sh Limitation
- The shell wrapper script `train_models_linux.sh` fails due to missing `startup_init` module
- **WORKAROUND**: Use `python3 scripts/enhanced_trainer.py` directly instead
- This is the recommended approach for all training operations

### API Restrictions  
- Binance API access blocked in sandboxed environments due to geolocation restrictions
- Models train successfully using existing cached data in `data/` directory
- Trading bot loads models correctly but cannot fetch live market data

### Missing Development Tools
- No linting tools (black, flake8) available in current environment
- No formal test suite exists
- Use manual validation commands above for testing changes

## Configuration Management

### Key Configuration Files
- `training_config.yaml`: Centralized configuration for all system behavior
- `requirements.txt`: Python dependencies 
- `setup.py`: Package installation configuration

### Important Directories
- `src/`: Core Python modules and trading logic
- `scripts/`: Training and trading entry points  
- `models/`: Trained models (auto-created)
- `data/`: Market data databases (auto-created)
- `logs/`: System logs (auto-created)

## Cross-Platform Notes

This system is designed for:
- **Linux environment**: Data collection and model training
- **Windows environment**: Model deployment and live trading

The current setup is optimized for Linux training environments.

## Common Tasks Reference

### Repository Structure
```
/home/runner/work/bot/bot/
├── fetch_training_data.sh      # Data acquisition (4 seconds)
├── train_models_linux.sh       # Training wrapper (broken - use enhanced_trainer.py)
├── scripts/enhanced_trainer.py # Direct training (9.5 min for 3 models)
├── scripts/enhanced_trader.py  # Trading bot execution
├── training_config.yaml        # Centralized configuration
├── requirements.txt            # Dependencies
├── setup.py                   # Package installation
├── src/                       # Core modules
├── data/                      # Market data (auto-created)
└── models/                    # Trained models (auto-created)
```

### Model Types
- **GRU**: Recurrent neural network for time series prediction
- **LightGBM**: Gradient boosting for structured data analysis  
- **PPO**: Proximal Policy Optimization for reinforcement learning

### Symbols Supported
- BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR (configurable in training_config.yaml)

## Critical Reminders

1. **NEVER CANCEL** long-running training operations - they may take 10+ minutes
2. **ALWAYS** set appropriate timeouts (15+ minutes for full training)
3. **USE** `enhanced_trainer.py` directly instead of shell wrapper scripts
4. **VALIDATE** changes with the manual validation steps above
5. **CHECK** that models are created in `models/` directory after training
6. **REMEMBER** API restrictions in sandboxed environments are normal and expected