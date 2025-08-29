# Streamlined Automated Trading System

A minimal yet comprehensive automated trading system with cross-platform compatibility between Linux training and Windows deployment environments.

## 🚀 Quick Start

### Linux Training Environment

```bash
# 1. Fetch training data
chmod +x fetch_training_data.sh
./fetch_training_data.sh

# 2. Train models and create deployment package
chmod +x train_models_linux.sh  
./train_models_linux.sh
```

### Windows Trading Environment

```batch
# 3. Import models (extract deployment package)
import_models.bat

# 4. Start live trading
deploy_trading.bat
```

## 📋 System Components

The system consists of exactly **5 core components**:

### 1. **fetch_training_data.sh** (Linux)
- Automatically downloads and preprocesses datasets
- Reads symbols from centralized `training_config.yaml`
- Supports Binance and YFinance data sources
- Progress indicators and error handling

### 2. **train_models_linux.sh** (Linux) 
- Trains GRU, LightGBM, and PPO models
- Reads parameters from centralized configuration
- Automated hyperparameter optimization
- Creates deployment-ready ZIP archive

### 3. **Automated ZIP Archive Generation**
- Integrated into training pipeline
- Contains all trained models and deployment files
- Cross-platform compatible format
- Includes metadata and configuration

### 4. **import_models.bat** (Windows)
- Extracts and configures imported models
- Robust error handling and validation
- Automatic directory structure creation
- Progress indicators and dependency checking

### 5. **deploy_trading.bat** (Windows)
- Initializes and starts live trading system
- Automatic model validation and loading
- Real-time monitoring and logging
- Telegram notifications support

## ⚙️ Configuration

All system behavior is controlled by the centralized `training_config.yaml` file:

```yaml
# Data acquisition settings
data_acquisition:
  symbols: ['BTCEUR', 'ETHEUR', 'ADAEUR']
  interval: '30m'
  lookback_days: 365

# Training parameters  
training:
  models: ['gru', 'lightgbm', 'ppo']
  epochs: 100
  optuna_trials: 100

# Trading settings
trading:
  initial_balance: 10000
  max_position_size: 0.1
  model_weights:
    gru: 0.45
    lightgbm: 0.45
    ppo: 0.1
```

## 📁 Directory Structure

```
trading-system/
├── fetch_training_data.sh      # Component 1: Data acquisition
├── train_models_linux.sh       # Component 2: Model training
├── import_models.bat           # Component 4: Model import
├── deploy_trading.bat          # Component 5: Trading deployment
├── training_config.yaml        # Centralized configuration
├── requirements.txt            # Python dependencies
├── src/                        # Core trading logic
├── scripts/                    # Training and trading scripts
├── models/                     # Trained models (auto-created)
├── data/                       # Market data (auto-created)
└── logs/                       # System logs (auto-created)
```

## 🔄 Cross-Platform Workflow

1. **Linux Training Computer**: Run data fetching and model training
2. **File Transfer**: Copy generated ZIP archive to Windows computer  
3. **Windows Trading Computer**: Import models and deploy trading system

## 📊 Model Types

- **GRU**: Recurrent neural networks for sequence prediction
- **LightGBM**: Gradient boosting for structured data  
- **PPO**: Reinforcement learning for trading decisions

## 🔧 Requirements

### Linux Training Environment
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended

### Windows Trading Environment  
- Python 3.8+
- 4GB+ RAM
- Stable internet connection

## 🚨 Error Handling

- Automatic dependency checking
- Robust retry mechanisms  
- Comprehensive logging
- Progress indicators
- Graceful failure recovery

## 📞 Support

- Check `logs/` directory for detailed error information
- All scripts include built-in help: `script_name --help`
- Telegram notifications for trading events and errors

---

**Note**: This streamlined system eliminates redundant files and focuses on the 5 core components for maximum efficiency and user-friendliness.