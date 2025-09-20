# Superior Ensemble Training System - Paperspace MLOps

This directory contains the revolutionary **Superior Ensemble Training System** specifically designed for Paperspace Gradient execution. The system unifies PPO, GRU, and LightGBM model training under a single, optimized architecture.

## 🏗️ Architecture Overview

### Core Components

1. **`superior_ensemble_trainer.py`** - Main training orchestrator
   - Unified training pipeline for all model types
   - Intelligent feature routing (103 features for PPO, 100 for others)
   - Optuna hyperparameter optimization
   - Parallel training coordination
   - S3 export integration

2. **`paperspace_superior_training.py`** - Paperspace execution runner
   - Resource monitoring and optimization
   - GPU utilization management
   - Automated S3 export workflow
   - Comprehensive error handling

3. **`superior_ppo_feature_expander.py`** - Advanced PPO feature engineering
   - 103 sophisticated trading features
   - Market microstructure indicators
   - Advanced momentum and volatility metrics

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Run on Paperspace Gradient
cd /opt/trading_bot/bot
./paperspace_mlops/setup_paperspace.sh
```

### 2. Training Execution

**Full Training (All Models, All Symbols)**
```bash
python paperspace_mlops/paperspace_superior_training.py
```

**Specific Symbols and Models**
```bash
python paperspace_mlops/paperspace_superior_training.py \
  --symbols BTCEUR,ETHEUR,ADAEUR \
  --models ppo,gru
```

**Quick Test**
```bash
python paperspace_mlops/paperspace_superior_training.py \
  --quick-test \
  --symbols BTCEUR \
  --models ppo
```

**With Resource Monitoring**
```bash
python paperspace_mlops/paperspace_superior_training.py \
  --monitor
```

## 🎯 Training Features

### Superior Architecture Benefits

1. **Unified Pipeline** - Single codebase for all model types
2. **Intelligent Feature Routing** - Optimized feature sets per model
3. **Resource Optimization** - Paperspace GPU/memory management
4. **Automated Export** - Direct S3 deployment pipeline
5. **Comprehensive Validation** - Walk-forward validation with embargo
6. **Hyperparameter Optimization** - Optuna-based parameter tuning

### Model Specifications

#### PPO (Reinforcement Learning)
- **Features**: 103 advanced trading indicators
- **Architecture**: Actor-Critic with attention mechanisms
- **Training**: Proximal Policy Optimization
- **Output**: Action probabilities for buy/sell/hold decisions

#### GRU (Neural Network)
- **Features**: 100 core technical indicators
- **Architecture**: Gated Recurrent Units with dropout
- **Training**: Time series prediction with walk-forward validation
- **Output**: Price movement predictions

#### LightGBM (Gradient Boosting)
- **Features**: 100 engineered features
- **Architecture**: Gradient boosting decision trees
- **Training**: Classification with early stopping
- **Output**: Buy/sell probability predictions

## 📊 Feature Engineering

### PPO Feature Set (103 features)
- **Price Action**: OHLCV derivatives, price patterns
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic
- **Market Microstructure**: Bid-ask spreads, volume profiles
- **Volatility**: ATR, Parkinson, Garman-Klass estimators
- **Momentum**: Rate of change, momentum oscillators
- **Pattern Recognition**: Candlestick patterns, support/resistance

### GRU/LightGBM Feature Set (100 features)
- **Core Technical Indicators**: Essential trading signals
- **Price Transformations**: Returns, log returns, z-scores
- **Volume Analysis**: Volume-weighted indicators
- **Trend Analysis**: Moving averages, trend strength
- **Mean Reversion**: Bollinger Band positions, z-scores

## 🔧 Configuration

### Training Configuration (`config/training_config.yaml`)
```yaml
training:
  models: ['ppo', 'gru', 'lightgbm']
  optuna_trials: 100
  parallel_training: true

data:
  symbols: ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']
  interval: '30m'
  lookback_days: 365
```

### Export Configuration (`paperspace_mlops/export_config.yaml`)
```yaml
s3_export:
  bucket_name: "${AWS_MODELS_BUCKET}"
  validation:
    min_sharpe_ratio: 0.5
    min_win_rate: 0.45
```

## 🔄 Workflow

### 1. Data Acquisition
- Fetch historical market data via CCXT
- Apply data quality validation
- Generate feature sets (103 for PPO, 100 for others)

### 2. Model Training
- **PPO**: Reinforcement learning with environment simulation
- **GRU**: Time series neural network training
- **LightGBM**: Gradient boosting with cross-validation

### 3. Validation
- Walk-forward validation with embargo periods
- Performance metric calculation
- Feature importance analysis

### 4. Export to S3
- Model serialization and packaging
- Metadata and feature schema export
- S3 upload with validation
- Production deployment notification

## 📈 Performance Monitoring

### Training Metrics
- **PPO**: Episode rewards, policy gradient convergence
- **GRU**: Training/validation loss curves
- **LightGBM**: Boosting iterations, feature importance

### Validation Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Profit Factor**: Gross profit / gross loss ratio

## 🛠️ Troubleshooting

### Common Issues

1. **GPU Memory Error**
   ```bash
   # Reduce batch sizes in config
   python paperspace_superior_training.py --quick-test
   ```

2. **S3 Access Error**
   ```bash
   # Check AWS credentials
   aws s3 ls s3://$AWS_MODELS_BUCKET/
   ```

3. **Training Timeout**
   ```bash
   # Use specific models/symbols
   python paperspace_superior_training.py --symbols BTCEUR --models ppo
   ```

### Debug Mode
```bash
# Enable detailed logging
export PYTHONUNBUFFERED=1
python paperspace_superior_training.py --monitor
```

## 📁 Directory Structure

```
paperspace_mlops/
├── superior_ensemble_trainer.py    # Main training orchestrator
├── paperspace_superior_training.py # Paperspace execution runner
├── superior_ppo_feature_expander.py # PPO feature engineering
├── export_config.yaml              # S3 export configuration
├── requirements.txt                 # Python dependencies
├── setup_paperspace.sh            # Environment setup script
└── PAPERSPACE_README.md           # This documentation
```

## 🎯 Production Deployment

After training completion:

1. **Models are automatically exported to S3**
2. **Production servers import models via S3**
3. **Feature schemas ensure compatibility**
4. **Validation metrics guide deployment decisions**

The superior ensemble training system provides a unified, optimized pipeline for developing high-performance trading models on Paperspace Gradient infrastructure.
