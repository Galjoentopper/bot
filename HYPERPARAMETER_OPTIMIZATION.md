# Hyperparameter Optimization for Crypto Trading Bot

This document describes the hyperparameter optimization capabilities implemented for the automated trading bot, following research-backed best practices for financial machine learning.

## Overview

The system implements comprehensive hyperparameter tuning for three model types:
- **LightGBM**: Gradient boosting for structured data analysis
- **GRU**: Recurrent neural network for time series prediction  
- **PPO**: Proximal Policy Optimization for reinforcement learning

The optimization uses Optuna with financial domain-specific parameter ranges and constraints.

## Usage

### Command Line Interface

Enable hyperparameter optimization with the `--tune-hyperparameters` flag:

```bash
# Basic hyperparameter optimization
python3 scripts/enhanced_trainer.py --config training_config.yaml \
    --models lightgbm --symbols BTCEUR \
    --tune-hyperparameters --optuna-trials 50

# Advanced configuration
python3 scripts/enhanced_trainer.py --config training_config.yaml \
    --models lightgbm gru --symbols BTCEUR ETHEUR \
    --tune-hyperparameters \
    --optuna-trials 100 \
    --optuna-timeout 3600 \
    --optimization-metric sharpe_ratio \
    --verbose
```

### Parameters

- `--tune-hyperparameters`: Enable hyperparameter optimization
- `--optuna-trials`: Number of optimization trials (default: 50)
- `--optuna-timeout`: Optimization timeout in seconds (default: 3600)
- `--optimization-metric`: Metric to optimize (choices: sharpe_ratio, sortino_ratio, calmar_ratio)

## Model-Specific Parameter Ranges

### LightGBM Parameters

Based on financial ML research for gradient boosting:

| Parameter | Range | Prior | Description |
|-----------|-------|-------|-------------|
| `n_estimators` | [100, 200, 500, 1000, 1500] | prefer_high | More trees with early stopping |
| `learning_rate` | [0.001, 0.1] (log) | prefer_low | Conservative learning rates |
| `num_leaves` | [20, 31, 50, 100, 150] | prefer_moderate | Balanced tree complexity |
| `max_depth` | [3, 5, 7, 10, 15] | prefer_moderate | Avoid overly deep trees |
| `feature_fraction` | [0.7, 1.0] | prefer_high | Conservative feature sampling |
| `bagging_fraction` | [0.7, 1.0] | prefer_high | Conservative data sampling |
| `min_data_in_leaf` | [10, 20, 50, 100] | prefer_high | Prevent overfitting |
| `reg_alpha` | [1e-6, 1e1] (log) | prefer_moderate | L1 regularization |
| `reg_lambda` | [1e-6, 1e1] (log) | prefer_moderate | L2 regularization |

### GRU Parameters

Optimized for financial time series prediction:

| Parameter | Range | Prior | Description |
|-----------|-------|-------|-------------|
| `learning_rate` | [1e-5, 5e-4] (log) | prefer_conservative | Stable learning rates |
| `hidden_size` | [32, 64, 96, 128, 192, 256] | prefer_moderate | Model capacity |
| `num_layers` | [1, 2, 3] | prefer_shallow | Simple models for finance |
| `dropout` | [0.2, 0.6] | prefer_high | Strong regularization |
| `batch_size` | [16, 32, 64, 128] | prefer_moderate | Balanced batch sizes |
| `sequence_length` | [15, 30, 45, 60] | prefer_moderate | Historical lookback |
| `weight_decay` | [1e-6, 5e-3] (log) | prefer_high | L2 regularization |

### PPO Parameters

Based on Stable-Baselines3 and financial RL research:

| Parameter | Range | Prior | Description |
|-----------|-------|-------|-------------|
| `learning_rate` | [1e-5, 1e-3] (log) | prefer_conservative | Stable policy updates |
| `n_steps` | [1024, 2048, 4096, 8192] | prefer_moderate | Rollout length |
| `batch_size` | [32, 64, 128, 256] | prefer_moderate | Balanced batches |
| `n_epochs` | [5, 10, 15, 20] | prefer_moderate | Training epochs |
| `gamma` | [0.95, 0.999] | prefer_high | Long-term focus |
| `gae_lambda` | [0.9, 0.99] | prefer_high | Bias-variance tradeoff |
| `clip_range` | [0.1, 0.3] | prefer_conservative | Conservative updates |
| `ent_coef` | [1e-6, 1e-2] (log) | prefer_low | Limited exploration |

## Asset Class Adjustments

The system automatically adjusts parameter ranges based on asset class:

### Crypto (Default)
- More volatile markets require stronger regularization
- Lower learning rates and higher dropout values
- Shorter sequence lengths to handle noise

### Forex
- Lower signal-to-noise ratio requires conservative parameters
- Very low learning rates and minimal complexity
- Strong regularization to prevent overfitting

### Stocks  
- More predictable patterns allow longer sequences
- Moderate regularization levels
- Balanced parameter ranges

## Market Regime Adjustments

Parameters are further adjusted based on market conditions:

### High Volatility
- Maximum regularization and conservative parameters
- Lower learning rates and higher dropout
- Reduced model complexity

### Low Volatility
- Slightly more aggressive parameters allowed
- Higher learning rates and larger models
- Enhanced signal extraction capabilities

## Optimization Process

1. **Parameter Range Loading**: System loads financial domain-specific ranges
2. **Bayesian Optimization**: Uses Gaussian Process for efficient search
3. **Financial Evaluation**: Optimizes for Sharpe ratio rather than RMSE
4. **Model Training**: Applies best parameters to actual training
5. **Results Storage**: Saves optimization results for analysis

## Results Storage

Optimization results are saved to `./models/optimization_results/`:

```json
{
  "success": true,
  "best_params": {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    ...
  },
  "best_value": 1.85,
  "optimization_metric": "sharpe_ratio",
  "n_trials": 50
}
```

## Feature Selection Integration

The system works with existing feature selection:
- **Method**: Mutual information (mutual_info)
- **Max Features**: 200 (configurable in training_config.yaml)
- **Feature Types**: All enabled (technical, statistical, fourier, wavelet, etc.)

## Performance Considerations

- **LightGBM**: ~15 seconds per trial
- **GRU**: ~2-3 minutes per trial  
- **PPO**: ~5-8 minutes per trial

Recommended trial counts:
- **Quick test**: 10-20 trials
- **Production**: 50-100 trials
- **Research**: 200+ trials

## Best Practices

1. **Start Small**: Use 10-20 trials for initial testing
2. **Increase Gradually**: Scale up based on computational resources
3. **Monitor Results**: Check optimization logs and saved results
4. **Asset-Specific**: Consider asset class and market regime
5. **Time Constraints**: Set appropriate timeouts for your environment

## Integration with Existing Workflow

The hyperparameter optimization seamlessly integrates with the existing training pipeline:

1. Data collection remains unchanged
2. Feature engineering uses existing configuration
3. Model training applies optimized parameters
4. Model packaging and deployment work normally

## Troubleshooting

### Common Issues

1. **Long Training Times**: Reduce trials or timeout
2. **Memory Issues**: Use smaller batch sizes or models
3. **Poor Optimization**: Increase trial count or check data quality
4. **PPO Data Format**: PPO requires specific DataFrame format

### Debugging

Enable verbose logging with `--verbose` to see detailed optimization progress.

## Example Workflows

### Quick Development Testing
```bash
python3 scripts/enhanced_trainer.py --config training_config.yaml \
    --models lightgbm --symbols BTCEUR \
    --tune-hyperparameters --optuna-trials 5 --optuna-timeout 300 \
    --verbose
```

### Production Optimization
```bash
python3 scripts/enhanced_trainer.py --config training_config.yaml \
    --models lightgbm gru --symbols BTCEUR ETHEUR ADAEUR \
    --tune-hyperparameters --optuna-trials 100 --optuna-timeout 7200 \
    --optimization-metric sharpe_ratio
```

### Research Mode
```bash
python3 scripts/enhanced_trainer.py --config training_config.yaml \
    --models lightgbm gru ppo --symbols BTCEUR ETHEUR ADAEUR DOTEUR LINKEUR \
    --tune-hyperparameters --optuna-trials 200 --optuna-timeout 14400 \
    --verbose
```

This hyperparameter optimization system follows financial ML best practices and research recommendations for achieving optimal risk-adjusted returns in cryptocurrency trading.