# Optimized Variables - Scientific Parameter Optimization

This document provides comprehensive usage instructions for the `optimized_variables.py` script, which uses advanced scientific optimization methods to find the best parameter combinations for the paper trading bot.

## Overview

The `optimized_variables.py` script replaces the previous optimization system (`run_backtest.py`, `optimization_config.py`, `parameter_optimizer.py`) with a single, more powerful tool that:

- Uses **Bayesian optimization** for efficient parameter search
- Supports **single or multiple symbols** via command line
- Implements **scientific optimization metrics** (Sharpe ratio, Calmar ratio, Total return, Profit factor)
- Automatically generates **optimized .env files** with symbol and date naming
- Integrates seamlessly with existing model and data infrastructure

## Key Features

### 🔬 Scientific Optimization Methods
- **Bayesian Optimization**: Uses Gaussian Process regression for efficient parameter exploration
- **Grid Search**: Comprehensive parameter space exploration (for comparison)
- **Research-Based Parameter Ranges**: Based on academic finance literature
- **Statistical Rigor**: Minimum trade requirements and significance testing

### 📊 Multiple Optimization Modes
- **Conservative**: Lower risk, higher confidence thresholds
- **Balanced**: Moderate risk/reward balance
- **Aggressive**: Higher risk, more trading opportunities
- **Profit Focused**: Optimized specifically for maximum profitability

### 🎯 Optimization Objectives
- **Sharpe Ratio**: Risk-adjusted returns
- **Total Return**: Absolute profit maximization
- **Calmar Ratio**: Return/drawdown ratio (preferred for risk management)
- **Profit Factor**: Gross profit/gross loss ratio

## Quick Start

### Basic Usage
```bash
# Optimize for Bitcoin
python optimized_variables.py --symbols BTCEUR

# Optimize for multiple symbols
python optimized_variables.py --symbols BTCEUR ETHEUR

# Optimize for all available symbols
python optimized_variables.py --symbols BTCEUR ETHEUR ADAEUR SOLEUR XRPEUR
```

### Advanced Usage
```bash
# Aggressive profit-focused optimization
python optimized_variables.py --symbols BTCEUR ETHEUR \
    --mode aggressive \
    --objective total_return \
    --iterations 100

# Conservative optimization with Calmar ratio
python optimized_variables.py --symbols BTCEUR \
    --mode conservative \
    --objective calmar_ratio \
    --iterations 50

# Quick grid search for comparison
python optimized_variables.py --symbols ETHEUR \
    --method grid \
    --grid-points 4
```

## Command Line Options

### Required Arguments
- `--symbols`: Cryptocurrency symbols to optimize for
  - Choices: `BTCEUR`, `ETHEUR`, `ADAEUR`, `SOLEUR`, `XRPEUR`
  - Can specify one or multiple symbols

### Optional Arguments
- `--method`: Optimization method (default: `bayesian`)
  - `bayesian`: Advanced Bayesian optimization (recommended)
  - `grid`: Comprehensive grid search

- `--mode`: Parameter space mode (default: `profit_focused`)
  - `conservative`: Lower risk, tighter parameters
  - `balanced`: Moderate risk/reward balance
  - `aggressive`: Higher risk, more opportunities
  - `profit_focused`: Optimized for maximum profit

- `--objective`: Optimization objective (default: `profit_factor`)
  - `sharpe_ratio`: Risk-adjusted returns
  - `total_return`: Absolute profit maximization
  - `calmar_ratio`: Return/maximum drawdown ratio
  - `profit_factor`: Gross profit/gross loss ratio

- `--iterations`: Number of optimization iterations (default: `50`)
  - For Bayesian optimization: total evaluations
  - Recommended: 50-200 for thorough optimization

- `--initial-random`: Initial random evaluations for Bayesian (default: `10`)
  - Number of random parameter combinations to try first
  - Helps initialize the Gaussian Process model

- `--grid-points`: Grid points per dimension for grid search (default: `3`)
  - Higher values = more thorough but slower search
  - Total combinations = grid-points ^ number_of_parameters

- `--quiet`: Suppress detailed output
  - Use for automated runs or cleaner logs

## Output

### Console Output
The script provides detailed progress information including:
- Parameter evaluation progress
- Best results found so far
- Memory usage monitoring
- Final optimization summary

### Generated Files
Each optimization run creates an optimized `.env` file with the naming convention:
```
.env_{SYMBOLS}_{YYYYMMDD_HHMMSS}
```

Examples:
- `.env_BTCEUR_20240115_143022`
- `.env_BTCEUR_ETHEUR_20240115_143045`
- `.env_BTCEUR_ETHEUR_ADAEUR_SOLEUR_XRPEUR_20240115_143102`

### Optimized .env File Contents
The generated `.env` file includes:
- **Header with optimization metadata** (method, objective, performance)
- **All original .env.example variables** (preserved for compatibility)
- **Optimized trading parameters** (updated with best values found)
- **Symbol configuration** (set to the optimized symbols)

## Understanding the Results

### Key Metrics Displayed
- **Objective Value**: The optimized metric (e.g., Sharpe ratio: 1.25)
- **Total Trades**: Number of trades in backtest validation
- **Performance Metrics**: Detailed breakdown of trading performance
- **Optimal Parameters**: The best parameter combination found

### Parameter Explanations
- `buy_threshold`: Confidence threshold for buy signals (0.6-0.8)
- `sell_threshold`: Confidence threshold for sell signals (0.2-0.4)
- `lstm_delta_threshold`: LSTM prediction sensitivity (0.005-0.025)
- `risk_per_trade`: Position size as % of capital (0.01-0.035)
- `stop_loss_pct`: Stop loss percentage (0.015-0.05)
- `take_profit_pct`: Take profit percentage (0.03-0.12)
- `max_capital_per_trade`: Maximum position size (0.05-0.18)
- `max_positions`: Maximum concurrent positions (3-15)

## Best Practices

### Recommended Workflows

#### 1. Quick Exploration (15-30 minutes)
```bash
python optimized_variables.py --symbols BTCEUR ETHEUR \
    --iterations 30 \
    --mode balanced
```

#### 2. Thorough Optimization (1-2 hours)
```bash
python optimized_variables.py --symbols BTCEUR ETHEUR ADAEUR \
    --iterations 100 \
    --mode profit_focused \
    --objective calmar_ratio
```

#### 3. Conservative Strategy (45-90 minutes)
```bash
python optimized_variables.py --symbols BTCEUR \
    --mode conservative \
    --objective sharpe_ratio \
    --iterations 75
```

#### 4. Aggressive Strategy (1-2 hours)
```bash
python optimized_variables.py --symbols BTCEUR ETHEUR ADAEUR SOLEUR XRPEUR \
    --mode aggressive \
    --objective total_return \
    --iterations 120
```

### Optimization Tips

1. **Start with fewer symbols** for faster iteration
2. **Use profit_focused mode** for maximum profitability
3. **Use calmar_ratio objective** for risk-conscious trading
4. **Run 50-100 iterations** for reliable results
5. **Test different modes** to understand parameter sensitivity

### Performance Expectations

| Setup | Symbols | Iterations | Time | Quality |
|-------|---------|------------|------|---------|
| Quick | 1-2 | 30 | 15-30 min | Good |
| Standard | 2-3 | 50 | 30-60 min | Very Good |
| Thorough | 3-5 | 100 | 1-2 hours | Excellent |
| Comprehensive | 5 | 200 | 2-4 hours | Maximum |

## Integration with Paper Trader

### Using Optimized Parameters
1. Run optimization to generate `.env_{symbols}_{date}` file
2. Copy or rename the file to `.env` in your project directory
3. Run the main paper trader:
   ```bash
   python main_paper_trader.py
   ```

### Validating Results
- Monitor the paper trader performance for several days
- Compare results with previous configurations
- Fine-tune if needed by running additional optimizations

## Troubleshooting

### Common Issues

#### 1. "Missing data files"
**Problem**: Database files not found
**Solution**: Ensure `data/{symbol}_15m.db` files exist for target symbols

#### 2. "Limited models available"
**Problem**: Model files missing for some symbols
**Solution**: Check `models/lstm/` and `models/xgboost/` directories

#### 3. "No valid optimization results"
**Problem**: All parameter combinations failed
**Solution**: 
- Try different optimization mode (e.g., `balanced` instead of `conservative`)
- Reduce minimum trade requirements
- Check data quality and date ranges

#### 4. "Optimization taking too long"
**Problem**: Long execution times
**Solution**:
- Reduce `--iterations`
- Use fewer `--symbols`
- Try `--method grid` with low `--grid-points`

### Performance Optimization

#### For Faster Results:
```bash
python optimized_variables.py --symbols BTCEUR \
    --iterations 25 \
    --initial-random 5 \
    --quiet
```

#### For Maximum Quality:
```bash
python optimized_variables.py --symbols BTCEUR ETHEUR ADAEUR \
    --iterations 150 \
    --initial-random 20 \
    --mode profit_focused \
    --objective calmar_ratio
```

## Scientific Background

### Bayesian Optimization
The script uses Gaussian Process Regression to model the parameter-performance relationship, enabling:
- **Efficient exploration**: Focus on promising parameter regions
- **Uncertainty quantification**: Balance exploration vs exploitation
- **Sample efficiency**: Find good parameters with fewer evaluations

### Parameter Ranges
Based on academic research in quantitative finance:
- **Kelly Criterion**: For position sizing (1-3% risk per trade)
- **Volatility Studies**: For stop loss levels (2-4% for crypto)
- **Signal Confidence**: ML reliability thresholds (65-80%)
- **Portfolio Theory**: Diversification (5-12 positions optimal)

### Statistical Rigor
- **Minimum trades**: 5+ trades required for significance
- **Cross-validation**: Walk-forward analysis inherent in backtesting
- **Risk adjustment**: Calmar ratio preferred over Sharpe for drawdown sensitivity
- **Realistic costs**: 0.2% trading fees and 0.1% slippage

## Example Complete Workflow

```bash
# 1. Quick exploration to understand parameter space
python optimized_variables.py --symbols BTCEUR --iterations 20 --quiet

# 2. Multi-symbol optimization for diversification
python optimized_variables.py --symbols BTCEUR ETHEUR ADAEUR \
    --mode profit_focused \
    --objective calmar_ratio \
    --iterations 75

# 3. Conservative validation run
python optimized_variables.py --symbols BTCEUR ETHEUR \
    --mode conservative \
    --objective sharpe_ratio \
    --iterations 50

# 4. Use best .env file with paper trader
cp .env_BTCEUR_ETHEUR_ADAEUR_20240115_143045 .env
python main_paper_trader.py
```

This comprehensive optimization approach ensures you find scientifically-validated parameters that maximize your paper trading bot's profitability while managing risk appropriately.