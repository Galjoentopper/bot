# Cryptocurrency Trading Model Backtesting System

A comprehensive backtesting framework for evaluating hybrid LSTM-XGBoost cryptocurrency trading models with walk-forward validation, realistic execution simulation, and robustness testing.

## 🚀 Features

### Core Backtesting
- **Walk-Forward Validation**: 4-month training, 1-month testing windows
- **Realistic Trading Simulation**: Slippage, fees, position sizing, stop-loss
- **Portfolio Management**: Risk management, position limits, trade frequency controls
- **Multi-Symbol Support**: Test across multiple cryptocurrency pairs

### Advanced Analysis
- **Bootstrap Robustness Testing**: Validate model stability with noise injection
- **Sensitivity Analysis**: Test parameter sensitivity across ranges
- **Multi-Configuration Testing**: Compare conservative, balanced, and aggressive strategies
- **Symbol-Specific Optimization**: Tailored parameters for each cryptocurrency

### Comprehensive Reporting
- **Performance Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, win rate
- **Visual Analytics**: Equity curves, trade distributions, risk-return plots
- **HTML Reports**: Detailed analysis with executive summaries
- **Excel Export**: Structured data for further analysis

## 📁 File Structure

```
bot/
├── run_backtest.py            # Main optimized backtesting script
├── train_hybrid_models.py     # Model training with walk-forward validation
├── README_BACKTEST.md         # This documentation
├── requirements_backtest.txt   # Python dependencies
├── data/                      # Market data databases
│   ├── btceur_15m.db
│   ├── etheur_15m.db
│   └── ...
├── models/                    # Trained models by window
│   ├── lstm/
│   ├── xgboost/
│   └── scalers/
└── backtests/                 # Results directory
    ├── BTCEUR/               # Symbol-specific results
    ├── ETHEUR/
    ├── configs/              # Configuration presets
    └── ...
```

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_backtest.txt
   ```

2. **Verify Data Structure**:
   Ensure your data is organized as:
   ```
   data/
   ├── adaeur_15m.db
   ├── btceur_15m.db
   ├── etheur_15m.db
   └── soleur_15m.db
   ```

3. **Verify Model Structure**:
   Ensure trained models are available:
   ```
   models/
   ├── lstm/
   ├── xgboost/
   └── scalers/
   ```

## 🎯 Quick Start

### Basic Backtest
```bash
# Run optimized backtest with balanced configuration
python run_backtest.py --symbol BTCEUR --config aggressive

# Run with different symbols
python run_backtest.py --symbol ETHEUR --config very_aggressive

# Run with random start date (auto-limited to 10 windows)
python run_backtest.py --symbol BTCEUR --config very_aggressive --random

# Run with custom window limit
python run_backtest.py --symbol BTCEUR --config aggressive --max-windows 5
```

### Advanced Testing
```bash
# Test different market conditions with random start
python run_backtest.py --symbol BTCEUR --config very_aggressive --random

# Quick testing with limited windows
python run_backtest.py --symbol ETHEUR --config aggressive --max-windows 10

# Generate performance plots
python run_backtest.py --symbol BTCEUR --config very_aggressive --save-plots

# Test multiple symbols sequentially
python run_backtest.py --symbol ADAEUR --config aggressive --random
python run_backtest.py --symbol SOLEUR --config very_aggressive --max-windows 15
```

## ⚙️ Configuration Options

### Available Configurations
- **aggressive**: Balanced risk settings (2% per trade, 70%/30% thresholds)
- **very_aggressive**: Higher risk settings (3% per trade, 60%/40% thresholds)
- **fast_test**: Quick testing configuration with relaxed parameters

### Command Line Options
```bash
--symbol SYMBOL          # Choose: BTCEUR, ETHEUR, ADAEUR, SOLEUR, XRPEUR
--config CONFIG          # Choose: aggressive, very_aggressive, fast_test
--max-windows N          # Limit number of testing windows (auto-set for --random)
--random                 # Randomly select starting date (prevents data leakage)
--save-plots             # Generate and save performance visualization plots
```

### Random Start Feature
The `--random` flag provides several benefits:
- **Prevents temporal data leakage** by using chronologically appropriate models
- **Tests different market conditions** by starting from random time periods
- **Auto-limits execution time** to 10 windows from the random start point
- **Maintains walk-forward validation integrity** with proper window calculation

## 📊 Understanding Results

### Key Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Output Files
```
backtests/SYMBOL/
├── trades.csv                 # Individual trade records
├── equity_curve.csv           # Portfolio value over time
├── performance_metrics.json   # Summary statistics
└── performance_plots.png      # Visualization charts
```

### Analysis Reports
- **HTML Report**: `backtests/detailed_analysis_report.html`
- **Excel Export**: `backtests/backtest_results.xlsx`
- **Comparison Plots**: `backtests/performance_comparison.png`

## 🔬 Advanced Features

### Bootstrap Robustness Testing
Tests model stability by adding small amounts of noise to price data:
```python
# Run 10 bootstrap iterations with 0.1% noise
python run_backtest.py --mode bootstrap --bootstrap-runs 10
```

### Sensitivity Analysis
Tests how sensitive results are to parameter changes:
- Buy/sell thresholds (0.6 to 0.8)
- Risk per trade (1% to 3%)
- Stop loss levels (2% to 4%)
- Maximum positions (5 to 15)

### Walk-Forward Validation
Prevents data leakage by using realistic time-based splits:
- Training: 4 months of historical data
- Testing: 1 month forward prediction
- Sliding: Move forward 1 month and repeat

## 🎛️ Customization

### Custom Configuration
```python
from scripts.backtest_config import BacktestConfig

# Create custom configuration
custom_config = BacktestConfig(
    initial_capital=50000.0,
    risk_per_trade=0.015,
    buy_threshold=0.75,
    max_positions=8
)

# Use in backtester
from scripts.backtest_models import ModelBacktester
backtester = ModelBacktester(custom_config)
results = backtester.run_backtest(['BTCEUR'])
```

### Symbol-Specific Settings
```python
from scripts.backtest_config import get_symbol_config

# Get optimized config for specific symbol
btc_config = get_symbol_config('BTCEUR')
eth_config = get_symbol_config('ETHEUR')
```

## 📈 Interpretation Guide

### Good Performance Indicators
- **Sharpe Ratio > 1.0**: Good risk-adjusted returns
- **Win Rate > 50%**: More winning than losing trades
- **Profit Factor > 1.5**: Wins significantly outweigh losses
- **Max Drawdown < 20%**: Reasonable risk exposure
- **Positive Total Return**: Profitable strategy

### Warning Signs
- **High Win Rate + Low Profit Factor**: Small wins, large losses
- **High Sharpe + High Drawdown**: Inconsistent performance
- **Very Few Trades**: Overly conservative, missing opportunities
- **Too Many Trades**: Over-trading, high transaction costs

## 🚨 Troubleshooting

### Common Issues

#### 1. No Trades Generated (0 trades, 0% return)
```bash
# Check if models exist for the calculated window
ls models/lstm/btceur_window_*.keras
ls models/xgboost/btceur_window_*.pkl

# Try a less aggressive configuration
python run_backtest.py --symbol BTCEUR --config fast_test --max-windows 5

# Use random start to test different market conditions
python run_backtest.py --symbol BTCEUR --config aggressive --random
```

#### 2. Model Loading Failures
```bash
# Verify models exist for the symbol and window
ls models/lstm/btceur_window_1.keras
ls models/xgboost/btceur_window_1.pkl
ls models/scalers/btceur_window_1_scaler.pkl

# Check if training was completed
python train_hybrid_models.py --symbol BTCEUR --config aggressive
```

#### 3. Random Backtest Takes Too Long
```bash
# The system automatically limits random backtests to 10 windows
# To manually control duration:
python run_backtest.py --symbol BTCEUR --config aggressive --random --max-windows 5
```

#### 4. Data Issues
```bash
# Check if data files exist
ls data/btceur_15m.db
ls data/etheur_15m.db

# Verify data has sufficient history (need at least 40+ months)
sqlite3 data/btceur_15m.db "SELECT MIN(timestamp), MAX(timestamp) FROM market_data;"
```

### Debug Mode
```bash
# Check available models and data
ls models/lstm/btceur_window_*.keras
ls data/btceur_15m.db

# Test with minimal windows for debugging
python run_backtest.py --symbol BTCEUR --config fast_test --max-windows 2
```

## 🔄 Workflow Recommendations

### 1. Initial Testing
```bash
# Start with basic backtest
python run_backtest.py --symbol BTCEUR --config aggressive --max-windows 5

# Test with random market conditions
python run_backtest.py --symbol BTCEUR --config aggressive --random
```

### 2. Configuration Optimization
```bash
# Test different risk levels
python run_backtest.py --symbol BTCEUR --config very_aggressive --random
python run_backtest.py --symbol BTCEUR --config fast_test --max-windows 10

# Compare across symbols
python run_backtest.py --symbol ETHEUR --config aggressive --random
python run_backtest.py --symbol ADAEUR --config very_aggressive --random
```

### 3. Performance Analysis
```bash
# Generate detailed plots
python run_backtest.py --symbol BTCEUR --config very_aggressive --save-plots

# Test longer periods
python run_backtest.py --symbol BTCEUR --config aggressive --max-windows 20
```

### 4. Production Validation
```bash
# Test multiple random periods for robustness
python run_backtest.py --symbol BTCEUR --config very_aggressive --random
python run_backtest.py --symbol BTCEUR --config very_aggressive --random
python run_backtest.py --symbol BTCEUR --config very_aggressive --random
```

## 📋 Performance Benchmarks

Based on historical cryptocurrency data:

| Metric | Conservative | Balanced | Aggressive |
|--------|-------------|----------|------------|
| Expected Sharpe | 0.8-1.2 | 1.0-1.5 | 0.6-1.8 |
| Win Rate | 45-55% | 40-60% | 35-65% |
| Max Drawdown | 5-15% | 10-25% | 15-35% |
| Trades/Month | 5-15 | 10-30 | 20-50 |

## 🤝 Contributing

To extend the backtesting system:

1. **Add New Metrics**: Extend `calculate_performance_metrics()` in `backtest_models.py`
2. **Custom Indicators**: Add to `TechnicalIndicators` class
3. **New Configurations**: Add presets to `backtest_config.py`
4. **Enhanced Analysis**: Extend `BacktestAnalyzer` class

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files in `backtests/` directory
3. Run with `--verbose` flag for detailed output
4. Ensure all dependencies are correctly installed

---

**Note**: This backtesting system is for research and educational purposes. Past performance does not guarantee future results. Always validate strategies with paper trading before live deployment.