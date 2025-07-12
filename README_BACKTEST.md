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
trade_bot_2.0/
├── scripts/                   # Core backtesting modules
│   ├── backtest_models.py     # Core backtesting engine
│   ├── backtest_config.py     # Configuration management
│   ├── backtest_analysis.py   # Results analysis and reporting
│   └── bootstrap_backtest.py  # Bootstrap robustness testing
├── run_backtest.py            # Main execution script
├── README_BACKTEST.md         # This documentation
├── requirements_backtest.txt   # Python dependencies
└── backtests/                 # Results directory
    ├── BTCEUR/               # Symbol-specific results
    ├── ETHEUR/
    ├── bootstrap/            # Bootstrap analysis results
    ├── sensitivity/          # Sensitivity analysis results
    └── multi_config/         # Multi-configuration results
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
# Run standard backtest with balanced configuration
python run_backtest.py --mode standard

# Run with specific symbols
python run_backtest.py --mode standard --symbols BTCEUR ETHEUR

# Run with aggressive configuration
python run_backtest.py --mode standard --config aggressive
```

### Advanced Testing
```bash
# Compare multiple configurations
python run_backtest.py --mode multi-config

# Test symbol-specific optimizations
python run_backtest.py --mode symbol-specific

# Run bootstrap robustness testing
python run_backtest.py --mode bootstrap --bootstrap-runs 10

# Sensitivity analysis
python run_backtest.py --mode sensitivity

# Comprehensive analysis of existing results
python run_backtest.py --mode analysis

# Run everything (comprehensive suite)
python run_backtest.py --mode all
```

## ⚙️ Configuration Options

### Trading Parameters
```python
class BacktestConfig:
    # Portfolio Management
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.02          # 2% risk per trade
    max_positions: int = 10               # Maximum simultaneous positions
    max_trades_per_hour: int = 3          # Trade frequency limit
    
    # Trading Costs
    trading_fee: float = 0.002            # 0.2% per trade
    slippage: float = 0.001               # 0.1% slippage
    
    # Risk Management
    stop_loss_pct: float = 0.03           # 3% stop loss
    
    # Signal Thresholds
    buy_threshold: float = 0.7            # XGBoost probability threshold
    sell_threshold: float = 0.3
    lstm_delta_threshold: float = 0.5     # LSTM prediction threshold
```

### Preset Configurations
- **Conservative**: Lower risk (1% per trade), higher thresholds (80%/20%)
- **Balanced**: Standard settings (2% risk, 70%/30% thresholds)
- **Aggressive**: Higher risk (3% per trade), lower thresholds (60%/40%)
- **High Frequency**: Many small trades (0.5% risk, 55%/45% thresholds)

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

1. **"Database not found" Error**:
   ```
   FileNotFoundError: Database not found: data/btceur_15m.db
   ```
   **Solution**: Ensure data files are in the correct location

2. **"Models not found" Warning**:
   ```
   Models not found for window 1, skipping...
   ```
   **Solution**: Train models first using `train_hybrid_models.py`

3. **Memory Issues**:
   **Solution**: Reduce bootstrap runs or test fewer symbols

4. **Slow Performance**:
   **Solution**: Use fewer symbols or shorter time periods for testing

### Debug Mode
```bash
# Run with verbose output for debugging
python run_backtest.py --mode standard --verbose
```

## 🔄 Workflow Recommendations

### 1. Initial Testing
```bash
# Start with standard backtest
python run_backtest.py --mode standard --symbols BTCEUR

# Analyze results
python run_backtest.py --mode analysis
```

### 2. Configuration Optimization
```bash
# Test different configurations
python run_backtest.py --mode multi-config --symbols BTCEUR ETHEUR

# Fine-tune with sensitivity analysis
python run_backtest.py --mode sensitivity --symbols BTCEUR
```

### 3. Robustness Validation
```bash
# Test stability with bootstrap
python run_backtest.py --mode bootstrap --symbols BTCEUR --bootstrap-runs 20
```

### 4. Production Deployment
```bash
# Final comprehensive test
python run_backtest.py --mode all
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