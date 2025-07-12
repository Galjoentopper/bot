# Cryptocurrency Trading Model Backtesting System

## What This Script Does

The `run_backtest.py` script tests the performance of hybrid LSTM-XGBoost cryptocurrency trading models using historical data. It simulates real trading conditions with:

- **Walk-Forward Validation**: Uses 4-month training periods to predict 1-month ahead
- **Realistic Trading**: Includes fees, slippage, position sizing, and stop-loss
- **Risk Management**: Controls position limits and trade frequency
- **Performance Analysis**: Generates detailed metrics and visualizations

The script loads pre-trained models for each time window and simulates trading decisions to evaluate strategy profitability.

## Installation

```bash
pip install -r requirements_backtest.txt
```

Ensure you have:
- Market data in `data/` folder (e.g., `btceur_15m.db`)
- Trained models in `models/` folder (LSTM, XGBoost, scalers)

## Available Flags

| Flag | Description | Options/Values |
|------|-------------|----------------|
| `--symbol` | Cryptocurrency pair to test | `BTCEUR`, `ETHEUR`, `ADAEUR`, `SOLEUR`, `XRPEUR` |
| `--config` | Trading strategy configuration | `aggressive`, `very_aggressive`, `fast_test` |
| `--max-windows` | Limit number of testing windows | Any positive integer (e.g., `5`, `10`, `20`) |
| `--random` | Start from random date | No value needed (flag only) |
| `--save-plots` | Generate performance charts | No value needed (flag only) |
| `--debug` | Generate detailed CSV with trading data | No value needed (flag only) |

### Configuration Details
- **aggressive**: 2% risk per trade, 70%/30% buy/sell thresholds
- **very_aggressive**: 3% risk per trade, 60%/40% buy/sell thresholds  
- **fast_test**: Quick testing with relaxed parameters

### Special Features
- `--random` automatically limits to 10 windows and prevents data leakage
- `--debug` creates `debug_detailed.csv` with comprehensive trading information
- `--save-plots` generates equity curves and performance visualizations

## Usage Examples

### Basic Usage
```bash
# Simple backtest with aggressive strategy
python run_backtest.py --symbol BTCEUR --config aggressive

# Test different cryptocurrency
python run_backtest.py --symbol ETHEUR --config very_aggressive
```

### Limited Testing
```bash
# Test only 5 windows for quick results
python run_backtest.py --symbol BTCEUR --config aggressive --max-windows 5

# Random start with automatic 10-window limit
python run_backtest.py --symbol BTCEUR --config very_aggressive --random
```

### Advanced Analysis
```bash
# Generate performance charts
python run_backtest.py --symbol BTCEUR --config aggressive --save-plots

# Debug mode with detailed CSV output
python run_backtest.py --symbol BTCEUR --config aggressive --debug --max-windows 3

# Comprehensive analysis
python run_backtest.py --symbol ETHEUR --config very_aggressive --random --save-plots --debug
```

### Quick Testing
```bash
# Fast configuration for rapid testing
python run_backtest.py --symbol BTCEUR --config fast_test --max-windows 5

# Multiple symbols comparison
python run_backtest.py --symbol ADAEUR --config aggressive --random
python run_backtest.py --symbol SOLEUR --config aggressive --random
```

## Output Files

Results are saved in `backtests/SYMBOL/`:
- `trades.csv` - Individual trade records
- `equity_curve.csv` - Portfolio value over time  
- `performance_metrics.json` - Summary statistics
- `performance_plots.png` - Charts (with `--save-plots`)
- `debug_detailed.csv` - Detailed trading data (with `--debug`)

### Key Metrics
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest loss from peak

## Troubleshooting

### Common Issues
- **No trades generated**: Try `--config fast_test` or `--random` for different market conditions
- **Model loading errors**: Ensure models exist in `models/` folder for the symbol
- **Data issues**: Check that `.db` files exist in `data/` folder

### Debug Mode
Use `--debug` to generate detailed CSV with:
- Model predictions (LSTM delta, XGBoost probabilities)
- Trading signals (BUY/SELL/HOLD decisions)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Trade execution details and portfolio state

---

**Note**: This backtesting system is for research and educational purposes. Past performance does not guarantee future results.