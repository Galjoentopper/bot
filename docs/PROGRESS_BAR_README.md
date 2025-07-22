# Progress Bar and Verbose Output Control

## Problem
During parameter optimization and backtesting, scripts can generate repetitive output that floods the terminal:

```
✅ Successfully scaled 97 data points
✅ Created 1 LSTM sequences
✅ Successfully scaled 97 data points
✅ Created 1 LSTM sequences
.... (repeated hundreds of times)
```

## Solution
The optimization and backtesting systems implement verbose output controls:

- **During optimization**: Clean progress indicators with minimal output
- **Manual backtesting**: Full detailed output available for debugging
- **Configurable verbosity**: Users can control output level via command-line flags

## Usage

### Clean Optimization (recommended)
```bash
# Quiet mode suppresses detailed output
python optimized_variables.py --symbols BTCEUR --quiet

# Normal mode shows progress but not verbose details
python optimized_variables.py --symbols BTCEUR
```

### Verbose Backtesting (for debugging)
```python
from scripts.backtest_models import ModelBacktester, BacktestConfig

# Enable verbose output for detailed debugging
config = BacktestConfig()
config.verbose = True
backtester = ModelBacktester(config)
results = backtester.run_backtest(['BTCEUR'])
```

### Scripts with Verbose Control

#### optimized_variables.py
- `--quiet` flag suppresses detailed optimization progress
- Default mode shows progress bars and summaries
- Useful for monitoring long-running optimizations

#### scripts/backtest_models.py  
- `BacktestConfig.verbose` controls detailed backtesting output
- Automatically set to `False` during optimization
- Can be manually enabled for debugging individual backtests

## What This Provides
- **Clean Optimization**: Progress indicators without information overload
- **Debugging Support**: Full detailed output when needed
- **Flexible Control**: Command-line and programmatic verbosity control
- **Better UX**: Clear progress tracking during long operations

## Implementation Details
- `BacktestConfig.verbose` controls all repetitive print statements in backtesting
- Progress indicators show meaningful updates instead of verbose logs  
- Parameter optimization uses clean progress tracking
- Debug information remains available when explicitly requested