# Parameter Optimization Engine

This module provides utilities to automatically search for the most profitable
configuration of the paper trader backtester. It runs multiple backtests with
different parameters and reports the top performing settings.

## Usage

Run one of the presets or create a custom optimization:

```bash
python optimization_config.py --preset quick_explore
python optimization_config.py --method random_search --objective total_return --param-space aggressive_space
```

Results are stored in the `optimization_results/` directory as JSON files
containing the highest ranking parameter sets.

Refer to `optimization_config.py` for examples of defining parameter spaces and
presets.
