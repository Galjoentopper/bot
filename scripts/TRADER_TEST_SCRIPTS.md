# Trader Test Scripts with 5-Minute Timeout

This directory contains scripts to run the trading bots with a maximum 5-minute timeout, addressing the user's guideline:

> "do run scripts/trader.py for max 5 minutes than kill it. currently you wait till the scripts ends for itself but it will run continiously."

## Available Scripts

### Python Scripts
1. `run_trader_test.py` - Runs `scripts/trader.py` with a 5-minute timeout
2. `run_enhanced_trader_test.py` - Runs `scripts/enhanced_trader.py` with a 5-minute timeout

### Shell Scripts (Linux/macOS)
1. `run_trader_5min.sh` - Runs `scripts/trader.py` with a 5-minute timeout
2. `run_enhanced_trader_5min.sh` - Runs `scripts/enhanced_trader.py` with a 5-minute timeout

### Batch Scripts (Windows)
1. `run_trader_5min.bat` - Runs `scripts/trader.py` with a 5-minute timeout
2. `run_enhanced_trader_5min.bat` - Runs `scripts/enhanced_trader.py` with a 5-minute timeout

## Usage

### Python Scripts
```bash
# Run trader.py for 5 minutes
python scripts/run_trader_test.py

# Run enhanced_trader.py for 5 minutes
python scripts/run_enhanced_trader_test.py
```

### Shell Scripts
```bash
# Run trader.py for 5 minutes
./scripts/run_trader_5min.sh

# Run enhanced_trader.py for 5 minutes
./scripts/run_enhanced_trader_5min.sh
```

### Batch Scripts
```cmd
# Run trader.py for 5 minutes
scripts\run_trader_5min.bat

# Run enhanced_trader.py for 5 minutes
scripts\run_enhanced_trader_5min.bat
```

## Features

- **Automatic Timeout**: All scripts will automatically terminate the trader after 5 minutes
- **Configuration Support**: Automatically detects and uses `training_config.yaml` if present
- **Iteration Limiting**: Uses the `--iterations` argument to limit trading cycles
- **Cross-Platform**: Available in Python, Shell, and Batch formats for different environments
- **Clear Feedback**: Provides clear success/failure messages and logs output

## How It Works

1. The scripts check for the existence of the trader script and configuration file
2. They execute the trader with a 5-minute (300 seconds) timeout
3. If the trader completes within the time limit, the script exits successfully
4. If the trader exceeds the time limit, it is automatically terminated
5. All output is captured and displayed for debugging purposes

## Customization

To modify the timeout duration, edit the scripts and change:
- Python: `timeout_seconds=300` parameter
- Shell: `timeout 300s` command
- Batch: `WaitForExit(300000)` parameter (300000 milliseconds = 300 seconds)

To modify the iteration limit, change the `--iterations` argument value in each script.