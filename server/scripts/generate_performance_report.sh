#!/bin/bash
# Generate performance report

SCRIPT_DIR="/opt/trading_bot"
source /etc/trading_bot/.env

# Generate performance metrics
python3 -c "
import json
import time
from pathlib import Path

# Mock performance data (replace with actual calculation)
metrics = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'portfolio_value': 10500.50,
    'daily_pnl': 150.25,
    'total_return': 0.05,
    'sharpe_ratio': 1.25,
    'win_rate': 0.65,
    'active_positions': 2,
    'cpu_usage': 15.2,
    'memory_usage': 45.8,
    'disk_usage': 12.3
}

# Save to file
with open('$SCRIPT_DIR/logs/performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print('Performance metrics updated')
"