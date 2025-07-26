#!/usr/bin/env python3
"""Script to fix attribute references in main_paper_trader.py"""

import re

# Read the file
with open('main_paper_trader.py', 'r') as f:
    content = f.read()

# Define the attribute mappings
trading_attributes = [
    'initial_capital', 'max_positions', 'max_positions_per_symbol', 'base_position_size',
    'max_position_size', 'min_position_size', 'take_profit_pct', 'stop_loss_pct',
    'trailing_stop_pct', 'min_profit_for_trailing', 'max_hold_hours', 'min_hold_time_minutes',
    'position_cooldown_minutes', 'max_daily_trades_per_symbol', 'candle_interval'
]

model_attributes = [
    'model_path', 'sequence_length', 'min_window', 'max_window', 'default_window',
    'min_confidence_threshold', 'min_signal_strength', 'lstm_weight', 'xgb_weight', 'caboose_weight'
]

# Fix trading settings attributes
for attr in trading_attributes:
    pattern = f'self\.settings\.{attr}'
    replacement = f'self.settings.trading_settings.{attr}'
    content = re.sub(pattern, replacement, content)

# Fix model settings attributes  
for attr in model_attributes:
    pattern = f'self\.settings\.{attr}'
    replacement = f'self.settings.model_settings.{attr}'
    content = re.sub(pattern, replacement, content)

# Write back the fixed content
with open('main_paper_trader.py', 'w') as f:
    f.write(content)

print("Fixed main_paper_trader.py attribute references")