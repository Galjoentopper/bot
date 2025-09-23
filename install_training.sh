#!/bin/bash
# Installation script for training dependencies
# This handles common dependency conflicts

echo "🔧 Installing training dependencies with conflict resolution..."

# Install requirements first
pip install -r requirements-training.txt

# Install missing packages that might not be in requirements
pip install torch>=2.0.0 stable-baselines3>=2.7.0

# Handle blinker conflict if it exists (ignore errors)
pip install --force-reinstall --no-deps blinker 2>/dev/null || echo "⚠️ Blinker conflict handled (expected)"

echo "✅ Installation complete!"
echo ""
echo "🧪 Testing critical imports..."

# Test critical imports
python -c "
import sys
missing = []
try:
    import pandas
    print('✅ pandas - OK')
except ImportError:
    missing.append('pandas')
    print('❌ pandas - MISSING')

try:
    import numpy
    print('✅ numpy - OK')
except ImportError:
    missing.append('numpy')
    print('❌ numpy - MISSING')

try:
    import ta
    print('✅ ta - OK')
except ImportError:
    missing.append('ta')
    print('❌ ta - MISSING')

try:
    import torch
    print('✅ torch - OK')
except ImportError:
    missing.append('torch')
    print('❌ torch - MISSING')

try:
    import lightgbm
    print('✅ lightgbm - OK')
except ImportError:
    missing.append('lightgbm')
    print('❌ lightgbm - MISSING')

try:
    import stable_baselines3
    print('✅ stable_baselines3 - OK')
except ImportError:
    missing.append('stable_baselines3')
    print('❌ stable_baselines3 - MISSING')

try:
    import gymnasium
    print('✅ gymnasium - OK')
except ImportError:
    missing.append('gymnasium')
    print('❌ gymnasium - MISSING')

if missing:
    print(f'\\n⚠️ Missing packages: {missing}')
    print('Try running the installation again.')
    sys.exit(1)
else:
    print('\\n🎉 All critical packages installed successfully!')
    print('You can now run Train.ipynb')
"
