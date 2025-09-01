#!/bin/bash

# Script to run trader.py for a maximum of 5 minutes
# This addresses the user's guideline to run scripts/trader.py for max 5 minutes
# and then kill it, rather than waiting for it to end by itself.

echo "Running trader.py for maximum 5 minutes (300 seconds)..."

# Check if trader.py exists
if [ ! -f "scripts/trader.py" ]; then
    echo "ERROR: scripts/trader.py not found!"
    exit 1
fi

# Check for configuration file
CONFIG_FILE=""
if [ -f "training_config.yaml" ]; then
    CONFIG_FILE="--config training_config.yaml"
    echo "Using configuration file: training_config.yaml"
else
    echo "No configuration file found, running with defaults"
fi

# Run trader.py with timeout
timeout 300s python scripts/trader.py $CONFIG_FILE --iterations 100

# Check the result
if [ $? -eq 124 ]; then
    echo "Trader script timed out after 5 minutes (this is expected per guidelines)"
    echo "✅ Test completed successfully - script was terminated as intended"
    exit 0
elif [ $? -eq 0 ]; then
    echo "Trader script completed successfully within 5 minutes"
    echo "✅ Test completed successfully"
    exit 0
else
    echo "Trader script failed with an error"
    echo "❌ Test failed"
    exit 1
fi