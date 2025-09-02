#!/bin/bash
# Test script to run trader.py for maximum 5 minutes then kill it
# This addresses the guideline: "do run scripts/trader.py for max 5 minutes than kill it"

echo "🧪 Trader Test Script (Shell)"
echo "This will run trader.py for a maximum of 5 minutes"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADER_PATH="$SCRIPT_DIR/trader.py"

if [ ! -f "$TRADER_PATH" ]; then
    echo "❌ Error: trader.py not found at $TRADER_PATH"
    exit 1
fi

echo "🚀 Starting trader.py with 5 minute timeout..."
echo "📍 Command: python $TRADER_PATH"
echo "⏰ Will automatically kill after 300 seconds"

# Start the trader in background
python "$TRADER_PATH" &
TRADER_PID=$!

echo "✅ Trader process started with PID: $TRADER_PID"

# Function to kill trader process
kill_trader() {
    echo "⏰ Timeout reached. Killing trader process..."
    if kill -TERM $TRADER_PID 2>/dev/null; then
        echo "📤 Sent SIGTERM to trader process"
        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $TRADER_PID 2>/dev/null; then
                echo "✅ Trader process terminated gracefully"
                return 0
            fi
            sleep 1
        done
        # Force kill if still running
        echo "⚠️  Trader process didn't respond to SIGTERM, force killing..."
        kill -KILL $TRADER_PID 2>/dev/null
        echo "💀 Trader process force killed"
    else
        echo "⚠️  Trader process already finished"
    fi
}

# Set up trap to kill trader on script exit
trap kill_trader EXIT

# Monitor for 5 minutes (300 seconds)
echo "⏳ Waiting for trader to complete or timeout..."
SECONDS_ELAPSED=0
while [ $SECONDS_ELAPSED -lt 300 ]; do
    # Check if trader process is still running
    if ! kill -0 $TRADER_PID 2>/dev/null; then
        echo "✅ Trader process completed on its own"
        break
    fi

    # Show progress every 30 seconds
    if [ $((SECONDS_ELAPSED % 30)) -eq 0 ] && [ $SECONDS_ELAPSED -gt 0 ]; then
        REMAINING=$((300 - SECONDS_ELAPSED))
        echo "⏱️  Elapsed: ${SECONDS_ELAPSED}s | Remaining: ${REMAINING}s"
    fi

    sleep 1
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + 1))
done

# Check final status
if kill -0 $TRADER_PID 2>/dev/null; then
    echo "⏰ 5 minute timeout reached"
else
    echo "✅ Trader completed before timeout"
fi

echo "📊 Test completed"
echo "🧹 Cleaning up..."

# Cleanup happens automatically via trap
echo "✅ Cleanup completed"