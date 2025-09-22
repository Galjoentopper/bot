#!/bin/bash
set -e

echo "🚀 Training Superior PPO Models for All Symbols"
echo "==============================================="

# Configuration
TIMESTEPS=${1:-200000}  # Default 200k, can override with first argument
MODE=${2:-full}         # full or demo

if [[ "$MODE" == "demo" ]]; then
    TIMESTEPS=50000
    echo "🎮 Demo mode: $TIMESTEPS timesteps per symbol"
else
    echo "🎯 Full training: $TIMESTEPS timesteps per symbol"
fi

# Symbols to train
SYMBOLS=("BTCEUR" "ETHEUR" "ADAEUR" "DOTEUR" "LINKEUR")

echo "📊 Will train ${#SYMBOLS[@]} symbols: ${SYMBOLS[*]}"
echo "⏱️  Estimated time: $((${#SYMBOLS[@]} * TIMESTEPS / 10000)) minutes total"
echo ""

# Train each symbol
for symbol in "${SYMBOLS[@]}"; do
    echo ""
    echo "🎯 Starting training for $symbol..."
    echo "----------------------------------------"

    start_time=$(date +%s)

    if [[ "$MODE" == "demo" ]]; then
        python train_real_superior_ppo.py --symbol "$symbol" --demo
    else
        python train_real_superior_ppo.py --symbol "$symbol" --timesteps "$TIMESTEPS"
    fi

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "✅ $symbol completed in ${duration} seconds"
    echo ""
done

echo ""
echo "🎉 ALL SYMBOLS TRAINING COMPLETED!"
echo "=================================="
echo ""
echo "📁 Models saved in:"
find models/superior/ -name "*.zip" -o -name "best_model.zip" | head -20
echo ""
echo "📊 Training summary:"
for symbol in "${SYMBOLS[@]}"; do
    if [[ -d "models/superior/$symbol" ]]; then
        model_count=$(find "models/superior/$symbol" -name "*.zip" | wc -l)
        echo "   $symbol: $model_count model files created"
    else
        echo "   $symbol: ❌ No models found"
    fi
done