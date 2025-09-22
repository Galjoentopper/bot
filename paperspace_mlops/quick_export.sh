#!/bin/bash
"""
Ultra-Simple Model Export
=========================

One-command export for busy professionals.
Just run: ./quick_export.sh
"""

set -e

echo "🚀 ULTRA-SIMPLE SUPERIOR MODEL EXPORT"
echo "======================================"

# Check if setup was run
if [ ! -f "/notebooks/bot/paperspace_mlops/hetzner_config.json" ]; then
    echo "⚠️  First-time setup required!"
    echo "   Run: ./paperspace_mlops/setup_hetzner_export.sh"
    exit 1
fi

# Load configuration
source /notebooks/bot/.env.hetzner 2>/dev/null || {
    echo "❌ Configuration not found. Run setup first."
    exit 1
}

echo "📋 Export Configuration:"
echo "   Target: $HETZNER_USER@$HETZNER_HOST"
echo "   Models: All 5 superior PPO models"
echo "   Action: Export + Validate + Configure"
echo ""

# Confirm export
read -p "🎯 Ready to export superior models? (y/N): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ Export cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting export..."

# Run the professional export
python3 /notebooks/bot/paperspace_mlops/export_to_hetzner.py \
    --config /notebooks/bot/paperspace_mlops/hetzner_config.json

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Your Hetzner server now has SUPERIOR models!"
    echo "=================================================="
    echo ""
    echo "Next steps on your Hetzner server:"
    echo "  ssh $HETZNER_USER@$HETZNER_HOST"
    echo "  cd /opt/trading_bot"
    echo "  ./bin/system_manager start"
    echo ""
    echo "Your trading system will now use the superior PPO models!"
else
    echo ""
    echo "❌ Export failed. Check logs for details."
    echo "   Log file: /notebooks/bot/logs/hetzner_export.log"
fi