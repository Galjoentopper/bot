#!/bin/bash
"""
PROFESSIONAL SUPERIOR MODEL EXPORT SYSTEM OVERVIEW
==================================================

Complete overview of your export system and current status.
"""

echo "🎯 SUPERIOR MODEL EXPORT SYSTEM"
echo "================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "export_to_hetzner.py" ]]; then
    echo "❌ Please run from /notebooks/bot/paperspace_mlops directory"
    exit 1
fi

echo "📁 System Files:"
echo "=================="
ls -la *.py *.sh *.md *.json 2>/dev/null | awk '{print "   " $0}'
echo ""

echo "🚀 Available Commands:"
echo "======================"
echo "   ./setup_hetzner_export.sh     - First-time setup (SSH, config)"
echo "   ./quick_export.sh             - Quick model export"
echo "   python3 export_to_hetzner.py  - Professional export"
echo "   python3 validate_integration.py - Test integration"
echo "   python3 prepare_hetzner_system.py - System preparation"
echo ""

# Check setup status
echo "🔍 Setup Status:"
echo "================"
if [ -f "hetzner_config.json" ]; then
    echo "   ✅ Configuration file exists"

    # Parse config
    HETZNER_HOST=$(grep '"hetzner_host"' hetzner_config.json | cut -d'"' -f4)
    HETZNER_USER=$(grep '"hetzner_user"' hetzner_config.json | cut -d'"' -f4)

    if [ "$HETZNER_HOST" != "your-hetzner-ip" ]; then
        echo "   ✅ Configured for: $HETZNER_USER@$HETZNER_HOST"
    else
        echo "   ⚠️  Configuration needs setup"
    fi
else
    echo "   ❌ No configuration found - run setup first"
fi

if [ -f "../.env.hetzner" ]; then
    echo "   ✅ Environment file exists"
else
    echo "   ❌ No environment file - run setup first"
fi

SSH_KEY="$HOME/.ssh/hetzner_key"
if [ -f "$SSH_KEY" ]; then
    echo "   ✅ SSH key exists"
else
    echo "   ❌ No SSH key - run setup first"
fi

echo ""

# Check models status
echo "📊 Local Models Status:"
echo "======================="
if [ -d "../models/superior" ]; then
    TOTAL_MODELS=$(find ../models/superior -name "*.zip" | wc -l)
    echo "   ✅ Superior models directory exists"
    echo "   📁 Total model files: $TOTAL_MODELS"

    echo ""
    echo "   Per-symbol breakdown:"
    for symbol in BTCEUR ETHEUR ADAEUR DOTEUR LINKEUR; do
        if [ -d "../models/superior/$symbol" ]; then
            MODEL_COUNT=$(find "../models/superior/$symbol" -name "*.zip" | wc -l)
            if [ $MODEL_COUNT -gt 0 ]; then
                echo "     ✅ $symbol: $MODEL_COUNT models"
            else
                echo "     ⚠️  $symbol: No models"
            fi
        else
            echo "     ❌ $symbol: Directory missing"
        fi
    done
else
    echo "   ❌ No superior models found"
    echo "   ℹ️  Run training first: ./train_all_symbols.sh"
fi

echo ""

# Quick system test
echo "🧪 Quick System Test:"
echo "===================="

# Test Python imports
echo "   Testing Python dependencies..."
python3 -c "
try:
    import subprocess, json, tempfile, logging
    print('   ✅ Core Python modules available')
except ImportError as e:
    print(f'   ❌ Missing Python modules: {e}')

try:
    import stable_baselines3
    print('   ✅ stable-baselines3 available')
except ImportError:
    print('   ⚠️  stable-baselines3 not available (needed for validation)')
"

echo ""

# Show next steps
echo "🎯 Next Steps:"
echo "==============="

if [ ! -f "hetzner_config.json" ]; then
    echo "   1. Run first-time setup:"
    echo "      ./setup_hetzner_export.sh"
    echo ""
elif [ ! -d "../models/superior" ]; then
    echo "   1. Train superior models first:"
    echo "      cd /notebooks/bot"
    echo "      ./train_all_symbols.sh 200000 full"
    echo ""
else
    echo "   🚀 Ready to export! Choose one:"
    echo ""
    echo "   Quick export:"
    echo "      ./quick_export.sh"
    echo ""
    echo "   Professional export:"
    echo "      python3 export_to_hetzner.py"
    echo ""
    echo "   Export with auto-restart:"
    echo "      python3 export_to_hetzner.py --auto-restart"
    echo ""
    echo "   Test first (recommended):"
    echo "      python3 export_to_hetzner.py --dry-run"
    echo ""
fi

echo "📖 Documentation:"
echo "=================="
echo "   📄 Complete guide: README_EXPORT.md"
echo "   📝 View with: cat README_EXPORT.md"
echo ""

echo "🎉 SYSTEM READY FOR PROFESSIONAL MODEL EXPORT"
echo "=============================================="
echo ""
echo "Your superior PPO models will seamlessly integrate with:"
echo "   ./bin/system_manager start  (on Hetzner)"
echo ""