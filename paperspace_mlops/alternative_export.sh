#!/bin/bash
# Alternative Export Methods When SSH is Not Available
# ===================================================

echo "🔄 ALTERNATIVE MODEL EXPORT METHODS"
echo "===================================="
echo ""

echo "Since SSH is not available, here are alternative methods:"
echo ""

echo "📦 METHOD 1: Create Transfer Package"
echo "===================================="
echo "Create a downloadable package of your models:"
echo ""

# Create transfer package
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="superior_models_${TIMESTAMP}.tar.gz"

echo "Creating model package..."
cd /notebooks/bot

if [ -d "models/superior" ]; then
    tar -czf "$PACKAGE_NAME" models/superior/
    PACKAGE_SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)

    echo "✅ Package created: $PACKAGE_NAME"
    echo "   Size: $PACKAGE_SIZE"
    echo "   Location: /notebooks/bot/$PACKAGE_NAME"
    echo ""
    echo "📥 Download this file through Paperspace interface"
    echo "📤 Upload to your Hetzner server manually"
    echo ""
else
    echo "❌ No superior models found in models/superior/"
    echo "   Run training first: ./train_all_symbols.sh"
    exit 1
fi

echo "📦 METHOD 2: S3 Transfer (if configured)"
echo "========================================"
echo "Upload to S3, then download on Hetzner:"
echo ""
echo "1. Upload from Paperspace:"
echo "   python3 paperspace_mlops/export_to_s3.py"
echo ""
echo "2. Download on Hetzner:"
echo "   aws s3 sync s3://your-bucket/models/superior/ /opt/trading_bot/models/superior/"
echo ""

echo "📦 METHOD 3: HTTP Transfer"
echo "=========================="
echo "Start a simple HTTP server for download:"
echo ""
echo "1. Start server (run this in another terminal):"
echo "   cd /notebooks/bot"
echo "   python3 -m http.server 8000"
echo ""
echo "2. Download on Hetzner:"
echo "   curl -O http://PAPERSPACE_IP:8000/$PACKAGE_NAME"
echo "   tar -xzf $PACKAGE_NAME"
echo ""

echo "📦 METHOD 4: Manual File Transfer Instructions"
echo "=============================================="
echo ""
echo "🎯 QUICK HETZNER SETUP (Manual)"
echo "==============================="
echo ""
echo "On your Hetzner server, create this structure:"
echo ""
echo "mkdir -p /opt/trading_bot/models/superior"
echo ""
echo "Then extract your models:"
echo "cd /opt/trading_bot"
echo "tar -xzf $PACKAGE_NAME"
echo ""

echo "🔧 HETZNER CONFIGURATION UPDATE"
echo "==============================="
echo ""
echo "After transferring models, update your trading config:"
echo ""

cat > hetzner_config_update.sh << 'EOF'
#!/bin/bash
# Run this on your Hetzner server after transferring models

cd /opt/trading_bot

# Backup current config
cp config/trading_config.yaml config/trading_config.yaml.backup_$(date +%Y%m%d_%H%M%S)

# Update configuration to use superior models
sed -i 's/ensemble_type: .*/ensemble_type: "superior_ppo"/' config/trading_config.yaml

# Add superior model configuration if not exists
if ! grep -q "superior_config:" config/trading_config.yaml; then
    cat >> config/trading_config.yaml << 'SUPERIOR_CONFIG'

# Superior model configuration
superior_config:
  feature_count: 104
  window_size: 32
  prediction_horizons: ['1h', '3h', '6h', '12h', '24h']
  cost_adjustment: true
  transaction_cost_bps: 10
  model_type: 'resource_aware_ppo'

model_weights:
  superior: 0.70
  lightgbm: 0.20
  gru: 0.10
  ppo: 0.00
SUPERIOR_CONFIG
fi

echo "✅ Configuration updated for superior models"

# Validate models are present
echo "🧪 Validating models..."
SUPERIOR_MODELS=$(find models/superior -name "*.zip" 2>/dev/null | wc -l)
echo "Found $SUPERIOR_MODELS superior model files"

if [ $SUPERIOR_MODELS -ge 5 ]; then
    echo "✅ Superior models ready for trading"
    echo "🚀 Start trading: ./bin/system_manager start"
else
    echo "⚠️  Some models missing - verify transfer"
fi
EOF

chmod +x hetzner_config_update.sh

echo "Created hetzner_config_update.sh - copy this to your Hetzner server"
echo ""

echo "🎉 SUMMARY"
echo "=========="
echo ""
echo "✅ Model package created: $PACKAGE_NAME"
echo "✅ Hetzner setup script created: hetzner_config_update.sh"
echo ""
echo "📋 NEXT STEPS:"
echo "==============="
echo "1. Download $PACKAGE_NAME from Paperspace"
echo "2. Upload to your Hetzner server"
echo "3. Extract: tar -xzf $PACKAGE_NAME"
echo "4. Run: ./hetzner_config_update.sh"
echo "5. Start trading: ./bin/system_manager start"
echo ""
echo "Your superior models will be integrated and ready!"