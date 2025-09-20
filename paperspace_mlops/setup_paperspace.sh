#!/bin/bash
# Paperspace Gradient Environment Setup Script
# This script configures the Paperspace environment for superior ensemble training

set -e

echo "🚀 Setting up Paperspace Gradient environment for Superior Ensemble Training"
echo "============================================================================"

# Check if running on Paperspace
if [ -z "$PAPERSPACE_JOB_ID" ] && [ -z "$GRADIENT_WORKSPACE_ID" ]; then
    echo "⚠️  Warning: Not running on Paperspace Gradient"
    echo "   This script is optimized for Paperspace but will continue anyway"
fi

# Environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p /tmp/training_logs
mkdir -p /tmp/model_exports
mkdir -p /tmp/validation_results

# Check system resources
echo "🔍 System Resource Check:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  Disk space: $(df -h / | tail -1 | awk '{print $4}')"

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
    while IFS=, read name total free; do
        echo "    $name: ${total}MB total, ${free}MB free"
    done
else
    echo "  GPU: Not available"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd /opt/trading_bot/bot
pip install --upgrade pip setuptools wheel

# Install main requirements
pip install -r paperspace_mlops/requirements.txt

# Install project in development mode
pip install -e .

# AWS CLI configuration check
echo "🔐 Checking AWS configuration..."
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "⚠️  AWS_ACCESS_KEY_ID not set"
    echo "   Please set AWS credentials for S3 model export"
else
    echo "✓ AWS credentials configured"
    # Test S3 access
    if aws s3 ls s3://$AWS_MODELS_BUCKET/ > /dev/null 2>&1; then
        echo "✓ S3 bucket access confirmed"
    else
        echo "⚠️  Cannot access S3 bucket: $AWS_MODELS_BUCKET"
    fi
fi

# Environment validation
echo "🔧 Validating environment..."
python3 -c "
import torch
import tensorflow as tf
import lightgbm
import optuna
import stable_baselines3
print('✓ PyTorch:', torch.__version__)
print('✓ TensorFlow:', tf.__version__)
print('✓ LightGBM:', lightgbm.__version__)
print('✓ Optuna:', optuna.__version__)
print('✓ Stable-Baselines3:', stable_baselines3.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU count:', torch.cuda.device_count())
    print('✓ GPU name:', torch.cuda.get_device_name(0))
"

# Create training script launcher
echo "📝 Creating training launcher..."
cat > /tmp/run_training.sh << 'EOF'
#!/bin/bash
# Quick training launcher script

cd /opt/trading_bot/bot

echo "🎯 Starting Superior Ensemble Training on Paperspace"
echo "Time: $(date)"
echo "Workspace: $GRADIENT_WORKSPACE_ID"
echo "Job: $PAPERSPACE_JOB_ID"

# Run the training with monitoring
python paperspace_mlops/paperspace_superior_training.py \
    --monitor \
    "$@"

echo "Training completed at: $(date)"
EOF

chmod +x /tmp/run_training.sh

# Create quick test launcher
cat > /tmp/quick_test.sh << 'EOF'
#!/bin/bash
# Quick test launcher

cd /opt/trading_bot/bot
python paperspace_mlops/paperspace_superior_training.py \
    --quick-test \
    --symbols BTCEUR \
    --models ppo
EOF

chmod +x /tmp/quick_test.sh

echo ""
echo "✅ Paperspace environment setup complete!"
echo ""
echo "🎯 Available commands:"
echo "  Full training:     /tmp/run_training.sh"
echo "  Quick test:        /tmp/quick_test.sh"
echo "  Custom training:   python paperspace_mlops/paperspace_superior_training.py [options]"
echo ""
echo "📚 Usage examples:"
echo "  # Train all models for all symbols"
echo "  /tmp/run_training.sh"
echo ""
echo "  # Train specific symbols and models"
echo "  /tmp/run_training.sh --symbols BTCEUR,ETHEUR --models ppo,gru"
echo ""
echo "  # Quick test with one symbol"
echo "  /tmp/quick_test.sh"
echo ""
echo "🚀 Ready for superior ensemble training!"
