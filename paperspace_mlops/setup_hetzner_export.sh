#!/bin/bash
# Hetzner Export Setup Script
# ===========================
#
# Sets up SSH keys and configuration for seamless model export.
# Run this once before your first export.

set -e

echo "🔧 Setting up Hetzner Export System"
echo "===================================="

# Get configuration
read -p "Enter Hetzner server IP/domain: " HETZNER_HOST
read -p "Enter Hetzner username: " HETZNER_USER
read -p "Enter SSH port (default 22): " SSH_PORT
SSH_PORT=${SSH_PORT:-22}

# Validate inputs
if [ -z "$HETZNER_HOST" ] || [ -z "$HETZNER_USER" ]; then
    echo "❌ Error: Host and username are required"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Host: $HETZNER_HOST"
echo "  User: $HETZNER_USER"
echo "  Port: $SSH_PORT"
echo ""

# Create SSH key in bot directory (accessible location)
SSH_KEY_PATH="/notebooks/bot/.ssh/hetzner_key"
mkdir -p "/notebooks/bot/.ssh"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "🔐 Creating SSH key pair in accessible location..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "paperspace-hetzner-export"
    chmod 700 "/notebooks/bot/.ssh"
    chmod 600 "$SSH_KEY_PATH"
    chmod 644 "$SSH_KEY_PATH.pub"
    echo "✅ SSH key created: $SSH_KEY_PATH"
else
    echo "✅ SSH key already exists: $SSH_KEY_PATH"
fi

# Test SSH connection
echo "🔗 Testing SSH connection..."
if ssh -i "$SSH_KEY_PATH" -p "$SSH_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$HETZNER_USER@$HETZNER_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "✅ SSH connection successful"
else
    echo "⚠️  SSH connection failed. You may need to:"
    echo "   1. Copy the public key to your Hetzner server:"
    echo "      ssh-copy-id -i $SSH_KEY_PATH.pub -p $SSH_PORT $HETZNER_USER@$HETZNER_HOST"
    echo "   2. Or manually add this key to ~/.ssh/authorized_keys on Hetzner:"
    cat "$SSH_KEY_PATH.pub"
    echo ""
    echo "   Then run this setup script again to verify."

    read -p "Do you want me to try copying the key now? (y/n): " COPY_KEY
    if [ "$COPY_KEY" = "y" ] || [ "$COPY_KEY" = "Y" ]; then
        echo "📤 Copying SSH key to Hetzner server..."
        ssh-copy-id -i "$SSH_KEY_PATH.pub" -p "$SSH_PORT" "$HETZNER_USER@$HETZNER_HOST"

        # Test again
        if ssh -i "$SSH_KEY_PATH" -p "$SSH_PORT" -o ConnectTimeout=10 "$HETZNER_USER@$HETZNER_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
            echo "✅ SSH connection now working"
        else
            echo "❌ SSH connection still failing"
            exit 1
        fi
    fi
fi

# Create export configuration
CONFIG_FILE="/notebooks/bot/paperspace_mlops/hetzner_config.json"
echo "📝 Creating export configuration..."

cat > "$CONFIG_FILE" << EOF
{
    "hetzner_host": "$HETZNER_HOST",
    "hetzner_user": "$HETZNER_USER",
    "ssh_key_path": "/notebooks/bot/.ssh/hetzner_key",
    "ssh_port": $SSH_PORT,
    "connection_timeout": 30,
    "transfer_timeout": 3600,
    "validation_enabled": true,
    "backup_enabled": true,
    "auto_restart": false
}
EOF

echo "✅ Configuration saved: $CONFIG_FILE"

# Create environment variables file
ENV_FILE="/notebooks/bot/.env.hetzner"
echo "📝 Creating environment variables..."

cat > "$ENV_FILE" << EOF
# Hetzner Export Configuration
export HETZNER_HOST="$HETZNER_HOST"
export HETZNER_USER="$HETZNER_USER"
export SSH_KEY_PATH="$SSH_KEY_PATH"
export SSH_PORT="$SSH_PORT"
EOF

echo "✅ Environment file created: $ENV_FILE"

# Test remote directory structure
echo "🔍 Checking remote directory structure..."
REMOTE_CHECK=$(ssh -i "$SSH_KEY_PATH" -p "$SSH_PORT" "$HETZNER_USER@$HETZNER_HOST" "
    if [ -d '/opt/trading_bot' ]; then
        echo 'TRADING_BOT_DIR_EXISTS'
    fi
    if [ -f '/opt/trading_bot/bin/system_manager' ]; then
        echo 'SYSTEM_MANAGER_EXISTS'
    fi
    if [ -d '/opt/trading_bot/models' ]; then
        echo 'MODELS_DIR_EXISTS'
    fi
" 2>/dev/null)

if echo "$REMOTE_CHECK" | grep -q "TRADING_BOT_DIR_EXISTS"; then
    echo "✅ Trading bot directory found: /opt/trading_bot"
else
    echo "⚠️  Trading bot directory not found at /opt/trading_bot"
    echo "   Please ensure your trading system is installed at the correct path"
fi

if echo "$REMOTE_CHECK" | grep -q "SYSTEM_MANAGER_EXISTS"; then
    echo "✅ System manager found: /opt/trading_bot/bin/system_manager"
else
    echo "⚠️  System manager not found at /opt/trading_bot/bin/system_manager"
fi

if echo "$REMOTE_CHECK" | grep -q "MODELS_DIR_EXISTS"; then
    echo "✅ Models directory found: /opt/trading_bot/models"
else
    echo "📁 Creating models directory on remote server..."
    ssh -i "$SSH_KEY_PATH" -p "$SSH_PORT" "$HETZNER_USER@$HETZNER_HOST" "mkdir -p /opt/trading_bot/models"
    echo "✅ Models directory created"
fi

# Create convenient export aliases
echo "📝 Creating export aliases..."
ALIAS_FILE="/notebooks/bot/export_shortcuts.sh"

cat > "$ALIAS_FILE" << EOF
#!/bin/bash
# Convenient shortcuts for Hetzner export

# Source environment
source /notebooks/bot/.env.hetzner

# Quick export function
export_models() {
    echo "🚀 Quick Export: Paperspace → Hetzner"
    python3 /notebooks/bot/paperspace_mlops/export_to_hetzner.py \\
        --config /notebooks/bot/paperspace_mlops/hetzner_config.json \\
        \$@
}

# Dry run test
test_export() {
    echo "🧪 Testing export setup..."
    python3 /notebooks/bot/paperspace_mlops/export_to_hetzner.py \\
        --config /notebooks/bot/paperspace_mlops/hetzner_config.json \\
        --dry-run
}

# Export with auto-restart
export_and_restart() {
    echo "🚀 Export with auto-restart..."
    python3 /notebooks/bot/paperspace_mlops/export_to_hetzner.py \\
        --config /notebooks/bot/paperspace_mlops/hetzner_config.json \\
        --auto-restart
}

# Check remote status
check_remote() {
    echo "📊 Checking remote status..."
    ssh -i "\$SSH_KEY_PATH" -p "\$SSH_PORT" "\$HETZNER_USER@\$HETZNER_HOST" "
        cd /opt/trading_bot
        echo '🔍 System Status:'
        if [ -f bin/system_manager ]; then
            ./bin/system_manager status 2>/dev/null || echo 'System manager not running'
        fi
        echo ''
        echo '📁 Model Status:'
        find models/superior -name '*.zip' 2>/dev/null | wc -l | xargs echo 'Superior models:'
        echo ''
        echo '💾 Disk Usage:'
        df -h /opt/trading_bot | tail -1
    "
}

echo "Available commands:"
echo "  export_models      - Export superior models to Hetzner"
echo "  test_export        - Test export setup (dry run)"
echo "  export_and_restart - Export and restart trading system"
echo "  check_remote       - Check remote server status"
EOF

chmod +x "$ALIAS_FILE"
echo "✅ Export shortcuts created: $ALIAS_FILE"

echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
echo "Quick start commands:"
echo "  # Load shortcuts"
echo "  source $ALIAS_FILE"
echo ""
echo "  # Test the setup"
echo "  test_export"
echo ""
echo "  # Export your superior models"
echo "  export_models"
echo ""
echo "  # Or export with auto-restart"
echo "  export_and_restart"
echo ""
echo "  # Check remote status"
echo "  check_remote"
echo ""
echo "📝 Configuration files created:"
echo "   $CONFIG_FILE"
echo "   $ENV_FILE"
echo "   $ALIAS_FILE"
echo ""
echo "🔑 SSH key pair:"
echo "   Private: $SSH_KEY_PATH"
echo "   Public:  $SSH_KEY_PATH.pub"
echo ""
echo "🎯 You're ready to export superior models to Hetzner!"