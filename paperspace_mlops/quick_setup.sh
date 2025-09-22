#!/bin/bash
# Quick Setup for Known Hetzner Configuration
# ==========================================

set -e

echo "🔧 Quick Setup for Hetzner Export"
echo "=================================="

# Known configuration
HETZNER_HOST="49.12.107.148"
HETZNER_USER="trading_bot"
SSH_PORT="22"

echo "Configuration:"
echo "  Host: $HETZNER_HOST"
echo "  User: $HETZNER_USER"
echo "  Port: $SSH_PORT"
echo ""

# Create SSH key in accessible location
SSH_KEY_PATH="/notebooks/bot/.ssh/hetzner_key"
mkdir -p "/notebooks/bot/.ssh"

if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "🔐 Creating SSH key pair in /notebooks/bot/.ssh/..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "paperspace-hetzner-export"
    chmod 700 "/notebooks/bot/.ssh"
    chmod 600 "$SSH_KEY_PATH"
    chmod 644 "$SSH_KEY_PATH.pub"
    echo "✅ SSH key created: $SSH_KEY_PATH"
else
    echo "✅ SSH key already exists: $SSH_KEY_PATH"
fi

# Create configuration file
CONFIG_FILE="/notebooks/bot/paperspace_mlops/hetzner_config.json"
echo "📝 Creating configuration file..."

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

# Create environment file
ENV_FILE="/notebooks/bot/.env.hetzner"
echo "📝 Creating environment file..."

cat > "$ENV_FILE" << EOF
# Hetzner Export Configuration
export HETZNER_HOST="$HETZNER_HOST"
export HETZNER_USER="$HETZNER_USER"
export SSH_KEY_PATH="/notebooks/bot/.ssh/hetzner_key"
export SSH_PORT="$SSH_PORT"
EOF

echo "✅ Environment file created: $ENV_FILE"

# Show the public key for manual addition
echo ""
echo "🔑 SSH PUBLIC KEY"
echo "=================="
echo "You need to add this public key to your Hetzner server:"
echo ""
cat "$SSH_KEY_PATH.pub"
echo ""
echo "📋 MANUAL STEPS NEEDED:"
echo "======================="
echo ""
echo "1. Copy the public key above"
echo "2. SSH into your Hetzner server: ssh $HETZNER_USER@$HETZNER_HOST"
echo "3. Create the SSH directory: mkdir -p ~/.ssh"
echo "4. Add the key: echo '[PASTE_PUBLIC_KEY_HERE]' >> ~/.ssh/authorized_keys"
echo "5. Set permissions: chmod 600 ~/.ssh/authorized_keys"
echo ""
echo "OR use Hetzner's web console to add the SSH key"
echo ""
echo "🧪 TEST CONNECTION:"
echo "==================="
echo "After adding the key, test with:"
echo "  ssh -i $SSH_KEY_PATH $HETZNER_USER@$HETZNER_HOST"
echo ""
echo "✅ THEN RUN EXPORT:"
echo "==================="
echo "  ./quick_export.sh"
echo ""