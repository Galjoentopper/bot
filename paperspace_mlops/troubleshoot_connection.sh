#!/bin/bash
"""
Hetzner Connection Troubleshooting
==================================

Diagnoses and fixes common SSH connection issues.
"""

set -e

echo "🔍 HETZNER CONNECTION TROUBLESHOOTING"
echo "====================================="

HETZNER_HOST="49.12.107.148"
HETZNER_USER="trading_bot"
SSH_PORT="22"

echo "Target: $HETZNER_USER@$HETZNER_HOST:$SSH_PORT"
echo ""

# Test 1: Basic connectivity
echo "🌐 Test 1: Basic Network Connectivity"
echo "======================================"
echo "Testing if host is reachable..."

if ping -c 3 "$HETZNER_HOST" >/dev/null 2>&1; then
    echo "✅ Host is reachable via ping"
else
    echo "❌ Host unreachable via ping"
    echo "   This could indicate:"
    echo "   - Server is down"
    echo "   - Firewall blocking ICMP"
    echo "   - Wrong IP address"
fi

echo ""

# Test 2: Port connectivity
echo "🔌 Test 2: SSH Port Connectivity"
echo "================================="
echo "Testing SSH port $SSH_PORT..."

if timeout 10 bash -c "</dev/tcp/$HETZNER_HOST/$SSH_PORT" >/dev/null 2>&1; then
    echo "✅ Port $SSH_PORT is open and reachable"
else
    echo "❌ Port $SSH_PORT is not reachable"
    echo "   Possible issues:"
    echo "   - SSH service not running"
    echo "   - Firewall blocking port $SSH_PORT"
    echo "   - SSH running on different port"
    echo ""
    echo "   Common SSH ports to try:"
    for port in 2222 2200 22022; do
        echo -n "     Testing port $port... "
        if timeout 5 bash -c "</dev/tcp/$HETZNER_HOST/$port" >/dev/null 2>&1; then
            echo "✅ Port $port is open!"
            SSH_PORT=$port
            break
        else
            echo "❌ Closed"
        fi
    done
fi

echo ""

# Test 3: SSH service detection
echo "🔍 Test 3: SSH Service Detection"
echo "================================="
echo "Attempting to connect and identify SSH service..."

ssh_output=$(timeout 10 nc -v "$HETZNER_HOST" "$SSH_PORT" 2>&1 | head -1 || echo "Connection failed")
echo "SSH banner: $ssh_output"

if echo "$ssh_output" | grep -q "SSH"; then
    echo "✅ SSH service detected"
else
    echo "❌ No SSH service detected on port $SSH_PORT"
fi

echo ""

# Test 4: Authentication methods
echo "🔐 Test 4: SSH Authentication Methods"
echo "======================================"
echo "Checking available authentication methods..."

auth_methods=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$SSH_PORT" "$HETZNER_USER@$HETZNER_HOST" 2>&1 | grep -o "Permission denied.*" || echo "Connection failed")
echo "Auth response: $auth_methods"

echo ""

# Test 5: Different usernames
echo "👤 Test 5: Common Username Testing"
echo "==================================="
echo "Testing common usernames..."

common_users=("root" "ubuntu" "debian" "admin" "user" "$HETZNER_USER")

for user in "${common_users[@]}"; do
    echo -n "   Testing user '$user'... "

    # Try to connect with a very short timeout
    result=$(timeout 5 ssh -o BatchMode=yes -o ConnectTimeout=3 -p "$SSH_PORT" "$user@$HETZNER_HOST" "echo 'success'" 2>&1 || echo "failed")

    if echo "$result" | grep -q "success"; then
        echo "✅ SUCCESS!"
        WORKING_USER="$user"
        break
    elif echo "$result" | grep -q "Permission denied"; then
        echo "🔑 User exists, needs authentication"
    else
        echo "❌ Failed"
    fi
done

echo ""

# Recommendations
echo "🎯 RECOMMENDATIONS"
echo "=================="
echo ""

if timeout 5 bash -c "</dev/tcp/$HETZNER_HOST/$SSH_PORT" >/dev/null 2>&1; then
    echo "✅ Network connection is working"
    echo ""
    echo "🔧 Next steps to fix authentication:"
    echo ""
    echo "Option 1 - Add SSH key manually:"
    echo "   1. Log into your Hetzner server console (web interface)"
    echo "   2. Create/edit ~/.ssh/authorized_keys"
    echo "   3. Add this public key:"
    echo ""
    cat ~/.ssh/hetzner_key.pub
    echo ""
    echo "Option 2 - Use password authentication temporarily:"
    echo "   ssh-copy-id -i ~/.ssh/hetzner_key.pub -p $SSH_PORT $HETZNER_USER@$HETZNER_HOST"
    echo "   (You'll be prompted for the password)"
    echo ""
    echo "Option 3 - Try different username:"
    echo "   The user might be 'root' instead of 'trading_bot'"
    echo "   ssh-copy-id -i ~/.ssh/hetzner_key.pub -p $SSH_PORT root@$HETZNER_HOST"
    echo ""
else
    echo "❌ Network connection failed"
    echo ""
    echo "🔧 Things to check:"
    echo "   1. Is the server IP correct? ($HETZNER_HOST)"
    echo "   2. Is the server running?"
    echo "   3. Is SSH enabled on the server?"
    echo "   4. Are there firewall rules blocking connections?"
    echo "   5. Is SSH running on a different port?"
    echo ""
    echo "🌐 Quick fixes to try:"
    echo "   1. Check Hetzner console for server status"
    echo "   2. Verify the server IP in Hetzner dashboard"
    echo "   3. Check if SSH is enabled in server settings"
    echo "   4. Try connecting from Hetzner web console first"
fi

echo ""
echo "📞 SUPPORT COMMANDS"
echo "=================="
echo ""
echo "Test specific port:"
echo "   nc -v $HETZNER_HOST [PORT]"
echo ""
echo "Test with different user:"
echo "   ssh -p $SSH_PORT [USERNAME]@$HETZNER_HOST"
echo ""
echo "Copy key with password:"
echo "   ssh-copy-id -i ~/.ssh/hetzner_key.pub -p $SSH_PORT [USERNAME]@$HETZNER_HOST"
echo ""
echo "Manual key addition:"
echo "   1. Login via Hetzner console"
echo "   2. mkdir -p ~/.ssh"
echo "   3. echo '[PUBLIC_KEY]' >> ~/.ssh/authorized_keys"
echo "   4. chmod 600 ~/.ssh/authorized_keys"
echo ""

# Save working configuration if found
if [ -n "$WORKING_USER" ]; then
    echo "💾 Found working user: $WORKING_USER"
    echo "Updating configuration..."

    # Update config file if it exists
    if [ -f "hetzner_config.json" ]; then
        sed -i "s/\"hetzner_user\": \".*\"/\"hetzner_user\": \"$WORKING_USER\"/" hetzner_config.json
        echo "✅ Configuration updated with working username"
    fi
fi

echo ""
echo "🎯 READY FOR NEXT STEPS"
echo "======================="
echo ""
echo "Once SSH access is working:"
echo "   1. Test connection: ssh -i ~/.ssh/hetzner_key -p $SSH_PORT $HETZNER_USER@$HETZNER_HOST"
echo "   2. Re-run setup: ./setup_hetzner_export.sh"
echo "   3. Export models: ./quick_export.sh"
echo ""