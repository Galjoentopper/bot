#!/bin/bash
# Quick verification for clean structure
# Run this on your Ubuntu server

echo "🔍 Quick Clean Structure Check"
echo "=============================="

cd /opt/trading_bot/bot

echo "✅ Checking for hardcoded /opt/trading_bot paths (excluding acceptable ones)..."

# Count problematic hardcoded paths
PROBLEM_COUNT=0

echo "📁 Scanning shell scripts..."
find . -name "*.sh" -type f | while read -r file; do
    # Look for problematic hardcoded paths
    if grep -q 'SCRIPT_DIR="/opt/trading_bot' "$file" && ! grep -q 'dirname.*BASH_SOURCE' "$file"; then
        echo "⚠️  $file still has hardcoded path"
        PROBLEM_COUNT=$((PROBLEM_COUNT + 1))
    fi
done

echo ""
echo "📊 Quick Summary:"
echo "=================="

# Check if key directories exist
if [ -d "logs" ] && [ -d "data" ] && [ -d "models" ] && [ -d "scripts" ]; then
    echo "✅ Core directories exist within bot folder"
else
    echo "⚠️  Some core directories missing"
fi

# Check if scripts are executable
if [ -x "deploy_full_system.sh" ]; then
    echo "✅ Main deployment script is executable"
else
    echo "⚠️  deploy_full_system.sh not executable"
fi

echo ""
echo "🎯 To fix any remaining issues:"
echo "1. Make sure all scripts use: SCRIPT_DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")/..\" && pwd)\""
echo "2. Run: chmod +x *.sh"
echo "3. Test: ./deploy_full_system.sh"
echo ""
echo "✨ Your directory structure should now be clean!"
