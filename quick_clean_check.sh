#!/bin/bash
# Quick verification for clean structure
# Run this on your Ubuntu server

echo "🔍 Quick Clean Structure Check"
echo "=============================="

cd /opt/trading_bot/bot

echo "✅ Checking for hardcoded /opt/trading_bot paths (excluding acceptable ones)..."

echo "📁 Scanning all shell scripts for problematic patterns..."

# Find all shell scripts and check each one
FOUND_ISSUES=false

for file in $(find . -name "*.sh" -type f); do
    # Skip our verification scripts
    if [[ "$file" == *"verify_clean_structure.sh"* ]] || [[ "$file" == *"quick_clean_check.sh"* ]]; then
        continue
    fi
    
    # Check for problematic hardcoded paths
    HARDCODED=$(grep -n 'SCRIPT_DIR="/opt/trading_bot' "$file" 2>/dev/null | grep -v 'dirname.*BASH_SOURCE')
    
    if [ -n "$HARDCODED" ]; then
        echo "⚠️  $file has hardcoded path:"
        echo "    $HARDCODED"
        FOUND_ISSUES=true
    fi
done

if [ "$FOUND_ISSUES" = false ]; then
    echo "✅ No hardcoded SCRIPT_DIR paths found!"
fi

echo ""
echo "🔍 Checking for other /opt/trading_bot references..."

# Check for other hardcoded references (excluding acceptable patterns)
OTHER_REFS=$(find . -name "*.sh" -type f -exec grep -l "/opt/trading_bot" {} \; | while read -r file; do
    if [[ "$file" == *"verify_clean_structure.sh"* ]] || [[ "$file" == *"quick_clean_check.sh"* ]]; then
        continue
    fi
    
    # Look for references that aren't in comments or acceptable patterns
    BAD_REFS=$(grep -n "/opt/trading_bot" "$file" | grep -v "^[[:space:]]*#" | grep -v "dirname.*BASH_SOURCE" | grep -v "Self-locate" | grep -v "echo.*level")
    
    if [ -n "$BAD_REFS" ]; then
        echo "⚠️  $file:"
        echo "$BAD_REFS" | head -3
    fi
done)

if [ -z "$OTHER_REFS" ]; then
    echo "✅ No other problematic /opt/trading_bot references found!"
else
    echo "$OTHER_REFS"
fi

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
