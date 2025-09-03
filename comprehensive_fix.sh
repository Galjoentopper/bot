#!/bin/bash
# Comprehensive Fix for All Hardcoded Paths
# This will fix ALL remaining issues found by quick_clean_check.sh

echo "🔧 Comprehensive Path Fix"
echo "========================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working in: $SCRIPT_DIR"
echo ""

FIXED_COUNT=0

# 1. Fix scripts/init_logs.sh if it exists
if [ -f "scripts/init_logs.sh" ]; then
    echo "🔧 Fixing scripts/init_logs.sh..."
    cp scripts/init_logs.sh scripts/init_logs.sh.backup
    sed -i 's|SCRIPT_DIR="/opt/trading_bot/bot"|# Self-locate the bot directory (scripts -> bot)\nSCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." \&\& pwd)"|g' scripts/init_logs.sh
    echo "✅ Fixed scripts/init_logs.sh"
    FIXED_COUNT=$((FIXED_COUNT + 1))
fi

# 2. Fix server/scripts/rotate_logs.sh
if [ -f "server/scripts/rotate_logs.sh" ]; then
    echo "🔧 Fixing server/scripts/rotate_logs.sh..."
    cp server/scripts/rotate_logs.sh server/scripts/rotate_logs.sh.backup
    sed -i 's|LOG_DIR="/opt/trading_bot/bot/logs"|# Self-locate the bot directory (server/scripts -> bot)\nSCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." \&\& pwd)"\nLOG_DIR="$SCRIPT_DIR/logs"|g' server/scripts/rotate_logs.sh
    sed -i 's|BACKUP_DIR="/opt/trading_bot/bot/backups/logs"|BACKUP_DIR="$SCRIPT_DIR/backups/logs"|g' server/scripts/rotate_logs.sh
    echo "✅ Fixed server/scripts/rotate_logs.sh"
    FIXED_COUNT=$((FIXED_COUNT + 1))
fi

# 3. Fix disk usage references in health_check.sh
if [ -f "server/scripts/health_check.sh" ]; then
    echo "🔧 Fixing server/scripts/health_check.sh disk references..."
    cp server/scripts/health_check.sh server/scripts/health_check.sh.backup
    sed -i 's|df -h /opt/trading_bot|df -h "$SCRIPT_DIR"|g' server/scripts/health_check.sh
    echo "✅ Fixed server/scripts/health_check.sh"
    FIXED_COUNT=$((FIXED_COUNT + 1))
fi

# 4. Fix disk usage references in generate_performance_report.sh
if [ -f "server/scripts/generate_performance_report.sh" ]; then
    echo "🔧 Fixing server/scripts/generate_performance_report.sh disk references..."
    cp server/scripts/generate_performance_report.sh server/scripts/generate_performance_report.sh.backup
    sed -i 's|df -h /opt/trading_bot|df -h "$SCRIPT_DIR"|g' server/scripts/generate_performance_report.sh
    echo "✅ Fixed server/scripts/generate_performance_report.sh"
    FIXED_COUNT=$((FIXED_COUNT + 1))
fi

# 5. Fix any other remaining hardcoded SCRIPT_DIR paths
echo ""
echo "🔍 Checking for any other hardcoded SCRIPT_DIR paths..."

for file in $(find . -name "*.sh" -type f); do
    # Skip our fix scripts and verification scripts
    if [[ "$file" == *"fix_hardcoded_paths.sh"* ]] || [[ "$file" == *"quick_clean_check.sh"* ]] || [[ "$file" == *"verify_clean_structure.sh"* ]] || [[ "$file" == *"comprehensive_fix.sh"* ]]; then
        continue
    fi
    
    # Check if this file has hardcoded SCRIPT_DIR and doesn't use self-location
    if grep -q 'SCRIPT_DIR="/opt/trading_bot/bot"' "$file" && ! grep -q 'dirname.*BASH_SOURCE' "$file"; then
        echo "🔧 Fixing remaining hardcoded path in $file..."
        
        # Create backup
        cp "$file" "$file.backup"
        
        # Determine the relative path from this script to the bot directory
        if [[ "$file" == "./server/scripts/"* ]]; then
            # server/scripts -> bot (go up two levels)
            RELATIVE_PATH="../.."
        elif [[ "$file" == "./scripts/"* ]]; then
            # scripts -> bot (go up one level)  
            RELATIVE_PATH=".."
        else
            # Assume it's in the root of bot directory
            RELATIVE_PATH="."
        fi
        
        # Replace hardcoded path with self-locating pattern
        sed -i "s|SCRIPT_DIR=\"/opt/trading_bot/bot\"|# Self-locate the bot directory\nSCRIPT_DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")/$RELATIVE_PATH\" \&\& pwd)\"|g" "$file"
        
        echo "✅ Fixed $file"
        FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
done

# 6. Make all shell scripts executable
echo ""
echo "🔧 Making all shell scripts executable..."
find . -name "*.sh" -type f -exec chmod +x {} \;
echo "✅ All shell scripts are now executable"

echo ""
echo "🎉 Comprehensive fix complete!"
echo "   Fixed $FIXED_COUNT files"
echo ""
echo "📝 What was fixed:"
echo "   ✅ scripts/init_logs.sh - Self-locating SCRIPT_DIR"
echo "   ✅ server/scripts/rotate_logs.sh - Self-locating paths"
echo "   ✅ server/scripts/health_check.sh - Relative disk usage"
echo "   ✅ server/scripts/generate_performance_report.sh - Relative disk usage"
echo "   ✅ All scripts made executable"
echo ""
echo "🧪 Test the fixes:"
echo "   ./quick_clean_check.sh"
echo "   ./start_system.sh"
