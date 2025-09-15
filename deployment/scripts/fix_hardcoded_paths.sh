#!/bin/bash
# Fix remaining hardcoded paths automatically
# Run this on your Ubuntu server

echo "🔧 Fixing Remaining Hardcoded Paths"
echo "===================================="

# Self-locate to current directory (should be /opt/trading_bot/bot)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find and fix any remaining hardcoded SCRIPT_DIR paths
echo "🔍 Searching for hardcoded SCRIPT_DIR paths..."

FIXED_COUNT=0

for file in $(find . -name "*.sh" -type f); do
    # Skip our verification scripts
    if [[ "$file" == *"verify_clean_structure.sh"* ]] || [[ "$file" == *"quick_clean_check.sh"* ]] || [[ "$file" == *"fix_hardcoded_paths.sh"* ]]; then
        continue
    fi

    # Check if this file has hardcoded SCRIPT_DIR and doesn't use self-location
    if grep -q 'SCRIPT_DIR="/opt/trading_bot' "$file" && ! grep -q 'dirname.*BASH_SOURCE' "$file"; then
        echo "🔧 Fixing $file..."

        # Determine the relative path from this script to the bot directory
        SCRIPT_RELATIVE_PATH=""

        if [[ "$file" == "./server/scripts/"* ]]; then
            # server/scripts -> bot (go up two levels)
            SCRIPT_RELATIVE_PATH="../.."
        elif [[ "$file" == "./scripts/"* ]]; then
            # scripts -> bot (go up one level)
            SCRIPT_RELATIVE_PATH=".."
        else
            # Assume it's in the root of bot directory
            SCRIPT_RELATIVE_PATH="."
        fi

        # Create backup
        cp "$file" "$file.backup"

        # Replace hardcoded path with self-locating pattern
        sed -i "s|SCRIPT_DIR=\"/opt/trading_bot/bot\"|# Self-locate the bot directory\nSCRIPT_DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")/$SCRIPT_RELATIVE_PATH\" \&\& pwd)\"|g" "$file"

        echo "✅ Fixed $file (backup saved as $file.backup)"
        FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
done

if [ "$FIXED_COUNT" -eq 0 ]; then
    echo "✅ No hardcoded paths found to fix!"
else
    echo ""
    echo "🎉 Fixed $FIXED_COUNT files!"
    echo ""
    echo "📝 Changes made:"
    echo "   - Replaced hardcoded SCRIPT_DIR paths with self-locating patterns"
    echo "   - Created .backup files for safety"
    echo ""
    echo "🧪 Test the fixes:"
    echo "   ./quick_clean_check.sh"
fi

echo ""
echo "✨ Hardcoded path fix complete!"
