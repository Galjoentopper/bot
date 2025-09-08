#!/bin/bash
# Verify Clean Directory Structure
# Checks that all scripts and configurations use relative paths within bot directory

set -e

echo "🔍 Verifying Clean Directory Structure..."
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS_COUNT=0
WARNING_COUNT=0
ERROR_COUNT=0

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((SUCCESS_COUNT++))
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNING_COUNT++))
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((ERROR_COUNT++))
}

echo ""
echo "📁 Checking for scattered directory references..."

# Check for hardcoded /opt/trading_bot paths (excluding comments and documentation)
echo "🔍 Searching for hardcoded paths in scripts..."

SCATTERED_PATHS_FOUND=0

# Check all shell scripts for problematic paths
while IFS= read -r -d '' file; do
    if [[ "$file" == *".sh" ]]; then
        # Look for problematic paths, excluding comments and our acceptable patterns
        PROBLEMATIC_LINES=$(grep -n "/opt/trading_bot" "$file" | \
            grep -v "SCRIPT_DIR.*=/opt/trading_bot/bot" | \
            grep -v "dirname.*BASH_SOURCE" | \
            grep -v "cd.*dirname.*BASH_SOURCE" | \
            grep -v "Self-locate.*bot.*directory" | \
            grep -v "^[[:space:]]*#" | \
            grep -v "echo.*directories.*bot.*level" || true)
        
        if [ -n "$PROBLEMATIC_LINES" ]; then
            log_error "Found scattered path references in $file:"
            echo "$PROBLEMATIC_LINES" | while read -r line; do
                echo "    $line"
            done
            ((SCATTERED_PATHS_FOUND++))
        fi
    fi
done < <(find . -type f -print0)

# Check for /var/log/trading_bot references
VAR_LOG_FOUND=$(grep -r "/var/log/trading_bot" . --include="*.sh" | grep -v "^[[:space:]]*#" | wc -l || echo 0)

if [ "$VAR_LOG_FOUND" -gt 0 ]; then
    log_error "Found /var/log/trading_bot references:"
    grep -r "/var/log/trading_bot" . --include="*.sh" | grep -v "^[[:space:]]*#" || true
else
    log_success "No /var/log/trading_bot references found"
fi

# Check for /etc/trading_bot references
ETC_TRADING_FOUND=$(grep -r "/etc/trading_bot" . --include="*.sh" | grep -v "backup" | grep -v "^[[:space:]]*#" | wc -l || echo 0)

if [ "$ETC_TRADING_FOUND" -gt 0 ]; then
    log_warning "Found /etc/trading_bot references (may be acceptable for system configs):"
    grep -r "/etc/trading_bot" . --include="*.sh" | grep -v "backup" | grep -v "^[[:space:]]*#" || true
fi

echo ""
echo "📂 Checking directory structure..."

# Verify expected directories exist within bot folder
EXPECTED_DIRS=("logs" "data" "models" "scripts" "src" "server" "backups")

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        log_success "Directory exists: $dir/"
    else
        log_warning "Directory missing: $dir/"
    fi
done

echo ""
echo "🔧 Checking script executability..."

# Check that key scripts are executable
KEY_SCRIPTS=("deploy_full_system.sh" "start_system.sh" "stop_system.sh")

for script in "${KEY_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            log_success "Script executable: $script"
        else
            log_warning "Script not executable: $script (run: chmod +x $script)"
        fi
    else
        log_warning "Script missing: $script"
    fi
done

echo ""
echo "🎯 Checking for self-locating script patterns..."

# Check if scripts use proper self-location patterns
GOOD_PATTERN_COUNT=0
SCRIPTS_CHECKED=0

while IFS= read -r -d '' file; do
    if [[ "$file" == *".sh" ]] && [[ "$file" != "./verify_clean_structure.sh" ]]; then
        ((SCRIPTS_CHECKED++))
        
        # Check for good self-location patterns
        if grep -q 'SCRIPT_DIR.*dirname.*BASH_SOURCE\|SCRIPT_DIR="/opt/trading_bot/bot"' "$file"; then
            ((GOOD_PATTERN_COUNT++))
        fi
    fi
done < <(find . -name "*.sh" -type f -print0)

if [ "$GOOD_PATTERN_COUNT" -gt 0 ]; then
    log_success "$GOOD_PATTERN_COUNT/$SCRIPTS_CHECKED scripts use proper directory detection"
else
    log_warning "No scripts found with proper self-location patterns"
fi

echo ""
echo "📊 Summary:"
echo "==========="
echo "✅ Successes: $SUCCESS_COUNT"
echo "⚠️  Warnings:  $WARNING_COUNT"
echo "❌ Errors:    $ERROR_COUNT"

if [ "$ERROR_COUNT" -eq 0 ] && [ "$WARNING_COUNT" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 Perfect! All scripts use clean directory structure.${NC}"
    exit 0
elif [ "$ERROR_COUNT" -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}🔶 Good! Only minor warnings found.${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}🚨 Issues found! Please fix the errors above.${NC}"
    exit 1
fi
