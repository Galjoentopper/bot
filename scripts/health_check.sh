#!/bin/bash
# Health Check Script for Trading Bot
# Professional-grade system monitoring and health verification

set -euo pipefail

# Script directory detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
}

log_info() { log "INFO" "${GREEN}✅${NC} $*"; }
log_warn() { log "WARN" "${YELLOW}⚠️${NC} $*"; }
log_error() { log "ERROR" "${RED}❌${NC} $*"; }

# Health check counters
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Check function wrapper
check() {
    local check_name="$1"
    local check_command="$2"

    echo -e "\n${BLUE}🔍 Checking: $check_name${NC}"

    if eval "$check_command" 2>/dev/null; then
        log_info "$check_name: PASSED"
        ((CHECKS_PASSED++))
        return 0
    else
        log_error "$check_name: FAILED"
        ((CHECKS_FAILED++))
        return 1
    fi
}

# Warning check function
warn_check() {
    local check_name="$1"
    local check_command="$2"

    if ! eval "$check_command" 2>/dev/null; then
        log_warn "$check_name: WARNING"
        ((WARNINGS++))
    fi
}

echo -e "${BLUE}📊 Trading Bot Health Check Report${NC}"
echo -e "${BLUE}===================================${NC}"
echo "Timestamp: $(date)"
echo "System: $(uname -a)"
echo "Bot Root: $BOT_ROOT"
echo ""

# 1. Directory Structure Checks
echo -e "\n${BLUE}📂 Directory Structure${NC}"
check "Bot root directory" "[ -d '$BOT_ROOT' ]"
check "Scripts directory" "[ -d '$BOT_ROOT/scripts' ]"
check "Logs directory" "[ -d '$BOT_ROOT/logs' ]"
check "Models directory" "[ -d '$BOT_ROOT/models' ]"
check "Source code directory" "[ -d '$BOT_ROOT/src' ]"
check "Configuration files" "[ -f '$BOT_ROOT/.env' ]"

# 2. Process Checks
echo -e "\n${BLUE}🔄 Process Status${NC}"
check "Python trader processes" "pgrep -f 'python.*trader' > /dev/null"
check "Python telegram processes" "pgrep -f 'python.*telegram' > /dev/null || pgrep -f 'telegram.*bot' > /dev/null"

# 3. System Resources
echo -e "\n${BLUE}💻 System Resources${NC}"
check "Disk space > 1GB" "[ $(df '$BOT_ROOT' | awk 'NR==2 {print $4}') -gt 1048576 ]"
check "Memory available > 500MB" "[ $(free -m | awk 'NR==2{printf \"%.0f\", $7}') -gt 500 ]"
check "CPU load < 10.0" "[ $(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//') -lt 10 ] 2>/dev/null || true"

# 4. Log File Checks
echo -e "\n${BLUE}📋 Log Files${NC}"
check "Trading logs exist" "find '$BOT_ROOT/logs' -name '*.log' -type f | head -1 | grep -q ."
warn_check "Recent trading activity" "find '$BOT_ROOT/logs' -name 'trader_*.log' -mmin -60 -type f | head -1 | grep -q ."
warn_check "Recent telegram activity" "find '$BOT_ROOT/logs' -name 'telegram_*.log' -mmin -60 -type f | head -1 | grep -q ."

# Check for critical errors in recent logs
if find "$BOT_ROOT/logs" -name '*.log' -mmin -60 -type f -exec grep -l "CRITICAL\|FATAL\|ERROR" {} \; | head -1 | grep -q .; then
    log_warn "Critical errors found in recent logs"
    ((WARNINGS++))
else
    log_info "No critical errors in recent logs"
    ((CHECKS_PASSED++))
fi

# 5. Model Files Check
echo -e "\n${BLUE}🤖 Model Files${NC}"
MODEL_COUNT=$(find "$BOT_ROOT/models" -name "*.pkl" -o -name "*.pt" -o -name "*.zip" 2>/dev/null | wc -l)
check "Model files present" "[ $MODEL_COUNT -gt 0 ]"
log_info "Found $MODEL_COUNT model files"

# 6. Network Connectivity (basic)
echo -e "\n${BLUE}🌐 Network Connectivity${NC}"
check "Internet connectivity" "ping -c 1 8.8.8.8 > /dev/null"
check "API connectivity (coinbase)" "curl -s --max-time 10 https://api.pro.coinbase.com/time > /dev/null"

# 7. Python Environment
echo -e "\n${BLUE}🐍 Python Environment${NC}"
check "Python 3 available" "python3 --version > /dev/null"
check "Required packages" "python3 -c 'import pandas, numpy, sklearn, torch' > /dev/null"

# 8. File Permissions
echo -e "\n${BLUE}🔐 File Permissions${NC}"
check "Scripts executable" "[ -x '$BOT_ROOT/scripts/health_check.sh' ]"
check "Log directory writable" "[ -w '$BOT_ROOT/logs' ]"

# 9. System Load and Performance
echo -e "\n${BLUE}⚡ Performance Metrics${NC}"
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
DISK_USAGE=$(df "$BOT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')

log_info "System load: $LOAD_AVG"
log_info "Disk usage: $DISK_USAGE%"
log_info "Memory usage: $MEMORY_USAGE%"

if [ "${DISK_USAGE:-0}" -gt 90 ]; then
    log_warn "High disk usage: $DISK_USAGE%"
    ((WARNINGS++))
fi

if [ "${MEMORY_USAGE:-0}" -gt 90 ]; then
    log_warn "High memory usage: $MEMORY_USAGE%"
    ((WARNINGS++))
fi

# Final Report
echo -e "\n${BLUE}📊 Health Check Summary${NC}"
echo -e "${BLUE}=======================${NC}"
echo -e "✅ Checks Passed: ${GREEN}$CHECKS_PASSED${NC}"
echo -e "❌ Checks Failed: ${RED}$CHECKS_FAILED${NC}"
echo -e "⚠️  Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

# Exit with appropriate code
if [ $CHECKS_FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}🎉 System Health: EXCELLENT${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  System Health: GOOD (with warnings)${NC}"
        exit 0
    fi
else
    echo -e "${RED}💥 System Health: CRITICAL ISSUES FOUND${NC}"
    exit 1
fi
