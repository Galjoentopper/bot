#!/bin/bash
# Resource Management and Monitoring Script

set -e

# Configuration
MEMORY_LIMIT_MB=2800  # 75% of 3.7GB system memory
DISK_LIMIT_PERCENT=85
CPU_LIMIT_PERCENT=80
SWAP_LIMIT_MB=1000

LOG_FILE="logs/resource_monitor.log"
ALERT_FILE="logs/resource_alerts.log"

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo -e "$1"
}

# Alert function
send_alert() {
    local level=$1
    local message=$2
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [$level] $message" >> "$ALERT_FILE"
    log_message "${RED}[ALERT-$level] $message${NC}"
}

# Check system resources
check_resources() {
    log_message "${BLUE}[INFO] Starting resource check...${NC}"
    
    # Memory check
    memory_used=$(free -m | awk 'NR==2{printf "%d", $3}')
    memory_percent=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    
    if [ $memory_used -gt $MEMORY_LIMIT_MB ]; then
        send_alert "CRITICAL" "Memory usage ${memory_used}MB exceeds limit ${MEMORY_LIMIT_MB}MB (${memory_percent}%)"
        return 1
    elif [ $memory_used -gt $((MEMORY_LIMIT_MB * 80 / 100)) ]; then
        log_message "${YELLOW}[WARNING] Memory usage ${memory_used}MB approaching limit (${memory_percent}%)${NC}"
    else
        log_message "${GREEN}[OK] Memory usage: ${memory_used}MB (${memory_percent}%)${NC}"
    fi
    
    # Disk space check
    disk_percent=$(df . | awk 'NR==2{print $5}' | sed 's/%//')
    if [ $disk_percent -gt $DISK_LIMIT_PERCENT ]; then
        send_alert "CRITICAL" "Disk usage ${disk_percent}% exceeds limit ${DISK_LIMIT_PERCENT}%"
        return 1
    elif [ $disk_percent -gt $((DISK_LIMIT_PERCENT * 80 / 100)) ]; then
        log_message "${YELLOW}[WARNING] Disk usage ${disk_percent}% approaching limit${NC}"
    else
        log_message "${GREEN}[OK] Disk usage: ${disk_percent}%${NC}"
    fi
    
    # CPU check (5-minute average)
    cpu_percent=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}' | cut -d'.' -f1)
    if [ $cpu_percent -gt $CPU_LIMIT_PERCENT ]; then
        send_alert "WARNING" "CPU usage ${cpu_percent}% exceeds limit ${CPU_LIMIT_PERCENT}%"
    else
        log_message "${GREEN}[OK] CPU usage: ${cpu_percent}%${NC}"
    fi
    
    # Swap check
    swap_used=$(free -m | awk 'NR==3{printf "%d", $3}')
    if [ $swap_used -gt $SWAP_LIMIT_MB ]; then
        send_alert "WARNING" "Swap usage ${swap_used}MB exceeds recommended limit ${SWAP_LIMIT_MB}MB"
    else
        log_message "${GREEN}[OK] Swap usage: ${swap_used}MB${NC}"
    fi
    
    return 0
}

# Check Python processes
check_python_processes() {
    log_message "${BLUE}[INFO] Checking Python processes...${NC}"
    
    # Find trading processes
    trading_pids=$(pgrep -f "enhanced_trader.py" || true)
    telegram_pids=$(pgrep -f "telegram_bot_listener" || true)
    
    if [ -n "$trading_pids" ]; then
        for pid in $trading_pids; do
            memory=$(ps -p $pid -o rss= | awk '{print int($1/1024)}')
            cpu=$(ps -p $pid -o pcpu= | sed 's/^[ \t]*//')
            log_message "${GREEN}[OK] Trading process PID $pid: ${memory}MB RAM, ${cpu}% CPU${NC}"
            
            # Alert if process is using too much memory
            if [ $memory -gt 1500 ]; then
                send_alert "WARNING" "Trading process PID $pid using ${memory}MB RAM (high)"
            fi
        done
    else
        log_message "${YELLOW}[WARNING] No trading processes found${NC}"
    fi
    
    if [ -n "$telegram_pids" ]; then
        for pid in $telegram_pids; do
            memory=$(ps -p $pid -o rss= | awk '{print int($1/1024)}')
            cpu=$(ps -p $pid -o pcpu= | sed 's/^[ \t]*//')
            log_message "${GREEN}[OK] Telegram process PID $pid: ${memory}MB RAM, ${cpu}% CPU${NC}"
        done
    else
        log_message "${YELLOW}[WARNING] No telegram processes found${NC}"
    fi
}

# Cleanup function
cleanup_resources() {
    log_message "${BLUE}[INFO] Starting resource cleanup...${NC}"
    
    # Clean Python cache
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Rotate large log files
    if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -gt 10485760 ]; then  # 10MB
        mv "$LOG_FILE" "${LOG_FILE}.old"
        touch "$LOG_FILE"
        log_message "${GREEN}[OK] Rotated resource monitor log${NC}"
    fi
    
    # Clean temporary files
    find /tmp -name "*trading*" -mtime +1 -delete 2>/dev/null || true
    
    log_message "${GREEN}[OK] Resource cleanup completed${NC}"
}

# Set Python memory limits
set_python_limits() {
    export PYTHONHASHSEED=0  # Consistent hash seeds
    export PYTHONOPTIMIZE=1  # Enable optimizations
    export MALLOC_TRIM_THRESHOLD_=100000  # More aggressive memory trimming
    export MALLOC_MMAP_THRESHOLD_=100000   # Use mmap for large allocations
    
    log_message "${GREEN}[OK] Python memory limits configured${NC}"
}

# Graceful degradation
graceful_degradation() {
    log_message "${YELLOW}[WARNING] Entering graceful degradation mode${NC}"
    
    # Reduce model ensemble if memory is critical
    if [ -f "training_config.yaml" ]; then
        # Create backup
        cp training_config.yaml training_config.yaml.backup
        
        # Temporarily disable model warmup and caching
        sed -i 's/model_warmup: true/model_warmup: false/g' training_config.yaml 2>/dev/null || true
        sed -i 's/cache_predictions: true/cache_predictions: false/g' training_config.yaml 2>/dev/null || true
        
        log_message "${YELLOW}[WARNING] Disabled model warmup and caching to conserve memory${NC}"
    fi
}

# Main execution
main() {
    mkdir -p logs
    
    case "${1:-check}" in
        "check")
            if ! check_resources; then
                graceful_degradation
            fi
            check_python_processes
            ;;
        "cleanup")
            cleanup_resources
            ;;
        "limits")
            set_python_limits
            ;;
        "monitor")
            while true; do
                check_resources
                check_python_processes
                sleep 300  # Check every 5 minutes
            done
            ;;
        "status")
            echo "Resource Monitor Status:"
            echo "Memory Limit: ${MEMORY_LIMIT_MB}MB"
            echo "Disk Limit: ${DISK_LIMIT_PERCENT}%"
            echo "CPU Limit: ${CPU_LIMIT_PERCENT}%"
            echo "Log File: $LOG_FILE"
            echo "Alert File: $ALERT_FILE"
            ;;
        *)
            echo "Usage: $0 {check|cleanup|limits|monitor|status}"
            echo "  check   - Run resource checks once"
            echo "  cleanup - Clean up temporary files and caches" 
            echo "  limits  - Set Python memory limits"
            echo "  monitor - Continuous monitoring (runs every 5 minutes)"
            echo "  status  - Show current configuration"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"