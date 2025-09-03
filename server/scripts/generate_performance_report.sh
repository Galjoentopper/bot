#!/bin/bash

# Generate Performance Report Script
# Self-locate the bot directory (server/scripts -> bot)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
METRICS_FILE="$LOG_DIR/performance_metrics.json"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Check if trading script is running and get basic metrics
if pgrep -f "enhanced_trader.py" > /dev/null; then
    TRADER_PID=$(pgrep -f "enhanced_trader.py")
    CPU_USAGE=$(ps -p $TRADER_PID -o pcpu= 2>/dev/null | tr -d ' ')
    MEM_USAGE=$(ps -p $TRADER_PID -o pmem= 2>/dev/null | tr -d ' ')
    
    # Get process start time
    START_TIME=$(ps -p $TRADER_PID -o lstart= 2>/dev/null)
    
    # Generate basic performance metrics JSON
    cat > "$METRICS_FILE" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "running",
    "process": {
        "pid": $TRADER_PID,
        "cpu_usage": "${CPU_USAGE}%",
        "memory_usage": "${MEM_USAGE}%",
        "start_time": "$START_TIME"
    },
    "system": {
        "uptime": "$(uptime -p)",
        "load_average": "$(uptime | awk '{print $(NF-2)}' | tr -d ',')",
        "memory_total": "$(free -h | grep Mem | awk '{print $2}')",
        "memory_used": "$(free -h | grep Mem | awk '{print $3}')",
        "disk_usage": "$(df -h /opt/trading_bot | tail -1 | awk '{print $5}')"
    },
    "trading": {
        "log_files_count": $(find "$LOG_DIR" -name "*.log" -type f | wc -l),
        "last_activity": "$(find "$LOG_DIR" -name "*.log" -type f -exec stat -c %Y {} \; 2>/dev/null | sort -n | tail -1 | xargs -I{} date -d @{} 2>/dev/null || echo 'N/A')"
    }
}
EOF
else
    # Generate metrics for stopped process
    cat > "$METRICS_FILE" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "stopped",
    "system": {
        "uptime": "$(uptime -p)",
        "load_average": "$(uptime | awk '{print $(NF-2)}' | tr -d ',')",
        "memory_total": "$(free -h | grep Mem | awk '{print $2}')",
        "memory_used": "$(free -h | grep Mem | awk '{print $3}')",
        "disk_usage": "$(df -h /opt/trading_bot | tail -1 | awk '{print $5}')"
    }
}
EOF
fi

# Create a simple trader daemon log if it doesn't exist
DAEMON_LOG="$LOG_DIR/trader_daemon.log"
if [ ! -f "$DAEMON_LOG" ]; then
    echo "$(date): Trading daemon log initialized" > "$DAEMON_LOG"
fi

# Append current status to daemon log
echo "$(date): Performance report generated - Status: $(cat "$METRICS_FILE" | grep '"status"' | cut -d'"' -f4)" >> "$DAEMON_LOG"

# Keep daemon log size manageable (last 1000 lines)
if [ -f "$DAEMON_LOG" ]; then
    tail -1000 "$DAEMON_LOG" > "${DAEMON_LOG}.tmp" && mv "${DAEMON_LOG}.tmp" "$DAEMON_LOG"
fi