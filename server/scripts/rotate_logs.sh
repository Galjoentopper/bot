#!/bin/bash

# Log Rotation Script
# Self-locate the bot directory (server/scripts -> bot)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
BACKUP_DIR="$SCRIPT_DIR/backups/logs"
MAX_LOG_SIZE="100M"
MAX_LOG_FILES=10

echo "=== Log Rotation Started ==="
echo "Time: $(date)"

# Create directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$BACKUP_DIR"

# Function to rotate a log file
rotate_log() {
    local log_file="$1"
    local log_name=$(basename "$log_file")
    
    if [ -f "$log_file" ]; then
        local file_size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo 0)
        local max_size_bytes=$(echo "$MAX_LOG_SIZE" | sed 's/M/*1024*1024/' | bc)
        
        if [ "$file_size" -gt "$max_size_bytes" ]; then
            echo "Rotating $log_name (size: $(du -h "$log_file" | cut -f1))"
            
            # Compress and move to backup
            timestamp=$(date +"%Y%m%d_%H%M%S")
            gzip -c "$log_file" > "$BACKUP_DIR/${log_name}_${timestamp}.gz"
            
            # Truncate original file
            > "$log_file"
            
            echo "  → Archived to ${log_name}_${timestamp}.gz"
        fi
    fi
}

# Rotate specific log files
echo "Checking log files for rotation..."

# Rotate trading logs
for log_file in "$LOG_DIR"/*.log; do
    [ -f "$log_file" ] && rotate_log "$log_file"
done

# Rotate daemon log
[ -f "$LOG_DIR/trader_daemon.log" ] && rotate_log "$LOG_DIR/trader_daemon.log"

# Clean up old backup files (keep only MAX_LOG_FILES)
echo "Cleaning up old backups..."
cd "$BACKUP_DIR" || exit 1

for log_type in $(ls *.gz 2>/dev/null | sed 's/_[0-9]*_[0-9]*.gz$//' | sort -u); do
    file_count=$(ls ${log_type}_*.gz 2>/dev/null | wc -l)
    if [ "$file_count" -gt "$MAX_LOG_FILES" ]; then
        excess=$((file_count - MAX_LOG_FILES))
        echo "Removing $excess old ${log_type} backup(s)"
        ls -t ${log_type}_*.gz | tail -n "$excess" | xargs rm -f
    fi
done

# Create performance summary
echo ""
echo "=== Log Summary ==="
echo "Active logs: $(find "$LOG_DIR" -name "*.log" -type f | wc -l)"
echo "Backup archives: $(find "$BACKUP_DIR" -name "*.gz" -type f | wc -l)"
echo "Total log disk usage: $(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)"
echo "Total backup disk usage: $(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)"

echo "=== Log Rotation Completed ==="