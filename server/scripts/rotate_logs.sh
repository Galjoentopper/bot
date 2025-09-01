#!/bin/bash
# Log rotation script

LOG_DIR="/var/log/trading_bot"
SCRIPT_LOGS="/opt/trading_bot/logs"

# Rotate system logs
find "$LOG_DIR" -name "*.log" -mtime +7 -exec gzip {} \;
find "$LOG_DIR" -name "*.log.gz" -mtime +30 -delete

# Rotate application logs
find "$SCRIPT_LOGS" -name "*.log" -mtime +7 -exec gzip {} \;
find "$SCRIPT_LOGS" -name "*.log.gz" -mtime +30 -delete

# Clean old performance data
find "$SCRIPT_LOGS" -name "performance_metrics.json" -mtime +30 -delete

# Clean old trade reports (keep last 90 days)
find "$SCRIPT_LOGS" -name "trades_report_*.csv" -mtime +90 -delete

echo "Log rotation completed at $(date)"