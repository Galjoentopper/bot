#!/bin/bash
# Configuration backup script

BACKUP_DIR="/opt/trading_bot/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

mkdir -p "$BACKUP_DIR"

# Create backup
tar -czf "$BACKUP_FILE" \
    --exclude="logs/*" \
    --exclude="data/*" \
    --exclude="models/*" \
    --exclude="venv" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    /opt/trading_bot/ \
    /etc/trading_bot/ \
    /etc/systemd/system/trading-bot.service \
    /etc/cron.d/trading_bot_monitor

echo "Backup created: $BACKUP_FILE"
echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"