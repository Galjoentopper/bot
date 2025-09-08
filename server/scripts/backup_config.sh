#!/bin/bash

# Configuration Backup Script
# Self-locate the bot directory (server/scripts -> bot)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$SCRIPT_DIR/backups/config"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=== Configuration Backup ==="
echo "Time: $(date)"
echo "Target: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Change to script directory
cd "$SCRIPT_DIR" || exit 1

# Create backup archive
BACKUP_FILE="$BACKUP_DIR/config_backup_$TIMESTAMP.tar.gz"

echo "Creating backup archive..."

# Backup configuration files
tar -czf "$BACKUP_FILE" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='logs' \
    --exclude='data/binance_*.csv' \
    --exclude='model_packages' \
    --exclude='models' \
    config/ \
    *.yaml \
    *.yml \
    *.json \
    requirements.txt \
    setup.py \
    src/config/ \
    2>/dev/null

if [ $? -eq 0 ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "✅ Backup created: config_backup_$TIMESTAMP.tar.gz ($BACKUP_SIZE)"
else
    echo "❌ Backup failed"
    exit 1
fi

# Clean up old backups (keep last 7 days)
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -name "config_backup_*.tar.gz" -mtime +7 -delete

REMAINING_BACKUPS=$(find "$BACKUP_DIR" -name "config_backup_*.tar.gz" | wc -l)
echo "Remaining backups: $REMAINING_BACKUPS"

# Backup summary
echo ""
echo "=== Backup Summary ==="
echo "Latest backup: config_backup_$TIMESTAMP.tar.gz"
echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"
echo "Location: $BACKUP_DIR"
echo "Total backups: $REMAINING_BACKUPS"
echo "=================================="