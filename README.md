# Hetzner Server Deployment Scripts

This directory contains all the necessary scripts and configuration files for deploying the trading system to a Hetzner server.

## Directory Structure

```
server/
├── scripts/                    # Executable scripts
│   ├── tmux_manager.sh        # Main tmux session manager
│   ├── health_check.sh        # System health monitoring
│   ├── deploy_trading.sh      # Deployment script
│   ├── generate_performance_report.sh
│   ├── rotate_logs.sh
│   ├── backup_config.sh
│   └── start_monitoring.sh
├── systemd/                    # Systemd service files
│   └── trading-bot.service
├── cron/                       # Cron job configurations
│   └── trading_bot_monitor
└── src/notifier/              # Enhanced notification system
    └── enhanced_telegram.py
```

## Deployment Steps

### 1. Upload to Server
```bash
# Copy the entire server directory to your Hetzner server
scp -r server/ trader@YOUR_SERVER_IP:/opt/trading_bot/
```

### 2. Set Permissions
```bash
# On the server as trader user
cd /opt/trading_bot
chmod +x scripts/*.sh
```

### 3. Install Systemd Service
```bash
# Copy systemd service
sudo cp systemd/trading-bot.service /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
```

### 4. Install Cron Jobs
```bash
# Copy cron configuration
sudo cp cron/trading_bot_monitor /etc/cron.d/

# Restart cron service
sudo systemctl restart cron
```

### 5. Update Enhanced Telegram Notifier
```bash
# Copy to your project structure
cp src/notifier/enhanced_telegram.py /opt/trading_bot/src/notifier/
```

## Usage

### Basic Operations
```bash
# Start the system
sudo systemctl start trading-bot

# Check status
/opt/trading_bot/scripts/tmux_manager.sh status

# View logs
/opt/trading_bot/scripts/tmux_manager.sh logs

# Attach to session
/opt/trading_bot/scripts/tmux_manager.sh attach

# Stop the system
sudo systemctl stop trading-bot
```

### Discover Available Models
```bash
# Print available symbols and model types (JSON) and exit
python scripts/enhanced_trader.py --config training_config.yaml --show-available
```

### Telegram Commands
Send these commands to your Telegram bot:
- `/status` - System status
- `/start` - Start trading
- `/stop` - Stop trading
- `/performance` - Performance metrics
- `/health` - System health
- `/balance` - Current balance
- `/trades` - Recent trades
- `/logs` - Recent logs

### Monitoring
```bash
# Start monitoring dashboard
/opt/trading_bot/scripts/start_monitoring.sh

# View system logs
sudo journalctl -u trading-bot -f

# Check cron jobs
sudo crontab -l -u trader
```

## Configuration

### Environment Variables
Make sure your `/etc/trading_bot/.env` file contains:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
BITVAVO_API_KEY=your_api_key
BITVAVO_API_SECRET=your_api_secret
TRADING_TIMEOUT=300
```

### Trading Configuration
Ensure your `training_config.yaml` includes the required symbols:
```yaml
symbols:
  - BTCEUR
  - ETHEUR
  - ADAEUR
  - DOTEUR
  - LINKEUR
```

Optional portfolio optimization tuning:
```yaml
portfolio_optimization:
  correlation_threshold: 0.8      # start scaling down above this
  correlation_min_scale: 0.5      # minimum scale at max correlation
  cash_min_pct: 0.1               # keep at least 10% cash
  cash_min_scale: 0.5             # minimum scale when below cash_min_pct
```

Disable Telegram notifications explicitly:
```yaml
notifications:
  telegram:
    enabled: false
```

## Security Notes

- All scripts run as the `trader` user (no root access)
- SSH keys only (password authentication disabled)
- UFW firewall configured
- Fail2ban intrusion prevention active
- Regular log rotation and backups

## Troubleshooting

### Common Issues

1. **Tmux session not starting**
   ```bash
   # Check tmux is installed
   which tmux

   # Check permissions
   ls -la /opt/trading_bot/scripts/tmux_manager.sh
   ```

2. **Telegram notifications not working**
   ```bash
   # Verify environment variables
   cat /etc/trading_bot/.env

   # Test Python import
   python3 -c "from src.notifier.enhanced_telegram import EnhancedTelegramNotifier"
   ```

3. **Systemd service failing**
   ```bash
   # Check service status
   sudo systemctl status trading-bot

   # View logs
   sudo journalctl -u trading-bot -f
   ```

### Log Locations
- Application logs: `/var/log/trading_bot/`
- System logs: `/var/log/syslog`
- Deployment logs: `/opt/trading_bot/logs/deployment.log`

## Backup and Recovery

### Automated Backups
The system includes automated backup scripts that run daily:
- Configuration backup: `backup_config.sh`
- Log rotation: `rotate_logs.sh`

### Manual Backup
```bash
# Create manual backup
/opt/trading_bot/scripts/backup_config.sh

# List backups
ls -la /opt/trading_bot/backups/
```

## Performance Monitoring

### Real-time Monitoring
```bash
# Start monitoring dashboard
/opt/trading_bot/scripts/start_monitoring.sh

# View performance metrics
cat /opt/trading_bot/logs/performance_metrics.json
```

## Contributor Docs

For development and contribution guidelines, see:
- AGENTS.md — repository structure, style, testing, and PR process
- DEPLOY_ROLLBACK_PLAYBOOK.md — deploy and rollback checklist for this server

### Health Checks
```bash
# Run health check
/opt/trading_bot/scripts/health_check.sh

# Automated health checks run every 5 minutes via cron
```

## Support

For issues or questions:
1. Check the logs in `/var/log/trading_bot/`
2. Review systemd service status
3. Verify all environment variables are set
4. Ensure all dependencies are installed

The system is designed for high availability with automatic restarts and comprehensive monitoring.
