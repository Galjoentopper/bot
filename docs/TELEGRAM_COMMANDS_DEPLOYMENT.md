# Telegram Commands Deployment Guide

## 🎯 Problem Solved
Your Telegram bot was only sending messages but not responding to commands. This guide fixes that by implementing a complete interactive Telegram command system.

## 📋 What's Been Created

### 1. Telegram Bot Listener (`telegram_bot_listener.py`)
- **Purpose**: Handles incoming Telegram messages and commands
- **Features**:
  - Responds to all commands: `/status`, `/start`, `/stop`, `/restart`, `/performance`, `/health`, `/balance`, `/trades`, `/logs`, `/config`, `/help`
  - Real-time command processing
  - Error handling and logging
  - Graceful shutdown handling

### 2. Enhanced Tmux Manager (`scripts/enhanced_tmux_manager.sh`)
- **Purpose**: Manages both trading system and Telegram bot
- **Features**:
  - Starts/stops both services simultaneously
  - Separate tmux sessions for each service
  - Status monitoring for both services
  - Log viewing for both services

### 3. Systemd Service (`telegram-bot-listener.service`)
- **Purpose**: Production service for Telegram bot listener
- **Features**:
  - Auto-start on boot
  - Proper user permissions
  - Resource limits
  - Logging integration

### 4. Full Deployment Script (`deploy_full_system.sh`)
- **Purpose**: Complete system deployment
- **Features**:
  - Automated setup of all components
  - Directory structure creation
  - Python environment setup
  - Service configuration
  - Testing and validation

## 🚀 Deployment Steps

### Step 1: Upload Files to Server
```bash
# Copy the new files to your server
scp telegram_bot_listener.py user@your-server:/path/to/bot/
scp scripts/enhanced_tmux_manager.sh user@your-server:/path/to/bot/scripts/
scp telegram-bot-listener.service user@your-server:/path/to/bot/
scp deploy_full_system.sh user@your-server:/path/to/bot/
```

### Step 2: Run Full Deployment
```bash
# Make deployment script executable
chmod +x deploy_full_system.sh

# Run deployment (will setup everything)
./deploy_full_system.sh
```

### Step 3: Manual Setup (Alternative)
If you prefer manual setup:

```bash
# 1. Setup directories
sudo mkdir -p /opt/trading_bot
sudo chown -R $USER:$USER /opt/trading_bot

# 2. Copy files
cp -r * /opt/trading_bot/
sudo cp telegram-bot-listener.service /etc/systemd/system/

# 3. Setup Python environment
cd /opt/trading_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Setup systemd
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot-listener
```

### Step 4: Start the System
```bash
cd /opt/trading_bot

# Start both services
./scripts/enhanced_tmux_manager.sh start

# Or start individually
./scripts/enhanced_tmux_manager.sh start  # Trading + Telegram
```

### Step 5: Test Commands
Send these commands to your Telegram bot:

```
/help          # Show all available commands
/status        # Check system status
/start         # Start trading system
/stop          # Stop trading system
/performance   # Get performance metrics
/health        # System health check
/balance       # Current balance and positions
/trades        # Recent trades
/logs          # System logs
/config        # Configuration info
```

## 🔧 Available Commands

| Command | Description | Example Response |
|---------|-------------|------------------|
| `/help` | Show all commands | Lists all available commands |
| `/status` | System status | ✅ Running on server-name |
| `/start` | Start trading | 🚀 Trading system started |
| `/stop` | Stop trading | 🛑 Trading system stopped |
| `/restart` | Restart trading | 🔄 Trading system restarted |
| `/performance` | Performance metrics | Portfolio: €10,000 P&L: +€150 |
| `/health` | System health | ✅ All systems healthy |
| `/balance` | Current balance | Cash: €9,850 Positions: €150 |
| `/trades` | Recent trades | Last 5 trades with details |
| `/logs` | System logs | Recent log entries |
| `/config` | Configuration | Current config settings |

## 📊 Monitoring Commands

### Check System Status
```bash
# Check both services
./scripts/enhanced_tmux_manager.sh status

# View logs
./scripts/enhanced_tmux_manager.sh logs

# Attach to sessions
./scripts/enhanced_tmux_manager.sh attach
```

### Systemd Management
```bash
# Check service status
sudo systemctl status telegram-bot-listener
sudo systemctl status trading-bot

# View service logs
sudo journalctl -u telegram-bot-listener -f
sudo journalctl -u trading-bot -f

# Restart services
sudo systemctl restart telegram-bot-listener
sudo systemctl restart trading-bot
```

## 🔍 Troubleshooting

### Telegram Bot Not Responding
1. **Check if service is running**:
   ```bash
   ./scripts/enhanced_tmux_manager.sh status
   ```

2. **Check Telegram logs**:
   ```bash
   ./scripts/enhanced_tmux_manager.sh logs
   ```

3. **Verify bot token and chat ID**:
   ```bash
   grep -E "(bot_token|chat_id)" training_config.yaml
   ```

4. **Test bot connectivity**:
   ```bash
   python3 -c "
   from src.notifier.telegram import TelegramNotifier
   from src.config.config_loader import ConfigLoader
   config = ConfigLoader().config
   notifier = TelegramNotifier.from_config(config)
   print('Bot initialized successfully')
   "
   ```

### Common Issues

**Issue**: "Telegram session already running"
**Solution**: Stop existing session first
```bash
./scripts/enhanced_tmux_manager.sh stop
./scripts/enhanced_tmux_manager.sh start
```

**Issue**: Commands work but responses are delayed
**Solution**: Check server resources and network connectivity
```bash
htop  # Check CPU/memory usage
ping api.telegram.org  # Check network
```

**Issue**: Bot token invalid
**Solution**: Update token in `training_config.yaml`
```yaml
notifications:
  telegram:
    bot_token: 'YOUR_NEW_BOT_TOKEN'
    chat_id: 'YOUR_CHAT_ID'
```

## 📈 Performance Optimization

### Resource Limits
The systemd service includes resource limits:
- Memory: 256MB max
- CPU: 50% max
- Proper user isolation

### Log Management
- Automatic log rotation (daily)
- 30-day retention
- Compressed old logs

### Monitoring
- Health checks every 15 minutes
- Automatic recovery on failures
- Performance reporting every hour

## 🎯 Next Steps

1. **Test all commands** in Telegram
2. **Monitor system performance** with `/status` and `/health`
3. **Set up automated alerts** if needed
4. **Configure backup systems** for critical data

## 📞 Support

If you encounter issues:

1. Check the logs: `./scripts/enhanced_tmux_manager.sh logs`
2. Verify configuration: `cat training_config.yaml`
3. Test individual components: `python3 telegram_bot_listener.py`
4. Check system resources: `htop` and `df -h`

The system is now fully interactive and will respond to all your Telegram commands! 🤖✨
