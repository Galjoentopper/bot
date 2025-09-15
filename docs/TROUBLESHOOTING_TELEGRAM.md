# Telegram Bot Troubleshooting Guide

## Issue Found: Error Log Detected

There's an error log at `/opt/trading_bot/bot/logs/error.log` that needs to be examined.

## Diagnostic Commands to Run:

### 1. Check the error log content:
```bash
cat logs/error.log
```

### 2. Check recent telegram logs:
```bash
ls -la logs/telegram_*.log
tail -50 logs/telegram_*.log
```

### 3. Check what's in the tmux session:
```bash
# Capture what's showing in the telegram tmux pane
tmux capture-pane -t telegram_session -p

# List all panes in the session
tmux list-panes -t telegram_session
```

### 4. Check environment variables:
```bash
echo "TELEGRAM_BOT_TOKEN: $TELEGRAM_BOT_TOKEN"
echo "TELEGRAM_CHAT_ID: $TELEGRAM_CHAT_ID"
```

### 5. Try manual start to see live errors:
```bash
cd /opt/trading_bot/bot
python3 telegram_bot_listener.py
```

## Common Issues and Solutions:

### Issue 1: Missing Environment Variables
**Symptoms**: Bot starts but can't connect to Telegram
**Solution**:
```bash
# Check if .env file exists
ls -la .env

# Source the environment file
source .env
# OR
source /etc/trading_bot/.env
```

### Issue 2: Missing Python Dependencies
**Symptoms**: Import errors in logs
**Solution**:
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

### Issue 3: Permission Issues
**Symptoms**: Permission denied errors
**Solution**:
```bash
# Fix permissions
chmod +x telegram_bot_listener.py
chmod -R 755 logs/
```

### Issue 4: Telegram Token Issues
**Symptoms**: Bot starts but gets API errors
**Solution**: Verify your bot token with BotFather on Telegram

## Next Steps:
1. Run `cat logs/error.log` to see the specific error
2. Based on the error, apply the appropriate solution above
3. Restart the system with `./start_system.sh`
