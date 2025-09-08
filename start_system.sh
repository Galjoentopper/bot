#!/bin/bash
# Trading System Startup Script

echo "🚀 Starting Trading System..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Start both services using tmux
if [ -f "scripts/enhanced_tmux_manager.sh" ]; then
    ./scripts/enhanced_tmux_manager.sh start
else
    echo "❌ Enhanced tmux manager not found. Starting services manually..."
    # Fallback to direct execution
    tmux new-session -d -s trading-bot "python3 scripts/enhanced_trader.py"
    tmux new-session -d -s telegram-bot "python3 telegram_bot_listener_fixed.py"
fi

echo "✅ Trading system started!"
echo ""
echo "📊 Available Telegram commands:"
echo "  /status     - System status"
echo "  /start      - Start trading"  
echo "  /stop       - Stop trading"
echo "  /restart    - Restart system"
echo "  /performance- Performance metrics"
echo "  /health     - Health check"
echo "  /balance    - Account balance"
echo "  /trades     - Recent trades"
echo "  /logs       - View logs"
echo "  /config     - Configuration"
echo "  /help       - Command help"
echo ""
echo "📊 System management commands:"
echo "  ./scripts/enhanced_tmux_manager.sh status    # Check status"
echo "  ./scripts/enhanced_tmux_manager.sh logs      # View logs"
echo "  ./scripts/enhanced_tmux_manager.sh stop      # Stop system"
echo ""
echo "🐧 Systemd commands:"
echo "  sudo systemctl start trading-bot             # Start trading service"
echo "  sudo systemctl start telegram-bot-listener   # Start telegram service"
echo "  sudo systemctl status trading-bot            # Check trading status"
echo "  sudo systemctl status telegram-bot-listener  # Check telegram status"
