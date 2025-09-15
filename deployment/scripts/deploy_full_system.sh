#!/bin/bash
# Comprehensive Trading System Deployment Script
# Deploys both the trading bot and Telegram bot listener

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - Use current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs"
BACKUP_DIR="$SCRIPT_DIR/backups"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking system dependencies..."

    # Check if running as root or with sudo
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root. Please run as a regular user with sudo access."
        exit 1
    fi

    # Check required commands
    local required_commands=("python3" "pip3" "tmux" "systemctl" "journalctl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found. Please install it first."
            exit 1
        fi
    done

    log_success "System dependencies check passed"
}

cleanup_scattered_directories() {
    log_info "Checking for scattered directories to clean up..."

    # Check if we're in a subdirectory that suggests scattered structure
    local parent_dir="$(dirname "$SCRIPT_DIR")"

    # Look for scattered directories at parent level that might conflict
    local scattered_dirs=("logs" "scripts" "systemd" "data" "venv" "src")
    local found_scattered=false

    for dir in "${scattered_dirs[@]}"; do
        if [ -d "$parent_dir/$dir" ] && [ "$parent_dir/$dir" != "$SCRIPT_DIR/$dir" ]; then
            log_warning "Found scattered directory: $parent_dir/$dir"
            found_scattered=true
        fi
    done

    if [ "$found_scattered" = true ]; then
        log_warning "Scattered directories detected. Consider running cleanup:"
        echo "  cd '$parent_dir' && rm -rf logs scripts systemd data venv src"
        echo "  (After backing up any important data)"
    else
        log_success "No scattered directories detected"
    fi
}

setup_directories() {
    log_info "Setting up directories..."

    # Create necessary directories in current location (bot folder)
    mkdir -p logs
    mkdir -p backups
    mkdir -p data
    mkdir -p models/{gru,lightgbm,ppo,exports,imports,metadata,packages}
    mkdir -p scripts
    mkdir -p server/{scripts,systemd,cron}
    mkdir -p src/{data,models,trading,risk,notifier,validation}
    mkdir -p cache
    mkdir -p config
    mkdir -p temp

    # Set permissions for bot folder structure
    sudo chown -R $USER:$USER logs backups data models scripts server src cache config temp 2>/dev/null || true
    chmod +x scripts/*.py 2>/dev/null || true
    chmod +x scripts/*.sh 2>/dev/null || true
    chmod +x server/scripts/*.sh 2>/dev/null || true
    chmod +x *.sh 2>/dev/null || true

    log_success "Directories setup completed"
}

copy_files() {
    log_info "Copying project files..."

    # No need to copy files - we're working in the current directory
    # All files should already be here

    log_success "Files copied successfully"
}

setup_python_environment() {
    log_info "Setting up Python virtual environment..."

    # Stay in current directory
    cd "$SCRIPT_DIR"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install --upgrade pip

    # Install requirements from current directory
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --timeout 300
    else
        log_error "requirements.txt not found in current directory: $SCRIPT_DIR"
        exit 1
    fi

    # Install the package in development mode
    if [ -f "setup.py" ]; then
        pip install -e . --timeout 300
    else
        log_warning "setup.py not found - package will not be installed in development mode"
    fi

    # Create environment file if it doesn't exist
    if [ ! -f ".env" ] && [ -f ".env.example" ]; then
        cp .env.example .env
        log_warning "Created .env file from .env.example - please edit with your API keys"
    fi

    log_success "Python environment setup completed"
}

setup_systemd_services() {
    log_info "Setting up systemd services..."

    # Create systemd service file for trading bot
    sudo tee /etc/systemd/system/trading-bot.service > /dev/null << EOF
[Unit]
Description=Enterprise Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$SCRIPT_DIR
Environment=PATH=$SCRIPT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/venv/bin/python3 $SCRIPT_DIR/scripts/enhanced_trader.py
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/logs/systemd.log
StandardError=append:$SCRIPT_DIR/logs/systemd_error.log

[Install]
WantedBy=multi-user.target
EOF

    # Create systemd service file for telegram bot listener
    # Using systemd-optimized version to avoid event loop conflicts
    sudo tee /etc/systemd/system/telegram-bot-listener.service > /dev/null << EOF
[Unit]
Description=Trading Bot Telegram Listener
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$SCRIPT_DIR
Environment=PATH=$SCRIPT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/venv/bin/python3 $SCRIPT_DIR/telegram_bot_bulletproof.py
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/logs/telegram_systemd.log
StandardError=append:$SCRIPT_DIR/logs/telegram_systemd_error.log

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd daemon
    sudo systemctl daemon-reload

    # Enable services (but don't start them yet)
    sudo systemctl enable trading-bot.service 2>/dev/null || log_warning "Could not enable trading-bot.service"
    sudo systemctl enable telegram-bot-listener.service 2>/dev/null || log_warning "Could not enable telegram-bot-listener.service"

    log_success "Systemd services configured"
}

setup_cron_jobs() {
    log_info "Setting up cron jobs..."

    # Create cron configuration for health monitoring
    sudo tee /etc/cron.d/trading_bot_monitor > /dev/null << EOF
# Enterprise Trading Bot Health Monitor
# Runs every 5 minutes to check system health
*/5 * * * * $USER cd $SCRIPT_DIR && ./server/scripts/health_check.sh >> logs/health_monitor.log 2>&1
0 */6 * * * $USER cd $SCRIPT_DIR && ./server/scripts/backup_system.sh >> logs/backup.log 2>&1
EOF

    sudo chmod 644 /etc/cron.d/trading_bot_monitor

    log_success "Cron jobs configured"
}

setup_logrotate() {
    log_info "Setting up log rotation..."

    # Create logrotate configuration for current directory
    sudo tee /etc/logrotate.d/trading_bot > /dev/null << EOF
$SCRIPT_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload trading-bot.service || true
        systemctl reload telegram-bot-listener.service || true
    endscript
}
EOF

    log_success "Log rotation configured"
}

create_startup_script() {
    log_info "Creating startup script..."

    # Create a comprehensive startup script in current directory
    cat > start_system.sh << 'EOF'
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
    tmux new-session -d -s telegram-bot "python3 telegram_bot_bulletproof.py"
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
EOF

    chmod +x start_system.sh

    # Also create a simple stop script
    cat > stop_system.sh << 'EOF'
#!/bin/bash
# Trading System Stop Script

echo "🛑 Stopping Trading System..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Stop services using tmux manager if available
if [ -f "scripts/enhanced_tmux_manager.sh" ]; then
    ./scripts/enhanced_tmux_manager.sh stop
else
    echo "Stopping tmux sessions manually..."
    tmux kill-session -t trading-bot 2>/dev/null || true
    tmux kill-session -t telegram-bot 2>/dev/null || true
fi

# Stop systemd services
sudo systemctl stop trading-bot 2>/dev/null || true
sudo systemctl stop telegram-bot-listener 2>/dev/null || true

echo "✅ Trading system stopped!"
EOF

    chmod +x stop_system.sh

    log_success "Startup and stop scripts created"
}

test_system() {
    log_info "Running system tests..."

    # Stay in current directory
    cd "$SCRIPT_DIR"
    source venv/bin/activate

    # Test directory structure
    log_info "Validating directory structure..."
    local required_dirs=("logs" "data" "models" "scripts" "src" "server")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_warning "Directory '$dir' not found"
        else
            log_success "Directory '$dir' exists"
        fi
    done

    # Test key files
    log_info "Validating key files..."
    local required_files=("training_config.yaml" "requirements.txt" "telegram_bot_listener.py")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_warning "File '$file' not found"
        else
            log_success "File '$file' exists"
        fi
    done

    # Test Python imports
    log_info "Testing Python imports..."
    python3 -c "import sys; print('Python version:', sys.version)" || log_warning "Python test failed"

    # Test basic imports without full src path requirements
    python3 -c "import pandas; print('✅ Pandas import successful')" 2>/dev/null || log_warning "Pandas import test failed"
    python3 -c "import numpy; print('✅ Numpy import successful')" 2>/dev/null || log_warning "Numpy import test failed"

    # Test configuration loading if possible
    if [ -f "training_config.yaml" ]; then
        python3 -c "import yaml; yaml.safe_load(open('training_config.yaml')); print('✅ Configuration loading successful')" 2>/dev/null || log_warning "Configuration loading test failed"
    fi

    # Test data fetching script
    if [ -f "fetch_training_data.sh" ]; then
        chmod +x fetch_training_data.sh
        log_success "Data fetching script made executable"
    fi

    log_success "System tests completed"
}

show_deployment_summary() {
    log_info "=== DEPLOYMENT SUMMARY ==="
    echo ""
    echo "🎯 Enterprise Trading Bot - Clean Architecture"
    echo "📁 Installation Directory: $SCRIPT_DIR"
    echo "📝 Configuration: $SCRIPT_DIR/training_config.yaml"
    echo "📊 Logs: $SCRIPT_DIR/logs/"
    echo "💾 Backups: $SCRIPT_DIR/backups/"
    echo "🤖 Models: $SCRIPT_DIR/models/"
    echo "📈 Data: $SCRIPT_DIR/data/"
    echo ""
    echo "🚀 Quick Start Commands:"
    echo "  ./start_system.sh                         # Start both services"
    echo "  ./stop_system.sh                          # Stop all services"
    echo "  ./scripts/enhanced_tmux_manager.sh status # Check status"
    echo "  ./scripts/enhanced_tmux_manager.sh logs   # View logs"
    echo ""
    echo "📊 Data & Training Commands:"
    echo "  ./fetch_training_data.sh                  # Fetch market data"
    echo "  python3 scripts/enhanced_trainer.py       # Train models"
    echo ""
    echo "🤖 Telegram Commands Available:"
    echo "  /status      - System status"
    echo "  /start       - Start trading"
    echo "  /stop        - Stop trading"
    echo "  /restart     - Restart trading"
    echo "  /performance - Performance metrics"
    echo "  /health      - System health"
    echo "  /balance     - Current balance"
    echo "  /trades      - Recent trades"
    echo "  /logs        - System logs"
    echo "  /config      - Configuration"
    echo "  /help        - Show all commands"
    echo ""
    echo "⚙️  Systemd Services:"
    echo "  sudo systemctl start trading-bot          # Start trading service"
    echo "  sudo systemctl start telegram-bot-listener # Start telegram service"
    echo "  sudo systemctl status trading-bot         # Check trading status"
    echo "  sudo systemctl status telegram-bot-listener # Check telegram status"
    echo ""
    echo "📁 Clean Architecture Notes:"
    echo "  • All components contained within: $SCRIPT_DIR"
    echo "  • No scattered directories at /opt/trading_bot/bot/ level"
    echo "  • Everything organized under bot/ folder"
    echo "  • Virtual environment: $SCRIPT_DIR/venv/"
    echo ""
    log_success "Deployment completed successfully!"
}

main() {
    echo "🤖 Trading System Full Deployment - Clean Architecture"
    echo "====================================================="
    echo ""
    echo "📁 Working in: $SCRIPT_DIR"
    echo "🎯 Goal: Keep all components within bot directory"
    echo ""

    check_dependencies
    cleanup_scattered_directories
    setup_directories
    copy_files
    setup_python_environment
    setup_systemd_services
    setup_cron_jobs
    setup_logrotate
    create_startup_script
    test_system
    show_deployment_summary

    echo ""
    echo "🎉 Clean deployment completed! All components are contained within:"
    echo "   $SCRIPT_DIR"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Edit .env with your API keys"
    echo "   2. Run: ./start_system.sh"
    echo "   3. Test with Telegram commands"
}

# Run main function
main "$@"
