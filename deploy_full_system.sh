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

setup_directories() {
    log_info "Setting up directories..."

    # Create necessary directories in current location
    mkdir -p logs
    mkdir -p backups
    mkdir -p data
    mkdir -p /etc/trading_bot 2>/dev/null || sudo mkdir -p /etc/trading_bot

    # Set permissions
    sudo chown -R $USER:$USER logs backups data 2>/dev/null || true

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
        pip install -r requirements.txt
    else
        log_error "requirements.txt not found in current directory: $SCRIPT_DIR"
        exit 1
    fi

    log_success "Python environment setup completed"
}

setup_systemd_services() {
    log_info "Setting up systemd services..."

    # Copy systemd service files from current directory
    if [ -f "$SCRIPT_DIR/trading-bot.service" ]; then
        sudo cp "$SCRIPT_DIR/trading-bot.service" /etc/systemd/system/
    else
        log_warning "trading-bot.service not found in $SCRIPT_DIR"
    fi

    if [ -f "$SCRIPT_DIR/telegram-bot-listener.service" ]; then
        sudo cp "$SCRIPT_DIR/telegram-bot-listener.service" /etc/systemd/system/
    else
        log_warning "telegram-bot-listener.service not found in $SCRIPT_DIR"
    fi

    # Reload systemd daemon
    sudo systemctl daemon-reload

    # Enable services (but don't start them yet)
    sudo systemctl enable trading-bot.service 2>/dev/null || log_warning "Could not enable trading-bot.service"
    sudo systemctl enable telegram-bot-listener.service 2>/dev/null || log_warning "Could not enable telegram-bot-listener.service"

    log_success "Systemd services configured"
}

setup_cron_jobs() {
    log_info "Setting up cron jobs..."

    # Copy cron configuration from current directory
    if [ -f "$SCRIPT_DIR/trading_bot_monitor" ]; then
        sudo cp "$SCRIPT_DIR/trading_bot_monitor" /etc/cron.d/
        sudo chmod 644 /etc/cron.d/trading_bot_monitor
    fi

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

# Stay in current directory
# cd is not needed since we're already here

# Activate virtual environment
source venv/bin/activate

# Start both services using tmux
./scripts/enhanced_tmux_manager.sh start

echo "✅ Trading system started!"
echo ""
echo "To check status: ./scripts/enhanced_tmux_manager.sh status"
echo "To view logs: ./scripts/enhanced_tmux_manager.sh logs"
echo "To stop system: ./scripts/enhanced_tmux_manager.sh stop"
EOF

    chmod +x start_system.sh

    log_success "Startup script created"
}

test_system() {
    log_info "Running system tests..."

    # Stay in current directory
    cd "$SCRIPT_DIR"
    source venv/bin/activate

    # Test Python imports
    python3 -c "import sys; sys.path.insert(0, 'src'); from src.notifier.telegram import TelegramNotifier; print('✅ Telegram import successful')" 2>/dev/null || log_warning "Telegram import test failed"

    # Test configuration loading
    python3 -c "from src.config.config_loader import ConfigLoader; config = ConfigLoader('training_config.yaml').config; print('✅ Configuration loading successful')" 2>/dev/null || log_warning "Configuration loading test failed"

    log_success "System tests completed"
}

show_deployment_summary() {
    log_info "=== DEPLOYMENT SUMMARY ==="
    echo ""
    echo "📁 Installation Directory: $SCRIPT_DIR"
    echo "📝 Configuration: $SCRIPT_DIR/training_config.yaml"
    echo "📊 Logs: $SCRIPT_DIR/logs/"
    echo "💾 Backups: $SCRIPT_DIR/backups/"
    echo ""
    echo "🚀 Quick Start Commands:"
    echo "  ./start_system.sh                    # Start both services"
    echo "  ./scripts/enhanced_tmux_manager.sh status    # Check status"
    echo "  ./scripts/enhanced_tmux_manager.sh logs      # View logs"
    echo "  ./scripts/enhanced_tmux_manager.sh stop      # Stop services"
    echo ""
    echo "🤖 Telegram Commands Available:"
    echo "  /status    - System status"
    echo "  /start     - Start trading"
    echo "  /stop      - Stop trading"
    echo "  /restart   - Restart trading"
    echo "  /performance - Performance metrics"
    echo "  /health    - System health"
    echo "  /balance   - Current balance"
    echo "  /trades    - Recent trades"
    echo "  /logs      - System logs"
    echo "  /config    - Configuration"
    echo "  /help      - Show all commands"
    echo ""
    echo "⚙️  Systemd Services:"
    echo "  sudo systemctl start trading-bot"
    echo "  sudo systemctl start telegram-bot-listener"
    echo "  sudo systemctl enable trading-bot"
    echo "  sudo systemctl enable telegram-bot-listener"
    echo ""
    log_success "Deployment completed successfully!"
}

main() {
    echo "🤖 Trading System Full Deployment"
    echo "=================================="
    echo ""

    check_dependencies
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
    echo "🎉 Deployment completed! Your trading system is ready to use."
    echo "   Run: ./start_system.sh"
}

# Run main function
main "$@"