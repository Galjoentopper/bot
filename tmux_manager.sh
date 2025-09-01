#!/bin/bash
# Tmux Session Manager for Trading System
# ======================================
# Provides session management, monitoring, and control for trading operations

set -e

# Configuration
SESSION_NAME="trading_session"
LOG_DIR="logs"
TMUX_LOG="$LOG_DIR/tmux_manager.log"
MAX_RETRIES=3
RETRY_DELAY=5

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging functions
log_info() {
    echo "[$(date)] [INFO] $1" | tee -a "$TMUX_LOG"
}

log_error() {
    echo "[$(date)] [ERROR] $1" | tee -a "$TMUX_LOG" >&2
}

log_success() {
    echo "[$(date)] [SUCCESS] $1" | tee -a "$TMUX_LOG"
}

# Check if tmux is available
check_tmux() {
    if ! command -v tmux &>/dev/null; then
        log_error "tmux is not installed or not in PATH"
        echo "Please install tmux: sudo apt-get install tmux"
        exit 1
    fi
}

# Check if session exists
session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# Get session status
get_session_status() {
    if session_exists; then
        if tmux list-panes -t "$SESSION_NAME" 2>/dev/null | grep -q "dead"; then
            echo "dead"
        else
            echo "running"
        fi
    else
        echo "not_found"
    fi
}

# Start tmux session with command
start_session() {
    local command="$1"
    local timeout="${2:-300}"  # Default 5 minutes

    if [ -z "$command" ]; then
        log_error "No command provided for session start"
        echo "Usage: $0 start <command> [timeout_seconds]"
        exit 1
    fi

    log_info "Starting tmux session '$SESSION_NAME' with command: $command (timeout: ${timeout}s)"

    # Kill existing session if it exists
    if session_exists; then
        log_info "Killing existing session '$SESSION_NAME'"
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        sleep 2
    fi

    # Create new session with timeout wrapper
    local wrapped_command="timeout ${timeout}s $command || echo 'Command timed out after ${timeout} seconds'"

    if tmux new-session -d -s "$SESSION_NAME" "$wrapped_command"; then
        log_success "Tmux session '$SESSION_NAME' started successfully"
        echo "Session started. Use '$0 attach' to view or '$0 logs' to see output"
    else
        log_error "Failed to start tmux session '$SESSION_NAME'"
        exit 1
    fi
}

# Stop tmux session
stop_session() {
    if ! session_exists; then
        log_info "Session '$SESSION_NAME' does not exist"
        return 0
    fi

    log_info "Stopping tmux session '$SESSION_NAME'"

    for attempt in $(seq 1 $MAX_RETRIES); do
        if tmux kill-session -t "$SESSION_NAME" 2>/dev/null; then
            log_success "Tmux session '$SESSION_NAME' stopped successfully"
            return 0
        else
            log_error "Failed to stop session (attempt $attempt/$MAX_RETRIES)"
            if [ $attempt -lt $MAX_RETRIES ]; then
                sleep $RETRY_DELAY
            fi
        fi
    done

    log_error "Failed to stop session after $MAX_RETRIES attempts"
    return 1
}

# Show session status
show_status() {
    local status=$(get_session_status)

    case "$status" in
        "running")
            log_info "Session '$SESSION_NAME' is running"
            echo "Status: RUNNING"
            echo "Session: $SESSION_NAME"
            echo "Created: $(tmux list-sessions -F "#{session_created} #{session_name}" 2>/dev/null | grep "$SESSION_NAME" | awk '{print strftime("%Y-%m-%d %H:%M:%S", $1)}' || echo "Unknown")"
            ;;
        "dead")
            log_info "Session '$SESSION_NAME' is dead"
            echo "Status: DEAD"
            ;;
        "not_found")
            log_info "Session '$SESSION_NAME' does not exist"
            echo "Status: NOT FOUND"
            ;;
    esac
}

# Attach to session
attach_session() {
    if ! session_exists; then
        log_error "Session '$SESSION_NAME' does not exist"
        echo "Use '$0 start <command>' to create a session first"
        exit 1
    fi

    local status=$(get_session_status)
    if [ "$status" = "dead" ]; then
        log_error "Session '$SESSION_NAME' is dead"
        echo "Use '$0 start <command>' to restart"
        exit 1
    fi

    log_info "Attaching to tmux session '$SESSION_NAME'"
    echo "Press Ctrl+B then D to detach from session"
    tmux attach-session -t "$SESSION_NAME"
}

# Show session logs/output
show_logs() {
    if ! session_exists; then
        log_error "Session '$SESSION_NAME' does not exist"
        echo "No logs available"
        return 1
    fi

    log_info "Showing logs for session '$SESSION_NAME'"

    # Try to capture pane output
    if tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null; then
        echo "=== Session Output ==="
        tmux capture-pane -t "$SESSION_NAME" -p
    else
        log_error "Failed to capture session output"
        echo "Session may not have any output or is not accessible"
    fi
}

# List all sessions
list_sessions() {
    log_info "Listing all tmux sessions"
    echo "=== Tmux Sessions ==="
    tmux list-sessions 2>/dev/null || echo "No tmux sessions found"
}

# Clean up dead sessions
cleanup() {
    log_info "Cleaning up dead tmux sessions"
    local cleaned=0

    for session in $(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -v "^$SESSION_NAME$" || true); do
        if tmux list-panes -t "$session" 2>/dev/null | grep -q "dead"; then
            log_info "Removing dead session: $session"
            tmux kill-session -t "$session" 2>/dev/null || true
            cleaned=$((cleaned + 1))
        fi
    done

    # Also check our main session
    if session_exists && [ "$(get_session_status)" = "dead" ]; then
        log_info "Removing dead main session: $SESSION_NAME"
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        cleaned=$((cleaned + 1))
    fi

    log_success "Cleaned up $cleaned dead sessions"
}

# Main command handler
main() {
    check_tmux

    case "${1:-help}" in
        "start")
            shift
            start_session "$@"
            ;;
        "stop")
            stop_session
            ;;
        "status")
            show_status
            ;;
        "attach")
            attach_session
            ;;
        "logs")
            show_logs
            ;;
        "list")
            list_sessions
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            echo "Tmux Session Manager for Trading System"
            echo ""
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  start <command> [timeout]  Start new session with command (default timeout: 300s)"
            echo "  stop                       Stop the trading session"
            echo "  status                     Show session status"
            echo "  attach                     Attach to running session"
            echo "  logs                       Show session output/logs"
            echo "  list                       List all tmux sessions"
            echo "  cleanup                    Remove dead sessions"
            echo "  help                       Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 start 'python3 scripts/trader.py --config config.yaml' 300"
            echo "  $0 status"
            echo "  $0 attach"
            echo "  $0 logs"
            echo "  $0 stop"
            ;;
    esac
}

# Run main function with all arguments
main "$@"