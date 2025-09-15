#!/bin/bash
# Development Tools Script
# Provides utilities for development process improvements

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."

    if ! command -v pre-commit &> /dev/null; then
        log_info "Installing pre-commit..."
        pip install pre-commit
    fi

    cd "$PROJECT_ROOT"
    pre-commit install
    log_info "Pre-commit hooks installed successfully"
}

# Function to run syntax validation on all Python files
validate_syntax() {
    log_info "Running syntax validation on all Python files..."

    cd "$PROJECT_ROOT"

    # Find all Python files and validate syntax
    find . -name "*.py" -not -path "./venv/*" -not -path "./__pycache__/*" -not -path "./.git/*" | while read -r file; do
        if ! python -m py_compile "$file"; then
            log_error "Syntax error in: $file"
            return 1
        fi
    done

    log_info "All Python files have valid syntax"
}

# Function to validate critical imports
validate_imports() {
    log_info "Validating critical system imports..."

    cd "$PROJECT_ROOT"

    # Test critical imports
    python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.notifier.telegram import TelegramNotifier
    print('✓ TelegramNotifier import OK')
except Exception as e:
    print(f'✗ TelegramNotifier import failed: {e}')
    sys.exit(1)

try:
    from src.notifications.telegram_service import TelegramService
    print('✓ TelegramService import OK')
except Exception as e:
    print(f'✗ TelegramService import failed: {e}')
    sys.exit(1)

try:
    from src.core.enhanced_logger import EnhancedLogger
    print('✓ EnhancedLogger import OK')
except Exception as e:
    print(f'✗ EnhancedLogger import failed: {e}')
    sys.exit(1)

print('All critical imports validated successfully')
    "
}

# Function to check for common issues
check_common_issues() {
    log_info "Checking for common issues..."

    cd "$PROJECT_ROOT"

    # Check for indentation issues
    log_info "Checking for mixed indentation..."
    if grep -r $'\t' --include="*.py" src/ 2>/dev/null; then
        log_warning "Found files with tab characters (should use spaces)"
    fi

    # Check for long lines
    log_info "Checking for long lines (>120 chars)..."
    if grep -r -n ".\{121\}" --include="*.py" src/ 2>/dev/null | head -5; then
        log_warning "Found lines longer than 120 characters"
    fi

    # Check for TODO/FIXME comments
    log_info "Checking for TODO/FIXME comments..."
    grep -r -n "TODO\|FIXME" --include="*.py" src/ 2>/dev/null | head -10 || true

    log_info "Common issues check completed"
}

# Function to run full validation suite
run_full_validation() {
    log_info "Running full validation suite..."

    validate_syntax
    validate_imports
    check_common_issues

    # Run pre-commit on all files
    if command -v pre-commit &> /dev/null; then
        log_info "Running pre-commit checks..."
        pre-commit run --all-files || true
    fi

    log_info "Full validation completed"
}

# Function to show help
show_help() {
    echo "Development Tools Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup         Setup pre-commit hooks"
    echo "  validate      Run syntax validation"
    echo "  imports       Test critical imports"
    echo "  check         Check for common issues"
    echo "  full          Run full validation suite"
    echo "  help          Show this help message"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "setup")
        setup_pre_commit
        ;;
    "validate")
        validate_syntax
        ;;
    "imports")
        validate_imports
        ;;
    "check")
        check_common_issues
        ;;
    "full")
        run_full_validation
        ;;
    "help"|*)
        show_help
        ;;
esac
