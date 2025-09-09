#!/bin/bash

echo "🚀 Setting up Trading Bot Improvements..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "training_config.yaml" ]; then
    print_error "Please run this script from the bot directory"
    exit 1
fi

print_info "Installing new dependencies..."

# Install new requirements
if pip install -r requirements.txt; then
    print_success "Dependencies installed successfully"
else
    print_warning "Some dependencies may have failed to install"
fi

echo ""
print_info "Setting up development tools..."

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found, skipping hook installation"
fi

# Create necessary directories
echo ""
print_info "Creating directory structure..."

dirs=("tests/unit" "tests/integration" "tests/performance" "tests/contract" "logs" ".github/workflows")
for dir in "${dirs[@]}"; do
    mkdir -p "$dir"
    print_success "Created $dir/"
done

# Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env template..."
    cat > .env << EOF
# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Exchange API Keys (Optional)
BITVAVO_API_KEY=your_bitvavo_key_here
BITVAVO_API_SECRET=your_bitvavo_secret_here
BINANCE_API_KEY=your_binance_key_here
BINANCE_API_SECRET=your_binance_secret_here

# Logging Configuration
LOG_LEVEL=INFO

# MLflow Configuration
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=trading-bot

# Database Configuration
DATABASE_URL=sqlite:///trading_bot.db

# Monitoring
PROMETHEUS_PORT=8000
EOF
    print_success "Created .env template"
    print_warning "Please edit .env with your actual API keys and tokens"
else
    print_info ".env file already exists"
fi

# Create GitHub Actions CI workflow
if [ ! -f ".github/workflows/ci.yml" ]; then
    print_info "Creating GitHub Actions CI workflow..."
    cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run security checks
      run: |
        bandit -r src/ --skip B101,B601
        safety check

    - name: Run linting
      run: |
        flake8 src/ --max-line-length=100
        black --check src/ --line-length=100
        isort --check-only src/ --profile=black

    - name: Run type checks
      run: mypy src/ --ignore-missing-imports

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing -v

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
EOF
    print_success "Created GitHub Actions CI workflow"
fi

echo ""
print_info "Running environment validation..."

# Run environment validation
if python validate_environment.py; then
    print_success "Environment validation passed"
else
    print_warning "Environment validation found some issues (see above)"
fi

echo ""
print_info "Testing imports..."

# Test if our new modules can be imported
python -c "
try:
    from src.core.structured_logger import get_logger
    from src.core.advanced_circuit_breaker import circuit_breaker
    from src.core.resilience import retry, timeout, bulkhead
    from src.config.environment_validator import validate_startup_environment
    print('✅ All new modules import successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_success "All new modules are working correctly"
else
    print_error "Some modules failed to import"
fi

echo ""
echo "========================================"
echo "🎉 Setup Complete!"
echo "========================================"
echo ""
echo "📋 What's been added:"
echo "• Enhanced error handling with circuit breakers"
echo "• Structured logging with correlation IDs"
echo "• Comprehensive test framework"
echo "• Environment validation"
echo "• Code quality tools (pre-commit, linting, etc.)"
echo "• CI/CD pipeline configuration"
echo "• Performance monitoring capabilities"
echo ""
echo "🚀 Next steps:"
echo "1. Edit .env file with your API keys and tokens"
echo "2. Run demo: python demo_improvements.py"
echo "3. Run tests: pytest tests/"
echo "4. Validate environment: python validate_environment.py"
echo "5. Start trading: python scripts/trader.py"
echo ""
echo "📚 Key files created:"
echo "• validate_environment.py - Environment validation"
echo "• demo_improvements.py - Feature demonstration"
echo "• .env - Environment variables template"
echo "• pytest.ini - Test configuration"
echo "• .pre-commit-config.yaml - Code quality hooks"
echo "• .github/workflows/ci.yml - CI/CD pipeline"
echo ""

# Check if dependencies need to be installed
if ! python -c "import structlog, circuitbreaker, great_expectations" &> /dev/null; then
    print_warning "Some optional dependencies may not be installed."
    print_info "Run: pip install -r requirements.txt"
fi

print_success "Trading bot improvements setup complete!"
