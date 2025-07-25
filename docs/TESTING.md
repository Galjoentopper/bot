# Testing Guide for Cryptocurrency Trading Bot

This guide explains how to run and understand the tests for the cryptocurrency trading bot.

## Test Structure

The test suite is organized into several modules:

- `test_config_simple.py` - Tests for configuration management and settings validation
- `test_portfolio.py` - Tests for portfolio management, position tracking, and P&L calculations
- `test_integration.py` - Integration tests for complete trading workflows

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install pytest pytest-mock pandas python-dotenv
```

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Module
```bash
# Configuration tests
python -m pytest tests/test_config_simple.py -v

# Portfolio management tests
python -m pytest tests/test_portfolio.py -v

# Integration tests
python -m pytest tests/test_integration.py -v
```

### Run Specific Test
```bash
python -m pytest tests/test_portfolio.py::TestPortfolioManager::test_open_position_success -v
```

### Run Tests with Coverage (if pytest-cov is installed)
```bash
pip install pytest-cov
python -m pytest tests/ --cov=paper_trader --cov-report=html
```

## Test Categories

### 1. Configuration Tests (`test_config_simple.py`)

Tests the `TradingSettings` class which manages configuration from environment variables:

- **Environment variable loading**: Verifies settings load correctly from .env file
- **Default values**: Ensures reasonable defaults when no env vars are set
- **Validation**: Tests validation logic for required fields and constraints
- **Type conversion**: Verifies proper parsing of strings to numbers and booleans

Key test methods:
- `test_trading_settings_import()` - Basic import functionality
- `test_validation_with_valid_credentials()` - Configuration validation
- `test_position_size_calculation()` - Position sizing logic

### 2. Portfolio Management Tests (`test_portfolio.py`)

Tests the `PortfolioManager` class which handles positions and trading:

- **Position management**: Opening and closing positions
- **P&L calculations**: Profit and loss tracking
- **Risk management**: Capital limits and position constraints
- **Portfolio tracking**: Capital allocation and performance metrics

Key test methods:
- `test_open_position_success()` - Opening trading positions
- `test_close_position_profit()` - Closing profitable positions
- `test_get_portfolio_summary()` - Portfolio performance summaries
- `test_drawdown_calculation()` - Risk management metrics

### 3. Integration Tests (`test_integration.py`)

Tests complete workflows that combine multiple components:

- **Full trading cycle**: Configuration → Position opening → Position closing
- **Risk management integration**: Testing limits and constraints across components
- **Portfolio summary integration**: End-to-end portfolio tracking

Key test methods:
- `test_basic_trading_workflow()` - Complete trading cycle
- `test_risk_management_integration()` - Risk limits enforcement
- `test_portfolio_summary_integration()` - Multi-position portfolio management

## Test Features

### Mocking and Isolation

Tests use proper mocking to:
- Isolate components from external dependencies (APIs, file system)
- Prevent interference from actual `.env` configuration files
- Create predictable test environments

### Temporary Directories

Portfolio tests use temporary directories to avoid cluttering the filesystem with test logs.

### Realistic Data

Tests use realistic cryptocurrency prices and trading scenarios to ensure the system works with actual market data patterns.

## Understanding Test Results

### Successful Test Run
```
tests/test_config_simple.py::TestTradingSettings::test_trading_settings_import PASSED
tests/test_portfolio.py::TestPortfolioManager::test_initialization PASSED
tests/test_integration.py::TestTradingIntegration::test_basic_trading_workflow PASSED
```

### Failed Test
```
FAILED tests/test_portfolio.py::TestPortfolioManager::test_open_position_success
AssertionError: assert False is True
```

## Common Issues and Solutions

### Import Errors
If you see `ModuleNotFoundError`, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install pytest pytest-mock pandas python-dotenv
```

### Environment Variable Conflicts
Tests may fail if they pick up actual environment variables. The tests are designed to isolate themselves, but if issues persist:
1. Temporarily rename your `.env` file during testing
2. Run tests in a clean environment

### Test Data Isolation
Each test creates its own temporary data to avoid conflicts. If tests are interfering with each other, check that proper cleanup is happening.

## Writing New Tests

When adding new functionality, follow these patterns:

### 1. Unit Tests
```python
def test_new_feature(self):
    """Test description."""
    # Arrange - set up test data
    # Act - execute the functionality
    # Assert - verify the results
```

### 2. Use Fixtures
```python
@pytest.fixture
def mock_portfolio(self):
    """Create a mock portfolio for testing."""
    return PortfolioManager(initial_capital=10000.0)
```

### 3. Mock External Dependencies
```python
@patch('external_api.get_price')
def test_with_mocked_api(self, mock_get_price):
    mock_get_price.return_value = 50000.0
    # Your test code here
```

## Test Maintenance

- Keep tests focused and independent
- Update tests when adding new features
- Remove obsolete tests when removing features
- Ensure tests run quickly (< 1 second per test typically)
- Use descriptive test names that explain what is being tested

## Continuous Integration

These tests are designed to run in CI/CD environments. They:
- Don't require external network access
- Clean up after themselves
- Have predictable, deterministic results
- Run quickly for fast feedback cycles