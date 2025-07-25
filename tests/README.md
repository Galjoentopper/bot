# Testing Guide for Cryptocurrency Trading Bot

This document explains how to run and understand the test suite for the trading bot.

## Overview

The test suite validates core functionality of the trading bot without requiring external API connections or live data. Tests are organized into several categories:

- **Configuration Tests** (`test_config.py`): Validate settings loading and environment variable handling
- **Portfolio Tests** (`test_portfolio.py`): Test portfolio management, position tracking, and P&L calculations
- **Data Quality Tests** (`test_data_quality.py`): Validate data quality monitoring and validation logic
- **Integration Tests** (`test_integration.py`): Test imports, environment setup, and basic functionality

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Or use pytest directly
python -m pytest tests/
```

### Test Options

```bash
# Run with verbose output
python run_tests.py --verbose

# Run with coverage report
python run_tests.py --coverage

# Run only fast tests (skip slow ones)
python run_tests.py --fast

# Run specific test files
python -m pytest tests/test_config.py -v
python -m pytest tests/test_portfolio.py -v
```

### Test Categories

```bash
# Run only unit tests
python run_tests.py --unit

# Run only integration tests  
python run_tests.py --integration
```

## Test Structure

### Configuration Tests
- Validate default values loading
- Test environment variable override mechanism
- Check risk management parameter validation
- Verify model path generation

### Portfolio Management Tests
- Test Position and Trade data classes
- Validate portfolio initialization
- Test P&L calculation logic
- Check position limits and capital tracking

### Data Quality Tests
- Test data quality issue and report structures
- Validate data completeness checking
- Test price and volume validation logic
- Check outlier and gap detection algorithms

### Integration Tests
- Verify all modules can be imported
- Check required dependencies are available
- Test basic object creation without external dependencies
- Validate environment setup

## What Tests Don't Cover

To keep tests simple and avoid external dependencies, the following are NOT tested:

- ❌ Live API calls to Bitvavo exchange
- ❌ Telegram notification sending
- ❌ WebSocket connections
- ❌ Real machine learning model predictions
- ❌ File I/O operations (mocked instead)

## Test Data

Tests use:
- **Mock data**: Generated DataFrames for testing algorithms
- **Mock objects**: Patched dependencies to avoid external calls
- **Sample configurations**: Isolated test environments

## Coverage Reports

When running with `--coverage`, an HTML coverage report is generated in `htmlcov/index.html`.

## Continuous Integration

Tests should pass on:
- Python 3.8+
- With or without GPU support
- Without requiring API credentials
- Without external network access

## Adding New Tests

When adding new functionality:

1. Create corresponding test files in `tests/`
2. Use `pytest` conventions (test files start with `test_`)
3. Mock external dependencies
4. Keep tests fast and isolated
5. Add appropriate test markers if needed

```python
import pytest

@pytest.mark.unit
def test_my_function():
    """Test description."""
    assert my_function() == expected_result

@pytest.mark.slow
def test_expensive_operation():
    """Mark slow tests appropriately."""
    pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **TensorFlow Warnings**: GPU-related warnings are normal and don't affect test results
3. **Environment Variables**: Tests account for existing `.env` file configuration

### Test Failures

If tests fail:
1. Check the error message and stack trace
2. Verify all dependencies are installed
3. Ensure you're running from the project root directory
4. Check if `.env` file exists and has valid configuration

## Test Philosophy

The test suite follows these principles:

- **Fast**: Tests should run quickly for rapid feedback
- **Isolated**: No external dependencies or API calls
- **Focused**: Each test validates a specific piece of functionality
- **Realistic**: Use realistic data and scenarios while remaining deterministic
- **Maintainable**: Clear test names and structure for easy maintenance