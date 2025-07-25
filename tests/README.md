# Testing Infrastructure for Cryptocurrency Trading Bot

This directory contains comprehensive unit tests for the cryptocurrency trading bot project using pytest.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                   # Pytest configuration and fixtures
├── test_settings.py              # Tests for configuration management
├── test_portfolio_manager.py     # Tests for portfolio management
├── test_feature_engineer.py      # Tests for feature engineering
└── README.md                     # This file
```

## Running Tests

### All Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage (if pytest-cov is installed)
python -m pytest tests/ --cov=paper_trader --cov-report=html

# Using the test runner script
python run_tests.py
```

### Specific Test Modules
```bash
# Test configuration settings
python -m pytest tests/test_settings.py -v

# Test portfolio management
python -m pytest tests/test_portfolio_manager.py -v

# Test feature engineering
python -m pytest tests/test_feature_engineer.py -v

# Using the test runner script for specific modules
python run_tests.py settings
python run_tests.py portfolio_manager
python run_tests.py feature_engineer
```

### Test Categories
Tests are organized by functionality:

- **Configuration Tests** (`test_settings.py`): Test environment variable loading, validation, and configuration management
- **Portfolio Tests** (`test_portfolio_manager.py`): Test position management, trade execution, P&L calculation, and risk management
- **Feature Engineering Tests** (`test_feature_engineer.py`): Test technical indicators, data preprocessing, and model feature preparation

## Test Configuration

The tests use `pytest.ini` configuration file with the following settings:

- **Test Discovery**: Automatically finds test files matching `test_*.py` pattern
- **Logging**: Enabled with INFO level for debugging
- **Markers**: Support for categorizing tests (unit, integration, slow, etc.)
- **Coverage**: Ready for code coverage analysis
- **Warnings**: Filtered to reduce noise from dependencies

## Fixtures and Test Data

### Common Fixtures (in `conftest.py`)
- `temp_dir`: Temporary directory for test files
- `mock_portfolio_settings`: Sample portfolio configuration
- `sample_trade_data`: Sample completed trade data
- `sample_position_data`: Sample open position data

### Environment Setup
Tests automatically set up a test environment with:
- Mock API credentials to prevent real API calls
- Temporary directories for file operations
- Consistent random seeds for reproducible tests

## Test Categories and Markers

Tests can be marked with categories for selective execution:

```bash
# Run only unit tests (fast, isolated)
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Run only portfolio-related tests
python -m pytest -m portfolio

# Skip slow tests
python -m pytest -m "not slow"
```

## Adding New Tests

### Test File Structure
```python
"""Test module description."""

import pytest
from your_module import YourClass

class TestYourClass:
    """Test cases for YourClass."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        instance = YourClass()
        assert instance.method() == expected_result
    
    def test_error_handling(self):
        """Test error handling."""
        instance = YourClass()
        with pytest.raises(ExpectedError):
            instance.problematic_method()
```

### Using Fixtures
```python
def test_with_fixture(self, temp_dir, sample_trade_data):
    """Test using common fixtures."""
    # Use temporary directory
    file_path = temp_dir / 'test_file.csv'
    
    # Use sample data
    trade = Trade(**sample_trade_data)
    assert trade.symbol == 'BTCEUR'
```

### Testing Async Code
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality."""
    result = await some_async_function()
    assert result is not None
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **Isolation**: Each test should be independent and not rely on other tests
4. **Fixtures**: Use fixtures for common test data and setup
5. **Assertions**: Use clear, specific assertions with meaningful error messages
6. **Coverage**: Aim for high test coverage of critical business logic
7. **Performance**: Keep unit tests fast; mark slow tests appropriately

## Continuous Integration

The test suite is designed to run in CI/CD environments:

- All tests should pass on Python 3.8+
- No external dependencies (APIs, databases) required for unit tests
- Tests use temporary files and directories that are cleaned up automatically
- Environment variables are mocked for testing

## Known Limitations

- Feature engineering tests require specific data formats and minimum data sizes
- Some ML model tests may be skipped if models are not available
- API integration tests are mocked to avoid real API calls
- Performance tests may vary based on system resources

## Dependencies

Required testing dependencies are listed in `requirements_paper_trader.txt`:
- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`

Optional but recommended:
- `pytest-cov` for coverage reporting
- `pytest-xdist` for parallel test execution
- `pytest-mock` for advanced mocking capabilities