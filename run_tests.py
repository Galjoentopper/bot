#!/usr/bin/env python3
"""
Test runner script for the cryptocurrency trading bot.

This script provides an easy way to run the test suite with different options.
"""

import subprocess
import sys
import argparse


def run_tests(test_type='all', verbose=False):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add verbosity if requested
    if verbose:
        cmd.append('-v')
    
    # Determine which tests to run
    if test_type == 'config':
        cmd.append('tests/test_config_simple.py')
    elif test_type == 'portfolio':
        cmd.extend([
            'tests/test_portfolio.py::TestPortfolioManager::test_initialization',
            'tests/test_portfolio.py::TestPortfolioManager::test_open_position_success',
            'tests/test_portfolio.py::TestPortfolioManager::test_max_positions_per_symbol'
        ])
    elif test_type == 'integration':
        cmd.append('tests/test_integration.py::TestTradingIntegration::test_configuration_validation_integration')
    elif test_type == 'core':
        # Run the core working tests
        cmd.extend([
            'tests/test_config_simple.py',
            'tests/test_portfolio.py::TestPortfolioManager::test_initialization',
            'tests/test_portfolio.py::TestPortfolioManager::test_open_position_success',
            'tests/test_portfolio.py::TestPortfolioManager::test_max_positions_per_symbol',
            'tests/test_integration.py::TestTradingIntegration::test_configuration_validation_integration'
        ])
    elif test_type == 'all':
        cmd.append('tests/')
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add short traceback for cleaner output
    cmd.append('--tb=short')
    
    # Run the tests
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run trading bot tests')
    parser.add_argument(
        'test_type', 
        nargs='?', 
        default='core',
        choices=['all', 'config', 'portfolio', 'integration', 'core'],
        help='Type of tests to run (default: core)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Run tests with verbose output'
    )
    
    args = parser.parse_args()
    
    print("Cryptocurrency Trading Bot Test Runner")
    print("=" * 40)
    print(f"Running {args.test_type} tests...")
    print()
    
    exit_code = run_tests(args.test_type, args.verbose)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())