#!/usr/bin/env python3
"""
Simple test runner for the cryptocurrency trading bot.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests  
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Run with verbose output
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return whether it succeeded."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run tests for the trading bot')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add test selection
    if args.unit:
        cmd.extend(['-m', 'unit'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(['--cov=paper_trader', '--cov-report=html'])
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    # Skip slow tests if requested
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    
    # Run pre-test checks
    print("🧪 Trading Bot Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('paper_trader').exists():
        print("❌ Error: paper_trader directory not found. Please run from the project root.")
        sys.exit(1)
    
    # Check if tests directory exists
    if not Path('tests').exists():
        print("❌ Error: tests directory not found.")
        sys.exit(1)
    
    print("✅ Pre-test checks passed")
    
    # Run the tests
    success = run_command(cmd, "Test Suite")
    
    if success:
        print("\n🎉 All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
            
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()