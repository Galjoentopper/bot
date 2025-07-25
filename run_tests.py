#!/usr/bin/env python3
"""
Simple test runner script for the cryptocurrency trading bot.
This script runs all tests and provides a summary.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests and provide summary."""
    print("🧪 Running Cryptocurrency Trading Bot Tests")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run pytest with detailed output
    cmd = [
        sys.executable, '-m', 'pytest', 
        'tests/', 
        '-v', 
        '--tb=short',
        '--color=yes',
        '--durations=10'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("📋 Test Output:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Some tests failed (exit code: {result.returncode})")
            
        return result.returncode
        
    except Exception as e:
        print(f"💥 Error running tests: {e}")
        return 1

def run_specific_module(module_name):
    """Run tests for a specific module."""
    print(f"🧪 Running tests for {module_name}")
    print("=" * 50)
    
    cmd = [
        sys.executable, '-m', 'pytest', 
        f'tests/test_{module_name}.py', 
        '-v', 
        '--tb=short'
    ]
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"💥 Error running tests: {e}")
        return 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific module tests
        module = sys.argv[1]
        exit_code = run_specific_module(module)
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)