#!/usr/bin/env python3
"""
Backtesting System Entry Point

This script provides a unified entry point for running backtests using the
comprehensive backtesting infrastructure located in the scripts/ directory.

Usage:
    python run_backtest.py

For more advanced backtesting options, use the scripts directly:
    python scripts/backtest_models.py      # Run model backtests
    python scripts/backtest_analysis.py    # Analyze backtest results
    python scripts/bootstrap_backtest.py   # Bootstrap backtesting
"""

import sys
import os

def main():
    """Main entry point for backtesting system"""
    print("🔄 Cryptocurrency Trading Bot - Backtesting System")
    print("=" * 50)
    print()
    print("Available backtesting scripts:")
    print("1. backtest_models.py      - Run comprehensive model backtests")
    print("2. backtest_analysis.py    - Analyze and visualize backtest results")
    print("3. bootstrap_backtest.py   - Statistical bootstrap backtesting")
    print("4. backtest_config.py      - Backtest configuration utilities")
    print()
    
    # Check if scripts directory exists
    scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    if not os.path.exists(scripts_dir):
        print("❌ Error: scripts/ directory not found")
        return 1
    
    # Default to running the main backtest models script
    backtest_script = os.path.join(scripts_dir, 'backtest_models.py')
    
    if os.path.exists(backtest_script):
        print(f"🚀 Running: {backtest_script}")
        print("=" * 50)
        
        # Execute the backtest script
        import subprocess
        result = subprocess.run([sys.executable, backtest_script], 
                              cwd=os.path.dirname(__file__))
        return result.returncode
    else:
        print(f"❌ Error: {backtest_script} not found")
        print()
        print("Available scripts in scripts/ directory:")
        for file in os.listdir(scripts_dir):
            if file.endswith('.py'):
                print(f"  - {file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())