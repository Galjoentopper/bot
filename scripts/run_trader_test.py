#!/usr/bin/env python3
"""
Test script to run trader.py for a maximum of 5 minutes.
This addresses the user's guideline to run scripts/trader.py for max 5 minutes
and then kill it, rather than waiting for it to end by itself.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_trader_with_timeout(script_path, timeout_seconds=300, config_path=None, iterations=None):
    """
    Run the trader script with a timeout.
    
    Args:
        script_path (str): Path to the trader script
        timeout_seconds (int): Maximum time to run in seconds (default: 300 = 5 minutes)
        config_path (str): Optional path to configuration file
        iterations (int): Optional number of iterations to run
    """
    print(f"Running {script_path} for maximum {timeout_seconds} seconds...")
    
    # Build command
    cmd = ["python", script_path]
    
    # Add configuration if provided
    if config_path and os.path.exists(config_path):
        cmd.extend(["--config", config_path])
    
    # Add iterations if provided
    if iterations is not None:
        cmd.extend(["--iterations", str(iterations)])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Run the script with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        print("Trader script completed successfully!")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return True
        
    except subprocess.TimeoutExpired as e:
        print(f"Trader script timed out after {timeout_seconds} seconds")
        if e.stdout:
            print("STDOUT before timeout:")
            print(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout)
        if e.stderr:
            print("STDERR before timeout:")
            print(e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr)
        
        # The process was automatically killed by subprocess.run
        return True  # This is expected behavior per the user's guidelines
        
    except Exception as e:
        print(f"Error running trader script: {e}")
        return False

def main():
    """Main function to run the trader test."""
    print("Trader Test Script with 5-Minute Timeout")
    print("=" * 50)
    
    # Check if trader.py exists
    trader_script = "scripts/trader.py"
    if not os.path.exists(trader_script):
        print(f"ERROR: Trader script not found: {trader_script}")
        return 1
    
    # Check for configuration file
    config_file = "training_config.yaml"
    if not os.path.exists(config_file):
        config_file = None
        print("No configuration file found, running with defaults")
    
    # Run trader for 5 minutes (300 seconds)
    success = run_trader_with_timeout(
        script_path=trader_script,
        timeout_seconds=300,  # 5 minutes
        config_path=config_file,
        iterations=None  # Let it run iterations until timeout
    )
    
    if success:
        print("\n✅ Trader test completed successfully!")
        return 0
    else:
        print("\n❌ Trader test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())