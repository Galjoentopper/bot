#!/usr/bin/env python3
"""
Test script to run enhanced_trader.py for exactly 5 minutes then kill it
(Following user rule: run scripts for max 5 minutes then kill)
"""
import subprocess
import time
import sys
import os
from pathlib import Path

def test_enhanced_trader():
    """Run enhanced_trader.py for 5 minutes then terminate."""
    script_path = Path(__file__).parent / "scripts" / "enhanced_trader.py"
    
    print("Starting enhanced_trader.py for 5-minute validation test...")
    print(f"Script path: {script_path}")
    
    # Start the enhanced trader process
    try:
        process = subprocess.Popen([
            sys.executable, str(script_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        print(f"Enhanced trader started with PID: {process.pid}")
        
        # Wait for exactly 5 minutes (300 seconds)
        timeout_seconds = 300
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check if process is still running
            if process.poll() is not None:
                print("Process terminated early")
                break
            
            # Show progress every 30 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"Running for {int(elapsed)} seconds...")
                
            time.sleep(1)
        
        # Terminate the process after 5 minutes
        if process.poll() is None:
            print("5 minutes elapsed - terminating enhanced_trader.py")
            process.terminate()
            
            # Give it 10 seconds to terminate gracefully
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing process...")
                process.kill()
                process.wait()
        
        # Get the output
        stdout, stderr = process.communicate()
        
        print("\n=== PROCESS COMPLETED ===")
        print(f"Return code: {process.returncode}")
        print(f"Runtime: {time.time() - start_time:.1f} seconds")
        
        if stdout:
            print("\n=== STDOUT (last 2000 chars) ===")
            print(stdout[-2000:] if len(stdout) > 2000 else stdout)
        
        if stderr:
            print("\n=== STDERR (last 1000 chars) ===")
            print(stderr[-1000:] if len(stderr) > 1000 else stderr)
        
        return process.returncode == 0 or process.returncode == -15  # 0 = success, -15 = terminated (OK)
        
    except Exception as e:
        print(f"Error running enhanced_trader.py: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_trader()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)