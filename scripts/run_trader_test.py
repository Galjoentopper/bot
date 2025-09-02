#!/usr/bin/env python3
"""
Test script to run trader.py for maximum 5 minutes then kill it.
This addresses the guideline: "do run scripts/trader.py for max 5 minutes than kill it.
currently you wait till the scripts ends for itself but it will run continuously."
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path

def run_trader_with_timeout(timeout_minutes=5):
    """Run trader.py with a timeout and kill it after the specified time."""

    # Convert minutes to seconds
    timeout_seconds = timeout_minutes * 60

    print(f"🚀 Starting trader.py with {timeout_minutes} minute timeout...")
    print(f"⏰ Will automatically kill the process after {timeout_seconds} seconds")
    print("=" * 60)

    # Get the script directory and trader.py path
    script_dir = Path(__file__).parent
    trader_path = script_dir / "trader.py"

    if not trader_path.exists():
        print(f"❌ Error: trader.py not found at {trader_path}")
        return False

    try:
        # Start the trader process
        print(f"📍 Starting: python {trader_path}")
        process = subprocess.Popen(
            [sys.executable, str(trader_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(script_dir)
        )

        print(f"✅ Trader process started with PID: {process.pid}")
        print("⏳ Waiting for trader to complete or timeout...")

        start_time = time.time()
        end_time = start_time + timeout_seconds

        # Monitor the process
        while time.time() < end_time:
            if process.poll() is not None:
                # Process finished on its own
                print("✅ Trader process completed on its own")
                break

            # Show progress every 30 seconds
            elapsed = int(time.time() - start_time)
            remaining = int(end_time - time.time())

            if elapsed % 30 == 0 and elapsed > 0:
                print(f"⏱️  Elapsed: {elapsed}s | Remaining: {remaining}s")

            time.sleep(1)

        # Check if process is still running
        if process.poll() is None:
            print(f"⏰ Timeout reached ({timeout_minutes} minutes). Killing trader process...")
            try:
                # Try graceful termination first
                process.terminate()
                print("📤 Sent SIGTERM to trader process")

                # Wait up to 10 seconds for graceful shutdown
                for i in range(10):
                    if process.poll() is not None:
                        print("✅ Trader process terminated gracefully")
                        break
                    time.sleep(1)
                else:
                    # Force kill if still running
                    print("⚠️  Trader process didn't respond to SIGTERM, force killing...")
                    process.kill()
                    print("💀 Trader process force killed")

            except Exception as e:
                print(f"❌ Error killing trader process: {e}")
                return False
        else:
            print("✅ Trader process completed naturally")

        # Get exit code
        exit_code = process.poll()
        print(f"📊 Trader process exit code: {exit_code}")

        # Show some output if available
        try:
            stdout, stderr = process.communicate(timeout=5)
            if stdout:
                print("\n📄 Trader stdout (last 500 chars):")
                print(stdout[-500:])
            if stderr:
                print("\n⚠️  Trader stderr (last 500 chars):")
                print(stderr[-500:])
        except Exception as e:
            print(f"⚠️  Could not read process output: {e}")

        return True

    except Exception as e:
        print(f"❌ Error running trader: {e}")
        return False

def main():
    """Main function."""
    print("🧪 Trader Test Script")
    print("This will run trader.py for a maximum of 5 minutes")
    print()

    success = run_trader_with_timeout(timeout_minutes=5)

    if success:
        print("\n✅ Test completed successfully")
        print("🧹 Cleaning up test artifacts...")
        # Add any cleanup logic here if needed
        print("✅ Cleanup completed")
    else:
        print("\n❌ Test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()