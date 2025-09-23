#!/usr/bin/env python3
"""
Monitor Training Progress
========================

Monitor the progress of the superior PPO training.
"""

import os
import subprocess
import time


def check_training_progress():
    """Check the current training progress."""

    print("🔍 SUPERIOR PPO TRAINING MONITOR")
    print("=" * 50)

    # Check if training process is running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_real_superior_ppo"], capture_output=True, text=True
        )
        if result.stdout.strip():
            print("✅ Training process is running")
            print(f"   Process ID: {result.stdout.strip()}")
        else:
            print("❌ No training process found")
            return
    except:
        print("⚠️  Cannot check process status")

    # Check log file
    if os.path.exists("training_output.log"):
        print("\n📊 RECENT TRAINING OUTPUT:")
        print("-" * 30)

        # Get last 15 lines
        try:
            with open("training_output.log", "r") as f:
                lines = f.readlines()
                recent_lines = lines[-15:] if len(lines) > 15 else lines

                for line in recent_lines:
                    if any(
                        keyword in line
                        for keyword in [
                            "INFO",
                            "fps",
                            "iterations",
                            "total_timesteps",
                            "ep_rew_mean",
                        ]
                    ):
                        print(line.strip())

        except Exception as e:
            print(f"Error reading log: {e}")
    else:
        print("\n❌ No training log file found")

    # Check for model outputs
    if os.path.exists("models/superior/BTCEUR"):
        print("\n📁 MODEL DIRECTORIES:")
        try:
            result = subprocess.run(
                ["find", "models/superior/BTCEUR", "-type", "f"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                files = result.stdout.strip().split("\n")
                for f in files[:10]:  # Show first 10 files
                    print(f"   {f}")
                if len(files) > 10:
                    print(f"   ... and {len(files) - 10} more files")
            else:
                print("   No model files yet")
        except:
            print("   Cannot check model directory")

    print(f"\n🕒 Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTo continue monitoring, run: python monitor_training.py")
    print("To stop training: pkill -f train_real_superior_ppo")


if __name__ == "__main__":
    check_training_progress()
