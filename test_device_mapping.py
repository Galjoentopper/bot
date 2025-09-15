#!/usr/bin/env python3
"""
Test PPO loading with explicit device mapping
"""
import torch
from stable_baselines3 import PPO


def test_device_mapped_loading():
    """Test loading with explicit device mapping"""
    print("Testing device-mapped PPO loading...")

    model_path = "models/ppo/BTCEUR/model.zip"

    try:
        # Force CPU device mapping
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Method 1: Use map_location in load (if supported)
        print("Attempting load with device='cpu'...")
        model = PPO.load(model_path, device="cpu")
        print("✅ Success with device='cpu'")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


if __name__ == "__main__":
    test_device_mapped_loading()
