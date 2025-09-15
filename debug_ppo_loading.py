#!/usr/bin/env python3
"""
Debug PPO loading with different threading contexts and device settings
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def test_basic_load():
    """Test basic PPO loading"""
    print("=== Basic PPO Loading Test ===")
    try:
        from stable_baselines3 import PPO

        model_path = "models/ppo/BTCEUR/model.zip"

        print(f"Loading {model_path}...")
        start = time.time()

        # Try with explicit CPU device
        model = PPO.load(model_path, device="cpu")

        load_time = time.time() - start
        print(f"✅ Model loaded successfully in {load_time:.2f}s")

        # Test basic prediction
        import numpy as np

        obs = np.random.random((1, 104)).astype(np.float32)
        action, _ = model.predict(obs)
        print(f"✅ Prediction successful: {action}")

        return True

    except Exception as e:
        print(f"❌ Basic load failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_threaded_load():
    """Test PPO loading in different threading contexts"""
    print("\n=== Threaded PPO Loading Test ===")

    def load_in_thread():
        try:
            from stable_baselines3 import PPO

            model_path = "models/ppo/BTCEUR/model.zip"

            print(f"Thread {threading.current_thread().name}: Loading model...")
            model = PPO.load(model_path, device="cpu")
            print(f"Thread {threading.current_thread().name}: ✅ Success")
            return model
        except Exception as e:
            print(f"Thread {threading.current_thread().name}: ❌ Failed: {e}")
            raise

    # Test 1: Single thread with timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_in_thread)
            model = future.result(timeout=30)  # 30 second timeout
            print("✅ Threaded loading successful")
    except TimeoutError:
        print("❌ Threaded loading timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Threaded loading failed: {e}")
        return False

    # Test 2: Multiple threads (simulating real environment)
    print("\n--- Testing with multiple threads ---")

    def dummy_work():
        """Simulate other work happening simultaneously"""
        import time

        for i in range(5):
            time.sleep(1)
            print(f"Background work: {i+1}/5")

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Start background work
            bg_future = executor.submit(dummy_work)

            # Load model in parallel
            model_future = executor.submit(load_in_thread)

            model = model_future.result(timeout=30)
            print("✅ Multi-threaded loading successful")

            # Wait for background work to complete
            bg_future.result()

    except TimeoutError:
        print("❌ Multi-threaded loading timed out")
        return False
    except Exception as e:
        print(f"❌ Multi-threaded loading failed: {e}")
        return False

    return True


def test_environment_context():
    """Test PPO loading in simulated trading environment context"""
    print("\n=== Trading Environment Context Test ===")

    try:
        # Import trading-specific modules to simulate real context
        import numpy as np
        import pandas as pd

        # Simulate having other models loaded (memory pressure)
        print("Creating memory pressure...")
        dummy_data = [np.random.random((1000, 100)) for _ in range(10)]

        # Load PPO in this context
        from stable_baselines3 import PPO

        model_path = "models/ppo/BTCEUR/model.zip"

        print("Loading PPO in trading context...")
        start = time.time()
        model = PPO.load(model_path, device="cpu")
        load_time = time.time() - start

        print(f"✅ Context loading successful in {load_time:.2f}s")

        # Clean up
        del dummy_data

        return True

    except Exception as e:
        print(f"❌ Context loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all PPO loading tests"""
    print("PPO Loading Debug Tests")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Active threads: {threading.active_count()}")

    results = []

    # Test 1: Basic loading
    results.append(test_basic_load())

    # Test 2: Threaded loading
    results.append(test_threaded_load())

    # Test 3: Trading context loading
    results.append(test_environment_context())

    print("\n" + "=" * 50)
    print("SUMMARY:")
    test_names = ["Basic Load", "Threaded Load", "Context Load"]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")

    overall = "✅ ALL TESTS PASSED" if all(results) else "❌ SOME TESTS FAILED"
    print(f"\nOverall: {overall}")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
