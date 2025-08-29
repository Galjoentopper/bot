#!/usr/bin/env python3
"""
Test script to validate deploy_trader.bat functionality on Linux
This simulates the Windows batch script logic for testing purposes.
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def test_config_extraction():
    """Test configuration file extraction."""
    print("=== Testing Configuration Extraction ===")
    
    config_file = "training_config.yaml"
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file not found: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract symbols
        symbols = config.get('data_acquisition', {}).get('symbols', [])
        if not symbols:
            symbols = config.get('data', {}).get('symbols', [])
        if not symbols:
            symbols = config.get('symbols', [])
            
        # Extract models
        models = config.get('training', {}).get('models', [])
        if not models:
            models = ['gru', 'lightgbm', 'ppo']  # defaults
        
        print(f"✓ Symbols found: {symbols}")
        print(f"✓ Models found: {models}")
        return symbols, models
    except Exception as e:
        print(f"ERROR: Failed to extract configuration: {e}")
        return False

def test_model_verification(symbols, models):
    """Test model verification logic."""
    print("\n=== Testing Model Verification ===")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("WARNING: Models directory not found")
        print("This is expected in a test environment - enhanced_trader.py will handle this")
        return [], []
    
    verified_symbols = []
    verified_models = set()
    
    for symbol in symbols:
        available_models = []
        for model_type in models:
            model_path = models_dir / model_type / symbol
            if model_path.exists():
                available_models.append(model_type)
                verified_models.add(model_type)
        
        if available_models:
            verified_symbols.append(symbol)
            print(f"✓ {symbol}: {available_models}")
        else:
            print(f"✗ {symbol}: no models available")
    
    print(f"\nVerified symbols: {verified_symbols}")
    print(f"Available models: {list(verified_models)}")
    return verified_symbols, list(verified_models)

def test_trader_invocation(symbols, models):
    """Test trader script invocation."""
    print("\n=== Testing Trader Invocation ===")
    
    trader_script = "scripts/enhanced_trader.py"
    if not os.path.exists(trader_script):
        print(f"ERROR: Trader script not found: {trader_script}")
        return False
    
    # Test with --help first
    try:
        result = subprocess.run([
            "python", trader_script, "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Trader script responds to --help")
        else:
            print(f"✗ Trader script --help failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Trader script --help timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing trader script: {e}")
        return False
    
    # Test with test mode
    try:
        cmd = [
            "python", trader_script,
            "--config", "training_config.yaml",
            "--symbols"
        ] + symbols + [
            "--models"
        ] + models + [
            "--test-mode"
        ]
        
        print(f"Testing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Trader test mode completed successfully")
            print("Sample output:")
            print("  " + "\n  ".join(result.stdout.split('\n')[-10:]))
            return True
        else:
            print(f"✗ Trader test mode failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Trader test mode timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing trader invocation: {e}")
        return False

def main():
    """Run comprehensive deploy_trader.bat functionality test."""
    print("Deploy Trader Script Functionality Test")
    print("=" * 50)
    
    # Test 1: Configuration extraction
    config_result = test_config_extraction()
    if not config_result:
        print("\n❌ Configuration extraction failed")
        return 1
    
    symbols, models = config_result
    
    # Test 2: Model verification
    verification_result = test_model_verification(symbols, models)
    verified_symbols, verified_models = verification_result
    
    if not verified_symbols:
        print("\n⚠️  No symbols have available models")
        print("This is expected in a test environment without trained models")
        print("Testing trader invocation with all symbols to verify error handling...")
        
        # Test trader with all symbols to verify it handles missing models gracefully
        if test_trader_invocation(symbols, models):
            print("\n✅ Deploy trader logic works correctly!")
            print("   Script handles missing models gracefully")
            print("   Enhanced trader will show available models and exit cleanly")
            return 0
        else:
            print("\n❌ Trader invocation failed")
            return 1
    
    # Test 3: Trader invocation with verified symbols
    if not test_trader_invocation(verified_symbols, verified_models):
        print("\n❌ Trader invocation failed")
        return 1
    
    print("\n✅ All deploy_trader.bat functionality tests passed!")
    print(f"   Ready for deployment with {len(verified_symbols)} symbols and {len(verified_models)} model types")
    return 0

if __name__ == "__main__":
    sys.exit(main())