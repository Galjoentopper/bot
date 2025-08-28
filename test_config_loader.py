#!/usr/bin/env python3
"""
Isolated test for ConfigLoader component
Tests configuration loading and symbol resolution
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.config_loader import ConfigLoader

def test_config_loader():
    """Test ConfigLoader functionality in isolation"""
    print("=== ConfigLoader Component Test ===")
    
    try:
        # Test 1: Basic config loading
        print("\n1. Testing basic config loading...")
        config = ConfigLoader()
        print(f"   Config path detected: {config._raw_path}")
        print(f"   Config loaded successfully: {config.config is not None}")
        print(f"   Config keys: {list(config.config.keys()) if config.config else 'None'}")
        
        # Test 2: Symbol configuration
        print("\n2. Testing symbol configuration...")
        if 'data' in config.config and 'symbols' in config.config['data']:
            symbols = config.config['data']['symbols']
            print(f"   Symbols from config: {symbols}")
            print(f"   ADAEUR in symbols: {'ADAEUR' in symbols}")
        else:
            print("   WARNING: No data.symbols found in config")
            print(f"   Available config sections: {list(config.config.keys())}")
            
        # Test 3: Config file existence check
        print("\n3. Testing config file existence...")
        config_files = [
            'src/config/config_trading.yaml',
            'config.yaml',
            'src/config/config_training.yaml'
        ]
        
        for config_file in config_files:
            file_path = project_root / config_file
            exists = file_path.exists()
            print(f"   {config_file}: {'EXISTS' if exists else 'MISSING'}")
            if exists:
                print(f"     Size: {file_path.stat().st_size} bytes")
                
        # Test 4: Manual config_trading.yaml loading
        print("\n4. Testing manual config_trading.yaml loading...")
        trading_config_path = project_root / 'src' / 'config' / 'config_trading.yaml'
        if trading_config_path.exists():
            try:
                import yaml
                with open(trading_config_path, 'r') as f:
                    trading_config = yaml.safe_load(f)
                
                if 'data' in trading_config and 'symbols' in trading_config['data']:
                    manual_symbols = trading_config['data']['symbols']
                    print(f"   Manual load symbols: {manual_symbols}")
                    print(f"   ADAEUR in manual symbols: {'ADAEUR' in manual_symbols}")
                else:
                    print("   WARNING: No data.symbols in manual load")
            except Exception as e:
                print(f"   ERROR in manual load: {e}")
        else:
            print("   config_trading.yaml not found for manual load")
            
        print("\n=== ConfigLoader Test Complete ===")
        return True
        
    except Exception as e:
        print(f"\nERROR in ConfigLoader test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_loader()
    sys.exit(0 if success else 1)