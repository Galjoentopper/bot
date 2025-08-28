#!/usr/bin/env python3
"""
Isolated test for FeatureEngine component
Tests feature alignment, padding, and schema validation
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.features import FeatureEngine

def create_test_data():
    """Create sample market data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(150, 250, n_samples),
        'low': np.random.uniform(50, 150, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.uniform(-5, 5, n_samples),
        'bb_upper': np.random.uniform(180, 220, n_samples),
        'bb_lower': np.random.uniform(80, 120, n_samples),
        'sma_20': np.random.uniform(90, 210, n_samples)
    }
    
    return pd.DataFrame(data)

def test_feature_engine():
    """Test FeatureEngine functionality in isolation"""
    print("=== FeatureEngine Component Test ===")
    
    try:
        # Test 1: Basic initialization
        print("\n1. Testing FeatureEngine initialization...")
        feature_engine = FeatureEngine()
        print(f"   FeatureEngine created: {feature_engine is not None}")
        
        # Test 2: Feature mapping loading
        print("\n2. Testing feature mapping loading...")
        mapping_file = project_root / 'feature_mapping.json'
        if mapping_file.exists():
            print(f"   Feature mapping file exists: {mapping_file}")
            print(f"   File size: {mapping_file.stat().st_size} bytes")
            
            # Try to load and inspect mapping
            try:
                import json
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                
                print(f"   Mapping keys: {list(mapping.keys())}")
                
                # Check for symbol-model combinations
                test_keys = ['BTCEUR_gru', 'ETHEUR_lightgbm', 'ADAEUR_ppo']
                for key in test_keys:
                    if key in mapping:
                        entry = mapping[key]
                        print(f"   {key}: expected_features={entry.get('expected_features', 'N/A')}")
                        if 'required_features' in entry:
                            print(f"     required_features count: {len(entry['required_features'])}")
                    else:
                        print(f"   {key}: NOT FOUND")
                        
            except Exception as e:
                print(f"   ERROR loading mapping: {e}")
        else:
            print("   WARNING: feature_mapping.json not found")
            
        # Test 3: Feature preparation with sample data
        print("\n3. Testing feature preparation...")
        test_df = create_test_data()
        print(f"   Test data shape: {test_df.shape}")
        print(f"   Test data columns: {list(test_df.columns)}")
        
        # Test different model types
        model_types = ['gru', 'lightgbm', 'ppo']
        symbol = 'BTCEUR'
        
        for model_type in model_types:
            print(f"\n   Testing {model_type.upper()} feature preparation:")
            try:
                if hasattr(feature_engine, 'pad_features_for_model'):
                    result = feature_engine.pad_features_for_model(
                        test_df.copy(), 
                        symbol, 
                        model_type
                    )
                    print(f"     Result shape: {result.shape if hasattr(result, 'shape') else type(result)}")
                    print(f"     Result type: {type(result)}")
                else:
                    print("     pad_features_for_model method not found")
                    
            except Exception as e:
                print(f"     ERROR: {e}")
                
        # Test 4: Feature validation
        print("\n4. Testing feature validation...")
        try:
            if hasattr(feature_engine, 'validate_feature_consistency'):
                validation_result = feature_engine.validate_feature_consistency(
                    test_df, 'BTCEUR', 'gru'
                )
                print(f"   Validation result: {validation_result}")
            else:
                print("   validate_feature_consistency method not found")
        except Exception as e:
            print(f"   Validation ERROR: {e}")
            
        # Test 5: Check for schema drift detection
        print("\n5. Testing schema drift detection...")
        # Create data with different column count
        modified_df = test_df.copy()
        modified_df['extra_column'] = np.random.random(len(modified_df))
        
        try:
            if hasattr(feature_engine, 'detect_schema_drift'):
                drift_result = feature_engine.detect_schema_drift(
                    modified_df, test_df
                )
                print(f"   Schema drift detected: {drift_result}")
            else:
                print("   No schema drift detection method found")
        except Exception as e:
            print(f"   Schema drift test ERROR: {e}")
            
        print("\n=== FeatureEngine Test Complete ===")
        return True
        
    except Exception as e:
        print(f"\nERROR in FeatureEngine test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_engine()
    sys.exit(0 if success else 1)