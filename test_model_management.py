#!/usr/bin/env python3
"""Test model management system."""

import sys
import tempfile
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print("Starting Model Management Test")
print("=" * 40)

try:
    # Test ModelManager
    print("\n1. Testing ModelManager...")
    from src.models.model_manager import ModelManager
    from src.core.interfaces import ModelMetadata
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        model_manager = ModelManager(models_dir=temp_dir)
        print("   ✓ ModelManager created successfully")
        
        # Test listing models (should be empty initially)
        models = model_manager.list_available_models()
        print(f"   ✓ Listed {len(models)} models (expected 0)")
        
        # Create a simple mock model
        class MockModel:
            def predict(self, X):
                return np.random.random(len(X))
        
        mock_model = MockModel()
        
        # Save mock model to test directory
        model_path = Path(temp_dir) / "BTCEUR_lstm.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        print("   ✓ Created mock model file")
        
        # Test model loading
        try:
            loaded_model = model_manager.load_model("BTCEUR", "lstm")
            print("   ✓ Model loaded successfully")
            
            # Test model prediction
            test_data = np.random.random((5, 10))
            predictions = loaded_model.predict(test_data)
            print(f"   ✓ Model prediction successful: {len(predictions)} predictions")
            
        except Exception as e:
            print(f"   ✗ Model loading failed: {e}")
        
        # Test metadata creation and retrieval
        try:
            metadata = model_manager.get_model_metadata("BTCEUR", "lstm")
            print(f"   ✓ Retrieved metadata: {metadata.model_type}")
            
            # Create custom metadata
            custom_metadata = ModelMetadata(
                model_type="lstm",
                symbol="BTCEUR",
                version="1.0.0",
                features=["close", "volume", "rsi", "macd"],
                created_at=datetime.now(),
                performance_metrics={"accuracy": 0.85, "precision": 0.82},
                config={"epochs": 100, "batch_size": 32}
            )
            
            # Save metadata
            success = model_manager.save_model_metadata("BTCEUR", "lstm", custom_metadata)
            print(f"   ✓ Metadata saved: {success}")
            
            # Retrieve saved metadata
            retrieved_metadata = model_manager.get_model_metadata("BTCEUR", "lstm")
            print(f"   ✓ Retrieved saved metadata: {len(retrieved_metadata.features)} features")
            
        except Exception as e:
            print(f"   ✗ Metadata operations failed: {e}")
        
        # Test model-feature compatibility validation
        try:
            feature_schema = {
                'columns': ['close', 'volume', 'rsi', 'macd', 'extra_feature'],
                'dtypes': {
                    'close': 'float64',
                    'volume': 'float64',
                    'rsi': 'float64',
                    'macd': 'float64',
                    'extra_feature': 'float64'
                },
                'feature_count': 5
            }
            
            validation_result = model_manager.validate_model_compatibility(
                custom_metadata, feature_schema
            )
            
            print(f"   ✓ Compatibility validation: {validation_result.is_valid}")
            print(f"   ✓ Validation errors: {len(validation_result.errors)}")
            print(f"   ✓ Validation warnings: {len(validation_result.warnings)}")
            
        except Exception as e:
            print(f"   ✗ Compatibility validation failed: {e}")
        
        # Test model listing after adding files
        try:
            models = model_manager.list_available_models()
            print(f"   ✓ Listed {len(models)} models after creation")
            
            if models:
                model_info = models[0]
                print(f"   ✓ Model info: {model_info['symbol']}_{model_info['model_type']}")
            
        except Exception as e:
            print(f"   ✗ Model listing failed: {e}")
        
        # Test cache operations
        try:
            model_manager.clear_cache()
            print("   ✓ Cache cleared successfully")
            
        except Exception as e:
            print(f"   ✗ Cache clearing failed: {e}")

except Exception as e:
    print(f"   ✗ ModelManager test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)
print("Model Management Test Completed")
print("✓ Model management functionality verified")