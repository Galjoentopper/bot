#!/usr/bin/env python3
"""
Isolated test for Model Loading component
Tests model discovery, loading strategies, and metadata validation
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_files_discovery():
    """Test model file discovery and structure"""
    print("=== Model Files Discovery Test ===")
    
    models_dir = project_root / 'models'
    if not models_dir.exists():
        print("ERROR: models directory not found")
        return False
        
    print(f"Models directory: {models_dir}")
    
    # Expected symbols and model types
    symbols = ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']
    model_types = ['gru', 'lightgbm', 'ppo']
    
    results = {}
    
    for symbol in symbols:
        symbol_dir = models_dir / symbol
        results[symbol] = {'exists': symbol_dir.exists(), 'models': {}}
        
        if symbol_dir.exists():
            print(f"\n{symbol} directory found")
            
            for model_type in model_types:
                model_dir = symbol_dir / model_type
                model_info = {
                    'dir_exists': model_dir.exists(),
                    'files': {}
                }
                
                if model_dir.exists():
                    # Check for expected files
                    expected_files = {
                        'model.pth': 'PyTorch model file',
                        'model.pkl': 'Pickle model file', 
                        'model.zip': 'Zipped model file',
                        'model_metadata.json': 'Model metadata',
                        'imported_metadata.json': 'Imported metadata'
                    }
                    
                    for filename, description in expected_files.items():
                        file_path = model_dir / filename
                        exists = file_path.exists()
                        size = file_path.stat().st_size if exists else 0
                        
                        model_info['files'][filename] = {
                            'exists': exists,
                            'size': size,
                            'description': description
                        }
                        
                        status = 'EXISTS' if exists else 'MISSING'
                        print(f"  {model_type}/{filename}: {status} ({size} bytes)")
                        
                    # Try to load and validate metadata
                    metadata_file = model_dir / 'model_metadata.json'
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            print(f"  {model_type} metadata keys: {list(metadata.keys())}")
                            
                            # Check for critical metadata fields
                            critical_fields = ['model_type', 'symbol', 'input_shape', 'feature_count']
                            for field in critical_fields:
                                if field in metadata:
                                    print(f"    {field}: {metadata[field]}")
                                else:
                                    print(f"    {field}: MISSING")
                                    
                        except Exception as e:
                            print(f"  ERROR loading {model_type} metadata: {e}")
                else:
                    print(f"  {model_type}: DIRECTORY MISSING")
                    
                results[symbol]['models'][model_type] = model_info
        else:
            print(f"\n{symbol}: DIRECTORY MISSING")
            
    return results

def test_model_loading_strategies():
    """Test different model loading strategies"""
    print("\n=== Model Loading Strategies Test ===")
    
    try:
        # Import trainer classes
        from src.trainers.gru_trainer import GRUTrainer
        from src.trainers.lightgbm_trainer import LightGBMTrainer
        from src.trainers.ppo_trainer import PPOTrainer
        
        trainers = {
            'gru': GRUTrainer,
            'lightgbm': LightGBMTrainer,
            'ppo': PPOTrainer
        }
        
        test_symbol = 'BTCEUR'
        
        for model_type, trainer_class in trainers.items():
            print(f"\nTesting {model_type.upper()} loading:")
            
            try:
                # Initialize trainer
                trainer = trainer_class(symbol=test_symbol)
                print(f"  Trainer initialized: {trainer is not None}")
                
                # Test model loading methods
                loading_methods = [
                    'load_model',
                    'load_from_packaged_models',
                    'load_from_imported_models', 
                    'load_from_latest_models'
                ]
                
                for method_name in loading_methods:
                    if hasattr(trainer, method_name):
                        try:
                            method = getattr(trainer, method_name)
                            result = method()
                            print(f"    {method_name}: {'SUCCESS' if result else 'FAILED'}")
                        except Exception as e:
                            print(f"    {method_name}: ERROR - {e}")
                    else:
                        print(f"    {method_name}: METHOD NOT FOUND")
                        
                # Check if model is loaded
                has_model = hasattr(trainer, 'model') and trainer.model is not None
                print(f"  Model loaded: {has_model}")
                
                if has_model:
                    print(f"  Model type: {type(trainer.model)}")
                    
            except Exception as e:
                print(f"  ERROR initializing {model_type} trainer: {e}")
                
    except ImportError as e:
        print(f"ERROR importing trainers: {e}")
        return False
        
    return True

def test_enhanced_trader_loading():
    """Test EnhancedUnifiedPaperTrader model loading"""
    print("\n=== Enhanced Trader Loading Test ===")
    
    try:
        from scripts.enhanced_trader import EnhancedUnifiedPaperTrader
        
        print("Testing EnhancedUnifiedPaperTrader initialization...")
        
        # Test with default symbols
        trader = EnhancedUnifiedPaperTrader()
        print(f"Trader initialized: {trader is not None}")
        print(f"Trader symbols: {getattr(trader, 'symbols', 'N/A')}")
        
        # Check loaded models
        if hasattr(trader, 'models'):
            print(f"Models attribute exists: {trader.models is not None}")
            if trader.models:
                print(f"Loaded models: {list(trader.models.keys())}")
                
                for key, model_info in trader.models.items():
                    print(f"  {key}: {type(model_info) if model_info else 'None'}")
            else:
                print("No models loaded")
        else:
            print("No models attribute found")
            
        # Test model loading method
        if hasattr(trader, 'load_all_models'):
            print("\nTesting load_all_models method...")
            try:
                result = trader.load_all_models()
                print(f"load_all_models result: {result}")
            except Exception as e:
                print(f"load_all_models ERROR: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR testing EnhancedUnifiedPaperTrader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Main test function"""
    print("=== Model Loading Component Test ===")
    
    try:
        # Test 1: File discovery
        discovery_results = test_model_files_discovery()
        
        # Test 2: Loading strategies
        strategies_success = test_model_loading_strategies()
        
        # Test 3: Enhanced trader loading
        trader_success = test_enhanced_trader_loading()
        
        print("\n=== Model Loading Test Complete ===")
        
        # Summary
        print("\nSUMMARY:")
        print(f"File discovery: {'PASS' if discovery_results else 'FAIL'}")
        print(f"Loading strategies: {'PASS' if strategies_success else 'FAIL'}")
        print(f"Enhanced trader: {'PASS' if trader_success else 'FAIL'}")
        
        return discovery_results and strategies_success and trader_success
        
    except Exception as e:
        print(f"\nERROR in model loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)