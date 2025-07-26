#!/usr/bin/env python3
"""
Comprehensive test script for the paper trading system.
Tests all components and identifies any bugs that need fixing.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent))

def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_test_logging()

class ComprehensiveTestSuite:
    """Comprehensive test suite for paper trading system"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        logger.info(f"🧪 Running test: {test_name}")
        try:
            result = test_func()
            if result:
                logger.info(f"✅ PASSED: {test_name}")
                self.test_results[test_name] = {"status": "PASSED", "error": None}
                self.passed_tests += 1
            else:
                logger.error(f"❌ FAILED: {test_name}")
                self.test_results[test_name] = {"status": "FAILED", "error": "Test returned False"}
                self.failed_tests += 1
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.test_results[test_name] = {"status": "ERROR", "error": str(e)}
            self.failed_tests += 1
        
        print("-" * 80)
    
    def test_model_directory_structure(self):
        """Test that model directory structure is correct"""
        logger.info("Testing model directory structure...")
        
        models_dir = Path("models")
        required_dirs = ["lstm", "xgboost", "scalers"]
        
        if not models_dir.exists():
            logger.error("Models directory does not exist")
            return False
        
        for dir_name in required_dirs:
            dir_path = models_dir / dir_name
            if not dir_path.exists():
                logger.error(f"Required directory {dir_name} does not exist")
                return False
            logger.info(f"✓ Directory {dir_name} exists")
        
        # Check for required model files
        lstm_model = models_dir / "lstm" / "btceur_window_1.keras"
        xgb_model = models_dir / "xgboost" / "btceur_window_1.json"
        lstm_scaler = models_dir / "scalers" / "btceur_window_1_lstm_scaler.pkl"
        xgb_scaler = models_dir / "scalers" / "btceur_window_1_xgb_scaler.pkl"
        
        required_files = [lstm_model, xgb_model, lstm_scaler, xgb_scaler]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file {file_path} does not exist")
                return False
            logger.info(f"✓ File {file_path.name} exists")
        
        return True
    
    def test_imports(self):
        """Test that all required imports work"""
        logger.info("Testing imports...")
        
        try:
            from paper_trader.config.settings import TradingSettings
            from paper_trader.data.bitvavo_collector import BitvavoDataCollector
            from paper_trader.models.feature_engineer import FeatureEngineer
            from paper_trader.models.model_loader import WindowBasedModelLoader, WindowBasedEnsemblePredictor
            from paper_trader.strategy.signal_generator import SignalGenerator
            from paper_trader.strategy.exit_manager import ExitManager
            from paper_trader.portfolio.portfolio_manager import PortfolioManager
            logger.info("✓ All imports successful")
            return True
        except ImportError as e:
            logger.error(f"Import failed: {e}")
            return False
    
    def test_feature_engineer(self):
        """Test feature engineering with sample data"""
        logger.info("Testing feature engineering...")
        
        try:
            from paper_trader.models.feature_engineer import FeatureEngineer, TRAINING_FEATURES, LSTM_FEATURES
            
            # Create sample data
            dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(40000, 41000, len(dates)),
                'high': np.random.uniform(40500, 41500, len(dates)),
                'low': np.random.uniform(39500, 40500, len(dates)),
                'close': np.random.uniform(40000, 41000, len(dates)),
                'volume': np.random.uniform(100, 1000, len(dates))
            })
            
            # Make prices realistic
            for i in range(1, len(sample_data)):
                sample_data.loc[i, 'open'] = sample_data.loc[i-1, 'close'] + np.random.uniform(-100, 100)
                sample_data.loc[i, 'high'] = sample_data.loc[i, 'open'] + np.random.uniform(0, 200)
                sample_data.loc[i, 'low'] = sample_data.loc[i, 'open'] - np.random.uniform(0, 200)
                sample_data.loc[i, 'close'] = sample_data.loc[i, 'open'] + np.random.uniform(-100, 100)
            
            engineer = FeatureEngineer()
            features_df = engineer.create_features(sample_data)
            
            logger.info(f"✓ Features created with shape: {features_df.shape}")
            logger.info(f"✓ Expected TRAINING_FEATURES: {len(TRAINING_FEATURES)}")
            logger.info(f"✓ Expected LSTM_FEATURES: {len(LSTM_FEATURES)}")
            
            # Check if all required features are present
            missing_training = set(TRAINING_FEATURES) - set(features_df.columns)
            missing_lstm = set(LSTM_FEATURES) - set(features_df.columns)
            
            if missing_training:
                logger.error(f"Missing TRAINING_FEATURES: {missing_training}")
                return False
            
            if missing_lstm:
                logger.error(f"Missing LSTM_FEATURES: {missing_lstm}")
                return False
            
            logger.info("✓ All required features present")
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return False
    
    def test_model_loading(self):
        """Test model loading functionality"""
        logger.info("Testing model loading...")
        
        try:
            from paper_trader.models.model_loader import WindowBasedModelLoader
            from paper_trader.config.settings import TradingSettings
            import asyncio
            
            settings = TradingSettings()
            loader = WindowBasedModelLoader(settings.model_settings.model_path)
            
            # Test loading models for BTCEUR 
            symbol = "BTC-EUR"
            
            async def test_load():
                result = await loader.load_symbol_models(symbol)
                return result
            
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(test_load())
            loop.close()
            
            if not result:
                logger.error("Model loading failed")
                return False
            
            # Check model status
            status = loader.get_model_status()
            symbol_key = symbol  # Use the original symbol format
            if symbol_key not in status:
                logger.error(f"No status found for {symbol}")
                logger.error(f"Available keys: {list(status.keys())}")
                return False
            
            logger.info(f"✓ Models loaded successfully: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def test_prediction_pipeline(self):
        """Test the full prediction pipeline"""
        logger.info("Testing prediction pipeline...")
        
        try:
            from paper_trader.models.model_loader import WindowBasedEnsemblePredictor, WindowBasedModelLoader
            from paper_trader.models.feature_engineer import FeatureEngineer, LSTM_FEATURES, TRAINING_FEATURES
            from paper_trader.config.settings import TradingSettings
            import asyncio
            
            settings = TradingSettings()
            
            # Create model loader first
            model_loader = WindowBasedModelLoader(settings.model_settings.model_path)
            
            # Create predictor with model loader
            predictor = WindowBasedEnsemblePredictor(model_loader)
            
            engineer = FeatureEngineer()
            
            # Create extended sample data for LSTM (need 96+ rows)
            dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='5min')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(40000, 41000, len(dates)),
                'high': np.random.uniform(40500, 41500, len(dates)),
                'low': np.random.uniform(39500, 40500, len(dates)),
                'close': np.random.uniform(40000, 41000, len(dates)),
                'volume': np.random.uniform(100, 1000, len(dates))
            })
            
            # Make prices realistic
            for i in range(1, len(sample_data)):
                sample_data.loc[i, 'open'] = sample_data.loc[i-1, 'close'] + np.random.uniform(-100, 100)
                sample_data.loc[i, 'high'] = sample_data.loc[i, 'open'] + np.random.uniform(0, 200)
                sample_data.loc[i, 'low'] = sample_data.loc[i, 'open'] - np.random.uniform(0, 200)
                sample_data.loc[i, 'close'] = sample_data.loc[i, 'open'] + np.random.uniform(-100, 100)
            
            logger.info(f"Created sample data with {len(sample_data)} rows")
            
            # Create features
            features_df = engineer.create_features(sample_data)
            logger.info(f"Features created with shape: {features_df.shape}")
            
            # Ensure we have enough data for LSTM
            if len(features_df) < 96:
                logger.error(f"Insufficient data for LSTM: {len(features_df)} < 96")
                return False
            
            # Test prediction
            symbol = "BTC-EUR"
            
            # Get the latest features
            latest_features = features_df.iloc[-1:] # Get as DataFrame instead of Series
            
            # Prepare LSTM sequence (last 96 rows)
            lstm_sequence = features_df[LSTM_FEATURES].iloc[-96:].values
            logger.info(f"LSTM sequence shape: {lstm_sequence.shape}")
            
            async def test_predict():
                # First load the models
                await model_loader.load_symbol_models(symbol)
                
                # Make prediction - pass the full features_df instead of just the latest row
                prediction = await predictor.predict(symbol, features_df, lstm_sequence)
                return prediction
            
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            prediction = loop.run_until_complete(test_predict())
            loop.close()
            
            if prediction is None:
                logger.error("Prediction returned None")
                return False
            
            logger.info(f"✓ Prediction successful: {prediction}")
            
            # Validate prediction structure
            required_keys = ['direction', 'confidence', 'lstm_prob', 'xgb_prob']
            for key in required_keys:
                if key not in prediction:
                    logger.error(f"Missing key in prediction: {key}")
                    return False
            
            logger.info("✓ Prediction structure is valid")
            return True
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            return False
    
    def test_main_paper_trader_init(self):
        """Test main paper trader initialization"""
        logger.info("Testing main paper trader initialization...")
        
        try:
            from main_paper_trader import PaperTrader
            
            # Create paper trader instance
            trader = PaperTrader()
            
            if trader.settings is None:
                logger.error("Settings not loaded")
                return False
            
            logger.info("✓ Paper trader initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Paper trader initialization failed: {e}")
            return False
    
    def test_settings_loading(self):
        """Test that settings load correctly"""
        logger.info("Testing settings loading...")
        
        try:
            from paper_trader.config.settings import TradingSettings
            
            settings = TradingSettings()
            
            # Check critical settings
            if not hasattr(settings, 'model_settings'):
                logger.error("Missing model_settings")
                return False
            
            if not hasattr(settings, 'trading_settings'):
                logger.error("Missing trading_settings")
                return False
            
            logger.info("✓ Settings loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Settings loading failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting comprehensive test suite...")
        print("="*80)
        
        tests = [
            ("Model Directory Structure", self.test_model_directory_structure),
            ("Imports", self.test_imports),
            ("Settings Loading", self.test_settings_loading),
            ("Feature Engineer", self.test_feature_engineer),
            ("Model Loading", self.test_model_loading),
            ("Prediction Pipeline", self.test_prediction_pipeline),
            ("Main Paper Trader Init", self.test_main_paper_trader_init),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("="*80)
        logger.info("📊 TEST SUMMARY")
        logger.info(f"✅ Passed: {self.passed_tests}")
        logger.info(f"❌ Failed: {self.failed_tests}")
        logger.info(f"Total: {self.passed_tests + self.failed_tests}")
        
        if self.failed_tests == 0:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.error(f"💥 {self.failed_tests} TESTS FAILED!")
            
            # Print failed test details
            logger.error("\n📋 FAILED TEST DETAILS:")
            for test_name, result in self.test_results.items():
                if result["status"] in ["FAILED", "ERROR"]:
                    logger.error(f"  • {test_name}: {result['error']}")
        
        return self.failed_tests == 0

def main():
    """Main test function"""
    test_suite = ComprehensiveTestSuite()
    success = test_suite.run_all_tests()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()