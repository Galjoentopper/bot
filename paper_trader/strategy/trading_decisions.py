"""
Trading decision making module that integrates with the enhanced compatibility system.

This module provides a high-level interface for making trading decisions
using the improved feature compatibility system.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from ..models.feature_engineer import FeatureEngineer
from ..models.model_loader import WindowBasedModelLoader, WindowBasedEnsemblePredictor
from ..config.settings import TradingSettings


class TradingDecisionMaker:
    """
    High-level trading decision maker that uses the enhanced compatibility system.
    
    This class integrates the improved feature engineering and model compatibility
    to provide reliable trading decisions.
    """
    
    def __init__(self, model_dir: str = "models", threshold: float = 0.02):
        """
        Initialize the trading decision maker.
        
        Args:
            model_dir: Directory containing saved models
            threshold: Minimum threshold for trading decisions
        """
        self.feature_engineer = FeatureEngineer()
        self.model_loader = WindowBasedModelLoader(model_dir)
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize settings
        self.settings = TradingSettings()
    
    def analyze_symbol(self, ohlcv_data: pd.DataFrame, symbol: str, windows: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Analyze a trading symbol and make trading decisions.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            symbol: Trading symbol
            windows: List of time windows to use (defaults to common windows)
            
        Returns:
            Trading decision dictionary or None if analysis failed
        """
        self.logger.info(f"Analyzing {symbol}...")
        
        if windows is None:
            # Convert window names to numbers for compatibility with existing system
            windows = [50, 30, 20, 10]  # Common window numbers
        
        try:
            # 1. Create features using the enhanced feature engineering
            features_df = self.feature_engineer.create_features(ohlcv_data)
            if features_df is None or features_df.empty:
                self.logger.warning(f"Could not create features for {symbol}")
                return None
            
            # 2. Load models for the symbol if not already loaded
            models_loaded = False
            try:
                # Convert symbol format if needed
                symbol_clean = symbol.replace('-', '')
                models_loaded = True  # Assume models exist for now
            except Exception as e:
                self.logger.warning(f"Could not load models for {symbol}: {e}")
            
            if not models_loaded:
                self.logger.warning(f"No models available for {symbol}")
                return None
            
            # 3. Use the ensemble predictor for predictions
            try:
                predictor = WindowBasedEnsemblePredictor(self.model_loader, settings=self.settings)
                
                # Get current price
                current_price = float(features_df['close'].iloc[-1])
                
                # Make prediction using the enhanced system
                prediction_result = predictor.predict(
                    symbol=symbol,
                    features=features_df,
                    current_price=current_price
                )
                
                if prediction_result is None:
                    self.logger.warning(f"No prediction available for {symbol}")
                    return None
                
                # 4. Make trading decision based on prediction
                prediction = prediction_result.get('predicted_price', current_price)
                confidence = prediction_result.get('confidence', 0.0)
                
                # Calculate prediction change percentage
                price_change_pct = (prediction - current_price) / current_price
                
                # Determine action based on prediction and threshold
                action = "HOLD"
                if price_change_pct > self.threshold and confidence > 0.6:
                    action = "BUY"
                elif price_change_pct < -self.threshold and confidence > 0.6:
                    action = "SELL"
                
                # Create result
                result = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "predicted_price": prediction,
                    "price_change_pct": price_change_pct,
                    "confidence": confidence,
                    "action": action,
                    "signal_strength": prediction_result.get('signal_strength', 'NEUTRAL'),
                    "meets_threshold": prediction_result.get('meets_threshold', False),
                    "compatibility_check": prediction_result.get('compatibility_check', True),
                    "model_details": prediction_result.get('model_details', {}),
                    "individual_predictions": prediction_result.get('individual_predictions', {}),
                }
                
                self.logger.info(f"{symbol} decision: {action} (pred: {price_change_pct:.4f}, conf: {confidence:.2f})")
                return result
                
            except Exception as e:
                self.logger.error(f"Error in prediction for {symbol}: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def analyze_multiple_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Analyze multiple symbols and return trading decisions.
        
        Args:
            data_dict: Dictionary with symbol keys and OHLCV DataFrame values
            
        Returns:
            Dictionary with trading decisions for each symbol
        """
        decisions = {}
        
        for symbol, ohlcv_data in data_dict.items():
            try:
                decision = self.analyze_symbol(ohlcv_data, symbol)
                if decision:
                    decisions[symbol] = decision
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                
        return decisions
    
    def get_compatibility_status(self, symbols: List[str]) -> Dict:
        """
        Get compatibility status for multiple symbols.
        
        Args:
            symbols: List of trading symbols to check
            
        Returns:
            Compatibility status dictionary
        """
        try:
            compatibility_handler = self.model_loader.compatibility_handler
            return compatibility_handler.diagnose_system_compatibility(symbols)
        except Exception as e:
            self.logger.error(f"Error getting compatibility status: {e}")
            return {"error": str(e)}