import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from tensorflow.keras.models import load_model
import xgboost as xgb
import pickle
from feature_factory import FeatureFactory

class ModelManager:
    """
    Manages multiple models with different time windows and combines their predictions.
    
    This class handles loading models, generating predictions from multiple time windows,
    and combining those predictions into a single trading signal.
    """
    
    def __init__(self, model_dir: str, window_sizes: List[int] = [30, 60, 90]):
        """
        Initialize the ModelManager.
        
        Args:
            model_dir: Directory containing the trained models
            window_sizes: List of window sizes to use (in days)
        """
        self.model_dir = model_dir
        self.window_sizes = window_sizes
        self.lstm_models = {}
        self.xgb_models = {}
        
        # Load models
        self.load_models()
    
    def load_models(self) -> None:
        """Load all LSTM and XGBoost models for the specified window sizes."""
        for window in self.window_sizes:
            # Load LSTM model
            lstm_path = os.path.join(self.model_dir, f"lstm_model_{window}.h5")
            if os.path.exists(lstm_path):
                try:
                    self.lstm_models[window] = load_model(lstm_path)
                    print(f"Loaded LSTM model for {window}-day window")
                except Exception as e:
                    print(f"Failed to load LSTM model for {window}-day window: {e}")
            else:
                print(f"LSTM model not found: {lstm_path}")
            
            # Load XGBoost model
            xgb_path = os.path.join(self.model_dir, f"xgb_model_{window}.pkl")
            if os.path.exists(xgb_path):
                try:
                    with open(xgb_path, 'rb') as f:
                        self.xgb_models[window] = pickle.load(f)
                    print(f"Loaded XGBoost model for {window}-day window")
                except Exception as e:
                    print(f"Failed to load XGBoost model for {window}-day window: {e}")
            else:
                print(f"XGBoost model not found: {xgb_path}")
    
    def predict(self, feature_factory: FeatureFactory, latest_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Generate predictions from all models and combine them.
        
        Args:
            feature_factory: FeatureFactory instance with processed data
            latest_data: Optional latest market data to include
            
        Returns:
            Dictionary with combined prediction and individual model predictions
        """
        predictions = {
            'lstm': {},
            'xgboost': {}
        }
        
        # Generate predictions from each LSTM model
        for window in self.window_sizes:
            if window in self.lstm_models:
                try:
                    lstm_features = feature_factory.get_prediction_features('lstm', window, latest_data)
                    if lstm_features is not None and lstm_features.shape[0] > 0:
                        pred = self.lstm_models[window].predict(lstm_features, verbose=0)[0][0]
                        predictions['lstm'][window] = float(pred)
                except Exception as e:
                    print(f"Error predicting with LSTM model ({window}): {e}")
        
        # Generate predictions from each XGBoost model
        for window in self.window_sizes:
            if window in self.xgb_models:
                try:
                    xgb_features = feature_factory.get_prediction_features('xgboost', window, latest_data)
                    if xgb_features is not None and xgb_features.shape[0] > 0:
                        pred = self.xgb_models[window].predict(xgb_features)[0]
                        predictions['xgboost'][window] = float(pred)
                except Exception as e:
                    print(f"Error predicting with XGBoost model ({window}): {e}")
        
        # Combine predictions using a weighted approach
        # Shorter windows get higher weight for short-term signals
        # Longer windows get higher weight for trend confirmation
        combined_prediction = self._combine_predictions(predictions)
        
        # Add combined prediction to the results
        predictions['combined'] = combined_prediction
        
        return predictions
    
    def _combine_predictions(self, predictions: Dict[str, Dict[int, float]]) -> float:
        """
        Combine predictions from multiple models and windows.
        
        Args:
            predictions: Nested dictionary with model type and window size predictions
            
        Returns:
            Combined prediction value
        """
        # Define weights for different window sizes
        # Adjust these weights based on backtesting performance
        lstm_weights = {
            30: 0.4,  # Short-term signals (higher weight)
            60: 0.3,  # Medium-term signals
            90: 0.3   # Long-term signals
        }
        
        xgb_weights = {
            30: 0.4,
            60: 0.3,
            90: 0.3
        }
        
        # Overall model type weights
        model_weights = {
            'lstm': 0.5,
            'xgboost': 0.5
        }
        
        # Calculate weighted average for each model type
        lstm_pred = 0
        lstm_weight_sum = 0
        for window, pred in predictions['lstm'].items():
            weight = lstm_weights.get(window, 0.3)
            lstm_pred += pred * weight
            lstm_weight_sum += weight
        
        if lstm_weight_sum > 0:
            lstm_pred /= lstm_weight_sum
        
        xgb_pred = 0
        xgb_weight_sum = 0
        for window, pred in predictions['xgboost'].items():
            weight = xgb_weights.get(window, 0.3)
            xgb_pred += pred * weight
            xgb_weight_sum += weight
        
        if xgb_weight_sum > 0:
            xgb_pred /= xgb_weight_sum
        
        # Combine model predictions
        # If only one model type has predictions, use that one
        if lstm_weight_sum > 0 and xgb_weight_sum > 0:
            combined_pred = (
                lstm_pred * model_weights['lstm'] +
                xgb_pred * model_weights['xgboost']
            )
        elif lstm_weight_sum > 0:
            combined_pred = lstm_pred
        elif xgb_weight_sum > 0:
            combined_pred = xgb_pred
        else:
            # No predictions available, return neutral (0.5)
            combined_pred = 0.5
        
        return combined_pred
    
    def get_model_info(self) -> Dict[str, List[int]]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with available models and their window sizes
        """
        info = {
            'lstm_windows': list(self.lstm_models.keys()),
            'xgboost_windows': list(self.xgb_models.keys()),
            'all_windows': sorted(set(list(self.lstm_models.keys()) + list(self.xgb_models.keys())))
        }
        return info