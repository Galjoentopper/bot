"""Enhanced model loading and prediction ensemble for trading signals with multi-window support."""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from .feature_engineer import LSTM_FEATURES
from tensorflow.keras import backend as K

# Standalone directional loss function for proper serialization
@tf.keras.utils.register_keras_serializable(package="Custom", name="directional_loss")
def directional_loss(y_true, y_pred):
    """Custom loss function that penalizes wrong directional predictions more."""
    # Standard MSE
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # Directional penalty
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    direction_penalty = tf.where(
        tf.equal(direction_true, direction_pred),
        0.0,
        tf.abs(y_true - y_pred) * 2.0  # Double penalty for wrong direction
    )
    
    return mse + tf.reduce_mean(direction_penalty)

# Quantile loss used during training
@tf.keras.utils.register_keras_serializable(package="Custom", name="quantile_loss")
def quantile_loss(q):
    """Return a quantile loss function configured for quantile ``q``."""

    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

    return loss

class WindowBasedModelLoader:
    """Loads and manages window-based pre-trained models for trading predictions."""
    
    def __init__(self, model_path: str = 'models'):
        self.model_path = Path(model_path)
        # Window-based model storage: {symbol: {window: model}}
        self.lstm_models: Dict[str, Dict[int, tf.keras.Model]] = {}
        self.xgb_models: Dict[str, Dict[int, xgb.XGBRegressor]] = {}
        self.scalers: Dict[str, Dict[int, StandardScaler]] = {}
        # Feature column order used during training
        self.feature_columns: Dict[str, Dict[int, List[str]]] = {}
        
        # Available windows for each symbol
        self.available_windows: Dict[str, List[int]] = {}
        
        self.logger = logging.getLogger(__name__)
        
    async def load_symbol_models(self, symbol: str, custom_objects=None) -> bool:
        """Load all window-based LSTM and XGBoost models for a specific symbol."""
        try:
            # Convert Bitvavo format (BTC-EUR) to model file format (btceur)
            symbol_lower = symbol.lower().replace('-', '')
            
            # Initialize storage for this symbol
            self.lstm_models[symbol] = {}
            self.xgb_models[symbol] = {}
            self.scalers[symbol] = {}
            self.feature_columns[symbol] = {}
            self.available_windows[symbol] = []
            
            # Discover available windows by scanning directories
            windows_found = set()
            
            # Scan LSTM models
            lstm_dir = self.model_path / 'lstm'
            if lstm_dir.exists():
                for lstm_file in lstm_dir.glob(f"{symbol_lower}_window_*.keras"):
                    try:
                        window_num = int(lstm_file.stem.split('_window_')[1])
                        if custom_objects is None:
                            custom_objects = {
                                'directional_loss': directional_loss,
                                'quantile_loss': lambda q: (
                                    lambda y_true, y_pred: quantile_loss(q)(y_true, y_pred)
                                ),
                            }
                        try:
                            model = tf.keras.models.load_model(str(lstm_file), custom_objects=custom_objects)
                            self.lstm_models[symbol][window_num] = model
                            self.logger.debug(f"Loaded LSTM model for {symbol} window {window_num}")
                        except Exception as e:
                            self.logger.warning(f"Failed to load LSTM model {lstm_file}: {e}")
                            try:
                                model = tf.keras.models.load_model(str(lstm_file), compile=False)
                                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                                self.lstm_models[symbol][window_num] = model
                                self.logger.info(f"Loaded {lstm_file} without custom objects")
                            except Exception as e2:
                                self.logger.warning(f"Fallback load failed for {lstm_file}: {e2}")
                    except Exception as e:
                        self.logger.warning(f"Failed to process LSTM model file {lstm_file}: {e}")
            
            # Scan XGBoost models
            xgb_dir = self.model_path / 'xgboost'
            if xgb_dir.exists():
                for xgb_file in xgb_dir.glob(f"{symbol_lower}_window_*.json"):
                    try:
                        window_num = int(xgb_file.stem.split('_window_')[1])
                        windows_found.add(window_num)
                        model = xgb.XGBClassifier()
                        model.load_model(xgb_file)
                        self.xgb_models[symbol][window_num] = model
                        self.logger.debug(f"Loaded XGBoost model for {symbol} window {window_num}")
                    except (ValueError, Exception) as e:
                        self.logger.warning(f"Failed to load XGBoost model {xgb_file}: {e}")
            
            # Scan scalers
            scalers_dir = self.model_path / 'scalers'
            if scalers_dir.exists():
                for scaler_file in scalers_dir.glob(f"{symbol_lower}_window_*_scaler.pkl"):
                    try:
                        window_num = int(scaler_file.stem.split('_window_')[1].split('_scaler')[0])
                        with open(scaler_file, 'rb') as f:
                            scaler = pickle.load(f)
                        self.scalers[symbol][window_num] = scaler
                        self.logger.debug(f"Loaded scaler for {symbol} window {window_num}")
                    except (ValueError, Exception) as e:
                        self.logger.warning(f"Failed to load scaler {scaler_file}: {e}")

            # Load feature column order per window
            feature_dir = self.model_path / 'feature_columns'
            if feature_dir.exists():
                for fc_file in feature_dir.glob(f"{symbol_lower}_window_*.pkl"):
                    try:
                        window_num = int(fc_file.stem.split('_window_')[1].split('_')[0])
                        with open(fc_file, 'rb') as f:
                            cols = pickle.load(f)
                        self.feature_columns[symbol][window_num] = cols
                        windows_found.add(window_num)
                        self.logger.debug(f"Loaded feature columns for {symbol} window {window_num}")
                    except (ValueError, Exception) as e:
                        self.logger.warning(f"Failed to load feature columns {fc_file}: {e}")
            
            # Update available windows
            self.available_windows[symbol] = sorted(list(windows_found))
            
            # Check if we have at least one model
            has_models = len(self.lstm_models[symbol]) > 0 or len(self.xgb_models[symbol]) > 0
            
            if has_models:
                self.logger.info(f"Successfully loaded models for {symbol}: {len(windows_found)} windows found")
                self.logger.info(f"Available windows for {symbol}: {self.available_windows[symbol]}")
            else:
                self.logger.error(f"No models loaded for {symbol}")
                
            return has_models
            
        except Exception as e:
            self.logger.error(f"Error loading models for {symbol}: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, dict]:
        """Get status of loaded models."""
        status = {}
        all_symbols = set(list(self.lstm_models.keys()) + list(self.xgb_models.keys()))
        
        for symbol in all_symbols:
            status[symbol] = {
                'lstm_windows': list(self.lstm_models.get(symbol, {}).keys()),
                'xgb_windows': list(self.xgb_models.get(symbol, {}).keys()),
                'scaler_windows': list(self.scalers.get(symbol, {}).keys()),
                'available_windows': self.available_windows.get(symbol, []),
                'total_windows': len(self.available_windows.get(symbol, []))
            }
        
        return status
    
    def get_optimal_window(self, symbol: str, market_volatility: float, trend_strength: float) -> Optional[int]:
        """Select optimal window based on market conditions."""
        if symbol not in self.available_windows or not self.available_windows[symbol]:
            return None

        windows = sorted(self.available_windows[symbol])

        if market_volatility > 0.7:
            return max(windows[-5:])
        elif trend_strength > 0.6 and len(windows) >= 10:
            return windows[-10]
        else:
            return windows[-15] if len(windows) >= 15 else windows[-1]
    
    def get_available_models(self, symbol: str, window: int) -> Dict[str, bool]:
        """Check which models are available for a specific symbol and window."""
        return {
            'lstm': window in self.lstm_models.get(symbol, {}),
            'xgb': window in self.xgb_models.get(symbol, {}),
            'scaler': window in self.scalers.get(symbol, {})
        }

class WindowBasedEnsemblePredictor:
    """Enhanced ensemble predictor with window-based model selection and improved confidence thresholds."""
    
    def __init__(self, model_loader: WindowBasedModelLoader, 
                 min_confidence_threshold: float = 0.6,
                 min_signal_strength: str = 'MODERATE'):
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
        
        # Enhanced prediction weights and thresholds (prioritize XGBoost when LSTM fails)
        self.lstm_weight = 0.6
        self.xgb_weight = 0.4
        self.min_confidence_threshold = min_confidence_threshold
        self.min_signal_strength = min_signal_strength
        
        # Signal strength hierarchy
        self.signal_hierarchy = {
            'WEAK': 1,
            'NEUTRAL': 2, 
            'MODERATE': 3,
            'STRONG': 4,
            'VERY_STRONG': 5
        }
        
    async def predict(self, symbol: str, features: pd.DataFrame,
                     market_volatility: float = 0.5) -> Optional[dict]:
        """Generate enhanced ensemble prediction with window-based model selection."""
        try:
            if len(features) < 10:  # Minimum features needed
                self.logger.warning(f"Insufficient features for prediction: {len(features)}")
                return None

            # Determine trend strength from engineered features
            if 'trend_strength' in features.columns:
                trend_strength = features['trend_strength'].tail(20).mean()
                trend_strength = float(np.abs(trend_strength)) if not np.isnan(trend_strength) else 0.0
            else:
                trend_strength = 0.0

            # Select optimal window based on market conditions
            optimal_window = self.model_loader.get_optimal_window(symbol, market_volatility, trend_strength)
            if optimal_window is None:
                self.logger.warning(f"No models available for {symbol}")
                return None

            required_cols = self.model_loader.feature_columns.get(symbol, {}).get(optimal_window)
            if required_cols:
                missing = [c for c in required_cols if c not in features.columns]
                if missing:
                    self.logger.warning(f"Missing required feature columns for {symbol}: {missing}")
                    return None
            
            # Get available models for the selected window
            available_models = self.model_loader.get_available_models(symbol, optimal_window)
            
            predictions = {}
            confidence_scores = {}
            model_details = {}
            
            # LSTM Prediction with selected window (skip if models are corrupted)
            if available_models['lstm']:
                try:
                    lstm_pred, lstm_conf = await self._predict_lstm_window(symbol, features, optimal_window)
                    if lstm_pred is not None:
                        predictions['lstm'] = lstm_pred
                        confidence_scores['lstm'] = lstm_conf
                        model_details['lstm_window'] = optimal_window
                except Exception as e:
                    self.logger.warning(f"LSTM model failed for {symbol} window {optimal_window}, skipping: {e}")
            
            # XGBoost Prediction with selected window
            if available_models['xgb']:
                xgb_pred, xgb_conf = await self._predict_xgboost_window(symbol, features, optimal_window)
                if xgb_pred is not None:
                    predictions['xgb'] = xgb_pred
                    confidence_scores['xgb'] = xgb_conf
                    model_details['xgb_window'] = optimal_window
            
            if not predictions:
                self.logger.warning(f"No valid predictions for {symbol} with window {optimal_window}")
                return None
            
            # Calculate ensemble prediction with enhanced weighting
            ensemble_pred = self._calculate_enhanced_ensemble(predictions, confidence_scores)

            # Calculate uncertainty and confidence
            uncertainty = self.calculate_prediction_uncertainty(predictions, market_volatility)
            avg_confidence = self._calculate_adjusted_confidence(confidence_scores, market_volatility)
            
            # Determine signal strength
            current_price = float(features['close'].iloc[-1])
            price_change_pct = (ensemble_pred - current_price) / current_price
            
            # Enhanced signal classification
            signal_strength = self._classify_enhanced_signal(price_change_pct, avg_confidence, market_volatility)
            
            # Check if prediction meets minimum thresholds
            meets_threshold = self._meets_trading_threshold(avg_confidence, signal_strength)
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': ensemble_pred,
                'price_change_pct': price_change_pct,
                'confidence': avg_confidence,
                'uncertainty': uncertainty,
                'signal_strength': signal_strength,
                'meets_threshold': meets_threshold,
                'optimal_window': optimal_window,
                'market_volatility': market_volatility,
                'individual_predictions': predictions,
                'individual_confidences': confidence_scores,
                'model_details': model_details,
                'timestamp': pd.Timestamp.now()
            }
            
            self.logger.debug(
                f"Enhanced prediction for {symbol}: {price_change_pct:.4f}% change, "
                f"confidence: {avg_confidence:.3f}, uncertainty: {uncertainty:.3f}, "
                f"window: {optimal_window}, meets_threshold: {meets_threshold}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced prediction for {symbol}: {e}")
            return None
    
    async def _predict_lstm_window(self, symbol: str, features: pd.DataFrame, window: int) -> Tuple[Optional[float], float]:
        """Generate LSTM prediction using specific window model."""
        try:
            if (symbol not in self.model_loader.lstm_models or 
                window not in self.model_loader.lstm_models[symbol]):
                return None, 0.0
            
            model = self.model_loader.lstm_models[symbol][window]
            scaler = self.model_loader.scalers[symbol].get(window)
            
            # Prepare features
            feature_data = features.copy()
            feature_cols = [col for col in LSTM_FEATURES if col in feature_data.columns]
            if scaler is not None and feature_cols:
                feature_data[feature_cols] = scaler.transform(feature_data[feature_cols])
            
            # Create sequence for LSTM (use window size)
            sequence_length = min(window * 2, len(feature_data))  # Dynamic sequence length
            if len(feature_data) < sequence_length:
                sequence_length = len(feature_data)
            
            # Select numeric features
            numeric_cols = feature_cols
            
            sequence = feature_data[numeric_cols].tail(sequence_length).values
            
            # Pad sequence if needed
            if sequence.shape[0] < window * 2:
                padding = np.zeros((window * 2 - sequence.shape[0], sequence.shape[1]))
                sequence = np.vstack([padding, sequence])
            
            sequence = sequence.reshape(1, sequence.shape[0], len(numeric_cols))
            
            # Make prediction
            prediction = model.predict(sequence, verbose=0)[0][0]
            
            # Enhanced confidence calculation
            current_price = features['close'].iloc[-1]
            price_volatility = features['close'].tail(10).std() / features['close'].tail(10).mean()
            prediction_error = abs(prediction - current_price) / current_price
            
            # Confidence based on prediction stability and volatility
            base_confidence = 1.0 - min(prediction_error * 2, 0.8)
            volatility_adjustment = max(0.1, 1.0 - price_volatility)
            confidence = min(0.95, max(0.1, base_confidence * volatility_adjustment))
            
            return float(prediction), confidence
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction for {symbol} window {window}: {e}")
            return None, 0.0
    
    async def _predict_xgboost_window(self, symbol: str, features: pd.DataFrame, window: int) -> Tuple[Optional[float], float]:
        """Generate XGBoost prediction using specific window model."""
        try:
            if (symbol not in self.model_loader.xgb_models or 
                window not in self.model_loader.xgb_models[symbol]):
                return None, 0.0
            
            model = self.model_loader.xgb_models[symbol][window]
            feature_columns = self.model_loader.feature_columns.get(symbol, {}).get(window)
            
            # Prepare features
            if feature_columns:
                available_features = [c for c in feature_columns if c in features.columns]
                if len(available_features) < len(feature_columns) * 0.8:
                    self.logger.warning(
                        f"Missing features for XGBoost prediction: {symbol} window {window}")
                    return None, 0.0
                feature_data = features[available_features].tail(1)
            else:
                # Use all numeric features
                numeric_cols = [col for col in features.columns 
                              if features[col].dtype in ['float64', 'int64']]
                feature_data = features[numeric_cols].tail(1)
            
            # Make prediction
            prediction = model.predict(feature_data)[0]
            
            # Enhanced confidence calculation for XGBoost
            current_price = features['close'].iloc[-1]
            price_volatility = features['close'].tail(10).std() / features['close'].tail(10).mean()
            prediction_error = abs(prediction - current_price) / current_price
            
            # Base confidence adjusted by prediction accuracy and market stability
            base_confidence = 0.75  # Higher base for XGBoost
            error_penalty = min(prediction_error * 1.5, 0.4)
            volatility_bonus = max(0, 0.2 - price_volatility)  # XGBoost performs better in stable markets
            
            confidence = min(0.9, max(0.2, base_confidence - error_penalty + volatility_bonus))
            
            return float(prediction), confidence
            
        except Exception as e:
            self.logger.error(f"Error in XGBoost prediction for {symbol} window {window}: {e}")
            return None, 0.0
    
    def _calculate_enhanced_ensemble(self, predictions: dict, confidence_scores: dict) -> float:
        """Calculate enhanced weighted ensemble prediction with adaptive weighting."""
        if not predictions:
            return 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        # Calculate adaptive weights based on confidence
        max_confidence = max(confidence_scores.values()) if confidence_scores else 1.0
        
        for model_name, prediction in predictions.items():
            confidence = confidence_scores[model_name]
            
            # Base weight
            if model_name == 'lstm':
                base_weight = self.lstm_weight
            elif model_name == 'xgb':
                base_weight = self.xgb_weight
            else:
                base_weight = 0.5
            
            # Adaptive confidence weighting
            confidence_multiplier = (confidence / max_confidence) ** 0.5  # Square root for smoother scaling
            weight = base_weight * confidence_multiplier
            
            weighted_sum += prediction * weight
            total_weight += weight
        
        if total_weight == 0:
            return np.mean(list(predictions.values()))

        return weighted_sum / total_weight

    def calculate_prediction_uncertainty(self, predictions: dict, market_volatility: float) -> float:
        """Calculate uncertainty based on model disagreement and market conditions."""
        if len(predictions) < 2:
            return 1.0

        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)

        cv = pred_std / pred_mean if pred_mean != 0 else 1.0
        uncertainty = cv * (1 + market_volatility)
        return min(1.0, uncertainty)
    
    def _calculate_adjusted_confidence(self, confidence_scores: dict, market_volatility: float) -> float:
        """Calculate confidence adjusted for market volatility."""
        if not confidence_scores:
            return 0.0
        
        base_confidence = np.mean(list(confidence_scores.values()))
        
        # Adjust confidence based on market volatility
        if market_volatility > 0.7:  # High volatility reduces confidence
            volatility_penalty = (market_volatility - 0.7) * 0.5
            adjusted_confidence = base_confidence * (1 - volatility_penalty)
        elif market_volatility < 0.3:  # Low volatility increases confidence
            volatility_bonus = (0.3 - market_volatility) * 0.3
            adjusted_confidence = min(0.95, base_confidence * (1 + volatility_bonus))
        else:
            adjusted_confidence = base_confidence
        
        return max(0.05, min(0.95, adjusted_confidence))
    
    def _classify_enhanced_signal(self, price_change_pct: float, confidence: float, market_volatility: float) -> str:
        """Enhanced signal classification with improved thresholds and volatility adjustment."""
        abs_change = abs(price_change_pct) * 100  # Convert to percentage
        
        # Adjust thresholds based on market volatility
        volatility_multiplier = 1.0 + (market_volatility - 0.5) * 0.5
        
        # Enhanced signal classification logic
        if confidence >= 0.8 and abs_change >= (2.5 * volatility_multiplier):
            return "VERY_STRONG"
        elif confidence >= 0.7 and abs_change >= (2.0 * volatility_multiplier):
            return "STRONG"
        elif confidence >= 0.6 and abs_change >= (1.5 * volatility_multiplier):
            return "MODERATE"
        elif confidence >= 0.4 and abs_change >= (1.0 * volatility_multiplier):
            return "WEAK"
        else:
            return "NEUTRAL"
    
    def _meets_trading_threshold(self, confidence: float, signal_strength: str) -> bool:
        """Determine if prediction meets minimum trading thresholds."""
        signal_score = self.signal_hierarchy.get(signal_strength, 0)
        min_signal_score = self.signal_hierarchy.get(self.min_signal_strength, 3)
        
        return (confidence >= self.min_confidence_threshold and
                signal_score >= min_signal_score)