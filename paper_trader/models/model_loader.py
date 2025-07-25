"""Enhanced model loading and prediction ensemble for trading signals with multi-window support."""

import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
from .feature_engineer import LSTM_FEATURES, LSTM_SEQUENCE_LENGTH
from tensorflow.keras import backend as K

# Import settings for configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import TradingSettings

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


def create_comprehensive_custom_objects():
    """Create a comprehensive custom object scope for loading legacy Keras models."""
    
    custom_objects = {}
    
    # Add all standard Keras classes with legacy module path compatibility
    keras_classes = {
        # Models - handle both current and legacy module paths
        'Functional': tf.keras.Model,
        'Sequential': tf.keras.Sequential,
        'Model': tf.keras.Model,
        # Legacy Keras module paths
        'keras.src.engine.functional.Functional': tf.keras.Model,
        'keras.engine.network.Functional': tf.keras.Model,
        'keras.models.Functional': tf.keras.Model,
        
        # Core layers
        'Dense': tf.keras.layers.Dense,
        'Dropout': tf.keras.layers.Dropout,
        'InputLayer': tf.keras.layers.InputLayer,
        
        # Convolutional layers
        'Conv1D': tf.keras.layers.Conv1D,
        'ConviD': tf.keras.layers.Conv1D,  # Handle typo from error message
        'Conv2D': tf.keras.layers.Conv2D,
        'MaxPooling1D': tf.keras.layers.MaxPooling1D,
        'MaxPooling2D': tf.keras.layers.MaxPooling2D,
        
        # Recurrent layers
        'LSTM': tf.keras.layers.LSTM,
        'GRU': tf.keras.layers.GRU,
        'SimpleRNN': tf.keras.layers.SimpleRNN,
        
        # Normalization
        'BatchNormalization': tf.keras.layers.BatchNormalization,
        'LayerNormalization': tf.keras.layers.LayerNormalization,
        
        # Merge layers
        'Dot': tf.keras.layers.Dot,
        'Add': tf.keras.layers.Add,
        'Concatenate': tf.keras.layers.Concatenate,
        
        # Initializers
        'GlorotUniform': tf.keras.initializers.GlorotUniform,
        'GlorotNormal': tf.keras.initializers.GlorotNormal,
        'HeUniform': tf.keras.initializers.HeUniform,
        'HeNormal': tf.keras.initializers.HeNormal,
        'Zeros': tf.keras.initializers.Zeros,
        'Ones': tf.keras.initializers.Ones,
        'Orthogonal': tf.keras.initializers.Orthogonal,
        'RandomNormal': tf.keras.initializers.RandomNormal,
        'RandomUniform': tf.keras.initializers.RandomUniform,
        
        # Optimizers
        'Adam': tf.keras.optimizers.Adam,
        'SGD': tf.keras.optimizers.SGD,
        'RMSprop': tf.keras.optimizers.RMSprop,
        
        # Learning rate schedules
        'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay,
        'PiecewiseConstantDecay': tf.keras.optimizers.schedules.PiecewiseConstantDecay,
        
        # Losses
        'MeanSquaredError': tf.keras.losses.MeanSquaredError,
        'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError,
        'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy,
        'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy,
        'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy,
        
        # Metrics
        'Accuracy': tf.keras.metrics.Accuracy,
        'CategoricalAccuracy': tf.keras.metrics.CategoricalAccuracy,
        'SparseCategoricalAccuracy': tf.keras.metrics.SparseCategoricalAccuracy,
        'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError,
        'MeanSquaredError': tf.keras.metrics.MeanSquaredError,
    }
    
    custom_objects.update(keras_classes)
    
    # Add custom functions
    custom_objects.update({
        'directional_loss': directional_loss,
        'quantile_loss': quantile_loss,
        'function': lambda x: x,  # Generic function placeholder
        'builtins': lambda x: x,  # For builtin function references
    })
    
    return custom_objects


def create_lstm_model_architecture():
    """
    Recreate the LSTM model architecture for weight loading.
    This creates a model compatible with the saved weights from training.
    """
    # Input layer - matches the saved model
    inputs = tf.keras.layers.Input(shape=(96, 17), name='input_1')
    
    # Conv1D layer
    x = tf.keras.layers.Conv1D(
        filters=64, 
        kernel_size=3, 
        strides=1, 
        padding='causal', 
        activation='relu',
        name='conv1d'
    )(inputs)
    
    # BatchNormalization
    x = tf.keras.layers.BatchNormalization(
        axis=2, 
        momentum=0.99, 
        epsilon=0.001,
        name='batch_normalization'
    )(x)
    
    # MaxPooling1D
    x = tf.keras.layers.MaxPooling1D(
        pool_size=2, 
        strides=2, 
        padding='valid',
        name='max_pooling1d'
    )(x)
    
    # First LSTM layer
    x = tf.keras.layers.LSTM(
        units=256, 
        return_sequences=True, 
        activation='tanh',
        recurrent_activation='sigmoid',
        name='lstm'
    )(x)
    
    # BatchNormalization
    x = tf.keras.layers.BatchNormalization(
        axis=2, 
        momentum=0.99, 
        epsilon=0.001,
        name='batch_normalization_1'
    )(x)
    
    # Second LSTM layer
    x = tf.keras.layers.LSTM(
        units=128, 
        return_sequences=True, 
        activation='tanh',
        recurrent_activation='sigmoid',
        name='lstm_1'
    )(x)
    
    # BatchNormalization
    x = tf.keras.layers.BatchNormalization(
        axis=2, 
        momentum=0.99, 
        epsilon=0.001,
        name='batch_normalization_2'
    )(x)
    
    # Third LSTM layer
    lstm_out = tf.keras.layers.LSTM(
        units=64, 
        return_sequences=True, 
        activation='tanh',
        recurrent_activation='sigmoid',
        name='lstm_2'
    )(x)
    
    # BatchNormalization
    lstm_norm = tf.keras.layers.BatchNormalization(
        axis=2, 
        momentum=0.99, 
        epsilon=0.001,
        name='batch_normalization_3'
    )(lstm_out)
    
    # Attention mechanism
    attention = tf.keras.layers.Dense(
        units=128, 
        activation='tanh',
        name='dense'
    )(lstm_norm)
    
    attention_weights = tf.keras.layers.Dense(
        units=1, 
        activation='softmax',
        name='dense_1'
    )(attention)
    
    # Apply attention
    attended = tf.keras.layers.Dot(axes=1, name='dot')([attention_weights, lstm_norm])
    
    # Final dense layers
    x = tf.keras.layers.Dense(
        units=64, 
        activation='relu',
        name='dense_2'
    )(attended)
    
    x = tf.keras.layers.Dropout(rate=0.3, name='dropout')(x)
    
    x = tf.keras.layers.BatchNormalization(
        axis=2, 
        momentum=0.99, 
        epsilon=0.001,
        name='batch_normalization_4'
    )(x)
    
    x = tf.keras.layers.Dense(
        units=32, 
        activation='relu',
        name='dense_3'
    )(x)
    
    x = tf.keras.layers.Dropout(rate=0.3, name='dropout_1')(x)
    
    x = tf.keras.layers.BatchNormalization(
        axis=2, 
        momentum=0.99, 
        epsilon=0.001,
        name='batch_normalization_5'
    )(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        units=1, 
        activation='linear',
        name='dense_4'
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model')
    
    return model


def load_model_weights_only(model_path: str) -> Optional[tf.keras.Model]:
    """
    Load model by recreating architecture and loading weights separately.
    This is the most robust approach for TensorFlow version compatibility issues.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract the keras file
            with zipfile.ZipFile(model_path, 'r') as zip_file:
                zip_file.extractall(temp_path)
            
            weights_path = temp_path / 'model.weights.h5'
            if not weights_path.exists():
                return None
            
            # Create a new model with the same architecture as the saved one
            model = create_lstm_model_architecture()
            
            # Load weights
            model.load_weights(str(weights_path))
            
            return model
            
    except Exception as e:
        print(f"Weights-only loading failed: {e}")
        return None


def load_keras_model_robust(model_path: str, custom_objects: Optional[Dict] = None) -> Optional[tf.keras.Model]:
    """
    Robust model loading with multiple fallback strategies for TensorFlow compatibility.
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Merge custom objects
    default_custom_objects = create_comprehensive_custom_objects()
    if custom_objects:
        default_custom_objects.update(custom_objects)
    
    # Strategy 1: Load weights into new architecture (most robust for version conflicts)
    # Move this first since it's most likely to work for cross-version models
    model = load_model_weights_only(str(model_path))
    if model:
        return model
    
    # Strategy 2: Direct loading with comprehensive custom objects
    try:
        model = tf.keras.models.load_model(
            str(model_path), 
            compile=False, 
            custom_objects=default_custom_objects
        )
        return model
    except Exception as e:
        pass  # Silently continue to next strategy
    
    # Strategy 3: Try loading with custom object scope
    try:
        with tf.keras.utils.custom_object_scope(default_custom_objects):
            model = tf.keras.models.load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        pass
    
    # Strategy 4: Handle keras.src.engine.functional module issue specifically
    try:
        # Temporarily monkey-patch the missing module
        import sys
        import types
        
        # Create fake modules to handle the missing keras.src.engine.functional
        if 'keras' not in sys.modules:
            sys.modules['keras'] = types.ModuleType('keras')
        if 'keras.src' not in sys.modules:
            sys.modules['keras.src'] = types.ModuleType('keras.src')
        if 'keras.src.engine' not in sys.modules:
            sys.modules['keras.src.engine'] = types.ModuleType('keras.src.engine')
        if 'keras.src.engine.functional' not in sys.modules:
            functional_module = types.ModuleType('keras.src.engine.functional')
            functional_module.Functional = tf.keras.Model
            sys.modules['keras.src.engine.functional'] = functional_module
        
        # Now try loading again
        model = tf.keras.models.load_model(
            str(model_path), 
            compile=False, 
            custom_objects=default_custom_objects
        )
        return model
    except Exception as e:
        pass
    
    # All strategies failed
    return None

class WindowBasedModelLoader:
    """Loads and manages window-based pre-trained models for trading predictions."""
    
    def __init__(self, model_path: str = 'models'):
        self.model_path = Path(model_path)
        # Window-based model storage: {symbol: {window: model}}
        self.lstm_models: Dict[str, Dict[int, tf.keras.Model]] = {}
        self.xgb_models: Dict[str, Dict[int, xgb.XGBRegressor]] = {}
        self.caboose_models: Dict[str, Dict[int, cb.CatBoostRegressor]] = {}
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
            self.caboose_models[symbol] = {}
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
                            custom_objects = create_comprehensive_custom_objects()
                        try:
                            # Use robust loading function that handles TensorFlow version compatibility
                            model = load_keras_model_robust(str(lstm_file), custom_objects)
                            if model is not None:
                                # Compile the model with the custom loss function
                                model.compile(optimizer='adam', loss=directional_loss, metrics=['mae'])
                                self.lstm_models[symbol][window_num] = model
                                self.logger.info(f"Successfully loaded LSTM model {lstm_file}")
                            else:
                                self.logger.warning(f"Failed to load LSTM model {lstm_file} with all strategies")
                        except Exception as e:
                            self.logger.warning(f"Failed to load LSTM model {lstm_file}. Error: {e}")
                            # Try loading from separate files as final fallback
                            try:
                                arch_path = str(lstm_file).replace('.keras', '_architecture.json')
                                weights_path = str(lstm_file).replace('.keras', '_weights.h5')
                                
                                if os.path.exists(arch_path) and os.path.exists(weights_path):
                                    with open(arch_path, 'r') as json_file:
                                        model_json = json_file.read()
                                    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
                                    model.load_weights(weights_path)
                                    model.compile(optimizer='adam', loss=directional_loss, metrics=['mae'])
                                    self.lstm_models[symbol][window_num] = model
                                    self.logger.info(f"Loaded model from separate architecture and weights for {symbol} window {window_num}")
                                else:
                                    self.logger.warning(f"Could not find separate architecture/weights files for {lstm_file}")
                            except Exception as e2:
                                self.logger.error(f"All attempts to load {lstm_file} failed. Final error: {e2}")
                    except Exception as e:
                        self.logger.warning(f"Failed to process LSTM model file {lstm_file}: {e}")
            
            # Scan XGBoost models
            xgb_dir = self.model_path / 'xgboost'
            if xgb_dir.exists():
                for xgb_file in xgb_dir.glob(f"{symbol_lower}_window_*.json"):
                    try:
                        window_num = int(xgb_file.stem.split('_window_')[1])
                        windows_found.add(window_num)
                        
                        # Try loading as classifier first (since your models are classifiers)
                        model = xgb.XGBClassifier()
                        model.load_model(xgb_file)
                        self.xgb_models[symbol][window_num] = model
                        self.logger.debug(f"Loaded XGBoost classifier for {symbol} window {window_num}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load XGBoost model {xgb_file}: {e}")

            # Scan Caboose (CatBoost) models if available
            caboose_dir = self.model_path / 'caboose'
            if caboose_dir.exists():
                for cb_file in caboose_dir.glob(f"{symbol_lower}_window_*.cbm"):
                    try:
                        window_num = int(cb_file.stem.split('_window_')[1])
                        windows_found.add(window_num)

                        model = cb.CatBoostRegressor()
                        model.load_model(cb_file)
                        self.caboose_models[symbol][window_num] = model
                        self.logger.debug(
                            f"Loaded Caboose model for {symbol} window {window_num}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load Caboose model {cb_file}: {e}")
            
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
                for fc_file in feature_dir.glob(f"{symbol_lower}_window_*_selected.pkl"):
                    try:
                        window_num = int(fc_file.stem.split('_window_')[1].split('_')[0])
                        with open(fc_file, 'rb') as f:
                            cols = pickle.load(f)
                        self.feature_columns[symbol][window_num] = cols
                        windows_found.add(window_num)
                        self.logger.debug(
                            f"Loaded feature columns for {symbol} window {window_num}"
                        )
                    except (ValueError, Exception) as e:
                        self.logger.warning(
                            f"Failed to load feature columns {fc_file}: {e}"
                        )
            
            # Update available windows
            self.available_windows[symbol] = sorted(list(windows_found))
            
            # Check if we have at least one model
            has_models = (
                len(self.lstm_models[symbol]) > 0
                or len(self.xgb_models[symbol]) > 0
                or len(self.caboose_models[symbol]) > 0
            )
            
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
                'caboose_windows': list(self.caboose_models.get(symbol, {}).keys()),
                'scaler_windows': list(self.scalers.get(symbol, {}).keys()),
                'available_windows': self.available_windows.get(symbol, []),
                'total_windows': len(self.available_windows.get(symbol, []))
            }
        
        return status
    
    def get_optimal_window(self, symbol: str, market_volatility: float, trend_strength: float) -> Optional[int]:
        """Select optimal window based on market conditions from the 15 most recent windows."""
        if symbol not in self.available_windows or not self.available_windows[symbol]:
            return None

        windows = sorted(self.available_windows[symbol])

        # For high volatility, use the most recent windows (last 5)
        if market_volatility > 0.7:
            return max(windows[-5:])
        # For strong trends, use a window from the more recent ones
        elif trend_strength > 0.6 and len(windows) >= 10:
            return windows[-10]
        # Default: use the most recent window available
        else:
            return windows[-1] if windows else None
    
    def get_available_models(self, symbol: str, window: int) -> Dict[str, bool]:
        """Check which models are available for a specific symbol and window."""
        return {
            'lstm': window in self.lstm_models.get(symbol, {}),
            'xgb': window in self.xgb_models.get(symbol, {}),
            'caboose': window in self.caboose_models.get(symbol, {}),
            'scaler': window in self.scalers.get(symbol, {})
        }

class WindowBasedEnsemblePredictor:
    """Enhanced ensemble predictor with window-based model selection and improved confidence thresholds."""
    
    def __init__(self, model_loader: WindowBasedModelLoader, 
                 min_confidence_threshold: float = 0.6,
                 min_signal_strength: str = 'MODERATE',
                 settings: TradingSettings = None):
        if settings is None:
            settings = TradingSettings()
        self.settings = settings
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
        
        # Enhanced prediction weights and thresholds (prioritize XGBoost when LSTM fails)
        self.lstm_weight = 0.6
        self.xgb_weight = 0.4
        self.caboose_weight = 0.3
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
        
    async def predict(
        self,
        symbol: str,
        features: pd.DataFrame,
        market_volatility: float = 0.5,
        current_price: float | None = None,
    ) -> Optional[dict]:
        """Generate enhanced ensemble prediction with window-based model selection.

        Args:
            symbol: Trading symbol
            features: Feature DataFrame with historical data
            market_volatility: Market volatility measure
            current_price: Optional latest price. If ``None`` the last ``close``
                value from ``features`` is used.
        """
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
                # Allow computing lstm_delta later if missing
                if 'lstm_delta' in missing:
                    missing.remove('lstm_delta')
                if missing:
                    self.logger.warning(f"Missing required feature columns for {symbol}: {missing}")
                    return None
            
            # Get available models for the selected window
            available_models = self.model_loader.get_available_models(symbol, optimal_window)
            
            predictions = {}
            confidence_scores = {}
            model_details = {}
            
            lstm_pred = None
            lstm_conf = 0.0
            # LSTM Prediction with selected window (skip if models are corrupted)
            if available_models['lstm']:
                try:
                    lstm_pred, lstm_conf = await self._predict_lstm_window(
                        symbol, features, optimal_window, current_price
                    )
                    if lstm_pred is not None:
                        predictions['lstm'] = lstm_pred
                        confidence_scores['lstm'] = lstm_conf
                        model_details['lstm_window'] = optimal_window
                except Exception as e:
                    self.logger.warning(f"LSTM model failed for {symbol} window {optimal_window}, skipping: {e}")

            # Append lstm_delta to features if required
            if lstm_pred is not None and required_cols and 'lstm_delta' in required_cols:
                base_price = current_price if current_price is not None else features['close'].iloc[-1]
                delta = (lstm_pred - base_price) / base_price
                features = features.copy()
                features['lstm_delta'] = delta
            
            # XGBoost Prediction with selected window
            if available_models['xgb']:
                xgb_pred, xgb_conf = await self._predict_xgboost_window(
                    symbol, features, optimal_window, current_price
                )
                if xgb_pred is not None:
                    predictions['xgb'] = xgb_pred
                    confidence_scores['xgb'] = xgb_conf
                    model_details['xgb_window'] = optimal_window

            # Caboose (CatBoost) Prediction with selected window
            if available_models.get('caboose'):
                caboose_pred, caboose_conf = await self._predict_caboose_window(
                    symbol, features, optimal_window, current_price
                )
                if caboose_pred is not None:
                    predictions['caboose'] = caboose_pred
                    confidence_scores['caboose'] = caboose_conf
                    model_details['caboose_window'] = optimal_window
            
            if not predictions:
                self.logger.warning(f"No valid predictions for {symbol} with window {optimal_window}")
                return None
            
            # Calculate ensemble prediction with jump-specific weighting
            jump_features_active = self._count_jump_features(features)
            jump_probability = self._calculate_jump_probability(predictions, confidence_scores, jump_features_active)
            ensemble_pred = self._calculate_enhanced_ensemble(predictions, confidence_scores, jump_probability)

            # Calculate uncertainty and confidence
            uncertainty = self.calculate_prediction_uncertainty(predictions, market_volatility)
            avg_confidence = self._calculate_adjusted_confidence(confidence_scores, market_volatility)
            
            # Determine signal strength
            if current_price is None:
                current_price = float(features['close'].iloc[-1])
            else:
                current_price = float(current_price)
            price_change_pct = (ensemble_pred - current_price) / current_price
            
            # Enhanced signal classification with jump consideration
            signal_strength = self._classify_enhanced_signal(price_change_pct, avg_confidence, market_volatility, jump_probability)
            
            # Prepare result data first
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': ensemble_pred,
                'price_change_pct': price_change_pct,
                'confidence': avg_confidence,
                'uncertainty': uncertainty,
                'signal_strength': signal_strength,
                'jump_probability': jump_probability,
                'jump_features_active': jump_features_active,
                'optimal_window': optimal_window,
                'market_volatility': market_volatility,
                'individual_predictions': predictions,
                'individual_confidences': confidence_scores,
                'model_details': model_details,
                'timestamp': pd.Timestamp.now()
            }
            
            # Check if prediction meets minimum thresholds (pass result for strict conditions)
            meets_threshold = self._meets_trading_threshold(avg_confidence, signal_strength, result)
            result['meets_threshold'] = meets_threshold
            
            self.logger.debug(
                f"Enhanced prediction for {symbol}: {price_change_pct:.4f}% change, "
                f"confidence: {avg_confidence:.3f}, uncertainty: {uncertainty:.3f}, "
                f"window: {optimal_window}, meets_threshold: {meets_threshold}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced prediction for {symbol}: {e}")
            return None
    
    async def _predict_lstm_window(
        self,
        symbol: str,
        features: pd.DataFrame,
        window: int,
        current_price: float | None = None,
    ) -> Tuple[Optional[float], float]:
        """Generate LSTM prediction using specific window model."""
        try:
            if (symbol not in self.model_loader.lstm_models or 
                window not in self.model_loader.lstm_models[symbol]):
                return None, 0.0
            
            model = self.model_loader.lstm_models[symbol][window]
            scaler = self.model_loader.scalers[symbol].get(window)
            
            # Prepare features using the correct feature columns for this window
            feature_data = features.copy()
            
            # Use stored feature columns if available, otherwise fall back to TRAINING_FEATURES
            if (symbol in self.model_loader.feature_columns and 
                window in self.model_loader.feature_columns[symbol]):
                expected_feature_cols = self.model_loader.feature_columns[symbol][window]
                feature_cols = [col for col in expected_feature_cols if col in feature_data.columns]
                missing_features = [col for col in expected_feature_cols if col not in feature_data.columns]
                
                if missing_features:
                    self.logger.warning(f"Missing features for {symbol} window {window}: {missing_features}")
                    # Fill missing features with zeros
                    for col in missing_features:
                        feature_data[col] = 0.0
                    feature_cols = expected_feature_cols
                
                self.logger.debug(f"Using {len(feature_cols)} features for {symbol} window {window}: {feature_cols}")
            else:
                # Fallback to TRAINING_FEATURES for compatibility with older models
                from .feature_engineer import TRAINING_FEATURES
                feature_cols = [col for col in TRAINING_FEATURES if col in feature_data.columns]
                self.logger.warning(f"No feature columns found for {symbol} window {window}, using TRAINING_FEATURES fallback")
                
            if scaler is not None and feature_cols:
                try:
                    # Check if the scaler expects the same number of features
                    if hasattr(scaler, 'n_features_in_') and len(feature_cols) != scaler.n_features_in_:
                        self.logger.warning(
                            f"Feature count mismatch for {symbol} window {window}: "
                            f"have {len(feature_cols)} features, scaler expects {scaler.n_features_in_}. "
                            f"Skipping scaling."
                        )
                        # Don't scale if there's a mismatch
                    else:
                        feature_data[feature_cols] = scaler.transform(
                            feature_data[feature_cols].to_numpy()
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error scaling features for {symbol} window {window}: {e}. "
                        f"Proceeding without scaling."
                    )

            # Sanitize feature values before sequence creation
            if np.isnan(feature_data[feature_cols].to_numpy()).any() or np.isinf(feature_data[feature_cols].to_numpy()).any():
                self.logger.debug("Replacing invalid values in LSTM feature data")
                feature_data[feature_cols] = np.nan_to_num(
                    feature_data[feature_cols].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
                )
            
            # Create sequence for LSTM using fixed training length
            sequence_length = min(LSTM_SEQUENCE_LENGTH, len(feature_data))

            numeric_cols = feature_cols

            sequence = feature_data[numeric_cols].tail(sequence_length).values

            # Replace any remaining invalid values
            if not np.isfinite(sequence).all():
                self.logger.debug("Cleaning invalid values in LSTM sequence")
                sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

            # Pad sequence if needed to match training length
            if sequence.shape[0] < LSTM_SEQUENCE_LENGTH:
                padding = np.zeros((LSTM_SEQUENCE_LENGTH - sequence.shape[0], sequence.shape[1]))
                sequence = np.vstack([padding, sequence])

            sequence = sequence.reshape(1, LSTM_SEQUENCE_LENGTH, len(numeric_cols))
            
            # Make prediction (percentage change from current price)
            prediction = model.predict(sequence, verbose=0)[0][0]

            if current_price is None:
                current_price = features['close'].iloc[-1]
            predicted_price = float(current_price * (1 + prediction))

            # Enhanced confidence calculation
            price_volatility = features['close'].tail(10).std() / features['close'].tail(10).mean()
            prediction_error = abs(predicted_price - current_price) / current_price

            # Confidence based on prediction stability and volatility
            base_confidence = 1.0 - min(prediction_error * 2, 0.8)
            volatility_adjustment = max(0.1, 1.0 - price_volatility)
            confidence = min(0.95, max(0.1, base_confidence * volatility_adjustment))

            return predicted_price, confidence
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction for {symbol} window {window}: {e}")
            return None, 0.0
    
    async def _predict_xgboost_window(
        self,
        symbol: str,
        features: pd.DataFrame,
        window: int,
        current_price: float | None = None,
    ) -> Tuple[Optional[float], float]:
        """Generate XGBoost prediction using specific window model."""
        try:
            if (symbol not in self.model_loader.xgb_models or 
                window not in self.model_loader.xgb_models[symbol]):
                return None, 0.0
            
            model = self.model_loader.xgb_models[symbol][window]
            feature_columns = self.model_loader.feature_columns.get(symbol, {}).get(window)

            if not feature_columns:
                self.logger.warning(
                    f"No feature column list for {symbol} window {window}")
                return None, 0.0

            available_features = [c for c in feature_columns if c in features.columns]
            missing_features = [c for c in feature_columns if c not in features.columns]

            if missing_features:
                self.logger.warning(
                    f"Missing features for XGBoost prediction: {symbol} window {window}"
                    f" -> {missing_features}. Filling with zeros.")

            feature_data = (features.reindex(columns=feature_columns)
                             .fillna(0)
                             .tail(1))
            
            # Make prediction (probability of price increase)
            prob_up = model.predict_proba(feature_data)[0][1]

            if current_price is None:
                current_price = features['close'].iloc[-1]
            predicted_price = float(current_price * (1 + prob_up * self.settings.price_prediction_multiplier))

            # Enhanced confidence calculation for XGBoost
            price_volatility = features['close'].tail(10).std() / features['close'].tail(10).mean()
            prediction_error = abs(predicted_price - current_price) / current_price

            # Base confidence adjusted by prediction accuracy and market stability
            base_confidence = 0.75  # Higher base for XGBoost
            error_penalty = min(prediction_error * 1.5, 0.4)
            volatility_bonus = max(0, 0.2 - price_volatility)  # XGBoost performs better in stable markets

            confidence = min(0.9, max(0.2, base_confidence - error_penalty + volatility_bonus))

            return predicted_price, confidence
            
        except Exception as e:
            self.logger.error(f"Error in XGBoost prediction for {symbol} window {window}: {e}")
            return None, 0.0

    async def _predict_caboose_window(
        self,
        symbol: str,
        features: pd.DataFrame,
        window: int,
        current_price: float | None = None,
    ) -> Tuple[Optional[float], float]:
        """Generate Caboose (CatBoost) prediction using specific window model."""
        try:
            if (
                symbol not in self.model_loader.caboose_models
                or window not in self.model_loader.caboose_models[symbol]
            ):
                return None, 0.0

            model = self.model_loader.caboose_models[symbol][window]
            feature_columns = self.model_loader.feature_columns.get(symbol, {}).get(window)

            if not feature_columns:
                self.logger.warning(
                    f"No feature column list for {symbol} window {window}")
                return None, 0.0

            feature_data = (
                features.reindex(columns=feature_columns)
                .fillna(0)
                .tail(1)
            )

            prediction = model.predict(feature_data)[0]

            if current_price is None:
                current_price = features['close'].iloc[-1]
            price_volatility = features['close'].tail(10).std() / features['close'].tail(10).mean()
            prediction_error = abs(prediction - current_price) / current_price

            base_confidence = 0.7
            error_penalty = min(prediction_error * 1.2, 0.4)
            volatility_bonus = max(0, 0.25 - price_volatility)

            confidence = min(0.9, max(0.2, base_confidence - error_penalty + volatility_bonus))

            return float(prediction), confidence

        except Exception as e:
            self.logger.error(f"Error in Caboose prediction for {symbol} window {window}: {e}")
            return None, 0.0
    
    def _calculate_enhanced_ensemble(self, predictions: dict, confidence_scores: dict, jump_probability: float = 0.0) -> float:
        """Calculate enhanced weighted ensemble prediction with jump-detection adaptive weighting."""
        if not predictions:
            return 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        # Calculate adaptive weights based on confidence and jump probability
        max_confidence = max(confidence_scores.values()) if confidence_scores else 1.0
        
        for model_name, prediction in predictions.items():
            confidence = confidence_scores[model_name]
            
            # Base weight with jump-detection bias
            if model_name == 'lstm':
                # Give LSTM higher weight for jump detection as it's now trained for binary classification
                base_weight = self.lstm_weight * (1.0 + 0.3 * jump_probability)
            elif model_name == 'xgb':
                # XGBoost maintains its weight for jump detection
                base_weight = self.xgb_weight * (1.0 + 0.2 * jump_probability)
            elif model_name == 'caboose':
                base_weight = self.caboose_weight * (1.0 + 0.1 * jump_probability)
            else:
                base_weight = 0.5
            
            # Multi-timeframe confidence weighting with jump focus
            confidence_multiplier = (confidence / max_confidence) ** 0.5  # Square root for smoother scaling
            
            # Boost weight when prediction indicates jump potential
            if prediction > 0.5:  # Binary prediction > 0.5 indicates jump likelihood
                confidence_multiplier *= (1.0 + 0.2 * jump_probability)
            
            weight = base_weight * confidence_multiplier
            
            weighted_sum += prediction * weight
            total_weight += weight
        
        if total_weight == 0:
            return np.mean(list(predictions.values()))

        return weighted_sum / total_weight
        
    def _count_jump_features(self, features: pd.DataFrame) -> int:
        """Count how many jump-specific features are active."""
        jump_features = [
            'volume_surge_5', 'volume_surge_10', 'volatility_breakout', 'atr_breakout',
            'resistance_breakout', 'support_bounce', 'price_gap_up', 'momentum_convergence',
            'squeeze_breakout', 'market_momentum_alignment', 'strong_trend'
        ]
        
        active_count = 0
        latest_features = features.iloc[-1] if len(features) > 0 else {}
        
        for feature in jump_features:
            if feature in latest_features and latest_features[feature] > 0:
                active_count += 1
                
        return active_count
    
    def _calculate_jump_probability(self, predictions: dict, confidence_scores: dict, jump_features_active: int) -> float:
        """Calculate probability of a price jump based on model predictions and features."""
        if not predictions:
            return 0.0
            
        # Base probability from model ensemble (assuming binary predictions)
        avg_prediction = np.mean(list(predictions.values()))
        avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
        
        # Jump feature contribution (normalized)
        feature_contribution = min(jump_features_active / 8.0, 1.0)  # Normalize to max 8 features
        
        # Combine probabilities
        jump_probability = (
            0.5 * avg_prediction +        # Model predictions (50% weight)
            0.3 * avg_confidence +        # Model confidence (30% weight) 
            0.2 * feature_contribution    # Feature contribution (20% weight)
        )
        
        return min(max(jump_probability, 0.0), 1.0)

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
    
    def _classify_enhanced_signal(self, price_change_pct: float, confidence: float, market_volatility: float, jump_probability: float = 0.0) -> str:
        """Enhanced signal classification with jump probability consideration."""
        abs_change = abs(price_change_pct) * 100  # Convert to percentage
        
        # Adjust thresholds based on market volatility
        volatility_multiplier = 1.0 + (market_volatility - 0.5) * 0.5
        
        # Jump probability boosts signal strength
        jump_boost = jump_probability * 0.3  # Up to 30% boost
        effective_confidence = min(confidence + jump_boost, 1.0)
        
        # Enhanced signal classification logic with jump consideration
        if effective_confidence >= 0.8 and (abs_change >= (1.5 * volatility_multiplier) or jump_probability > 0.7):
            return "VERY_STRONG"
        elif effective_confidence >= 0.7 and (abs_change >= (1.0 * volatility_multiplier) or jump_probability > 0.6):
            return "STRONG"
        elif effective_confidence >= 0.6 and (abs_change >= (0.5 * volatility_multiplier) or jump_probability > 0.5):
            return "MODERATE"
        elif effective_confidence >= 0.3 and (abs_change >= (0.05 * volatility_multiplier) or jump_probability > 0.3):
            return "WEAK"
        else:
            return "NEUTRAL"
    
    def _meets_trading_threshold(self, confidence: float, signal_strength: str, prediction_data: dict = None) -> bool:
        """Determine if prediction meets minimum trading thresholds."""
        signal_score = self.signal_hierarchy.get(signal_strength, 0)
        min_signal_score = self.signal_hierarchy.get(self.min_signal_strength, 3)
        
        # Basic threshold check
        meets_basic = (confidence >= self.min_confidence_threshold and
                      signal_score >= min_signal_score)
        
        if not meets_basic:
            return False
        
        # Additional strict entry conditions if enabled
        if hasattr(self.settings, 'enable_strict_entry_conditions') and self.settings.enable_strict_entry_conditions:
            if prediction_data:
                # Check prediction uncertainty
                uncertainty = prediction_data.get('uncertainty', 0.0)
                if uncertainty > getattr(self.settings, 'max_prediction_uncertainty', 0.3):
                    return False
                
                # Check ensemble agreement count
                individual_predictions = prediction_data.get('individual_predictions', {})
                model_count = len(individual_predictions)
                min_agreement = getattr(self.settings, 'min_ensemble_agreement_count', 2)
                if model_count < min_agreement:
                    return False
                
                # Apply confidence boost for high-confidence predictions
                confidence_boost_threshold = getattr(self.settings, 'strong_signal_confidence_boost', 0.85)
                if confidence >= confidence_boost_threshold:
                    # Allow MODERATE signals if confidence is very high
                    if signal_strength in ['MODERATE', 'STRONG', 'VERY_STRONG']:
                        return True
        
        return meets_basic