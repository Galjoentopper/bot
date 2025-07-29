import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta
from typing import Dict, List, Tuple, Union, Optional

class FeatureFactory:
    """
    Feature Factory for creating consistent features across different models and time windows.
    
    This factory handles feature engineering, scaling, and preparation for different
    model types (LSTM, XGBoost) and different time windows to capture various
    market dynamics at different time scales.
    """
    
    def __init__(self, base_data: pd.DataFrame):
        """
        Initialize the Feature Factory with base price data.
        
        Args:
            base_data: DataFrame with columns including 'open', 'high', 'low', 'close', 'volume'
        """
        self.base_data = base_data.copy()
        self.calculated_features = {}  # Cache for calculated features
        self.scalers = {}  # Store scalers for each model type and window
        
        # Initialize indicator cache
        self.indicator_cache = {}
        
        # Configure which features to use for each model type
        self.feature_configs = {
            'lstm': {
                'price_features': ['close', 'open', 'high', 'low'],
                'volume_features': ['volume'],
                'momentum_indicators': ['rsi', 'macd', 'macd_signal', 'macd_hist'],
                'volatility_indicators': ['atr', 'bbands_upper', 'bbands_middle', 'bbands_lower'],
                'trend_indicators': ['ema5', 'ema10', 'ema20', 'sma5', 'sma10', 'sma20'],
                'custom_features': ['price_change', 'volume_change', 'price_volume_ratio']
            },
            'xgboost': {
                'price_features': ['close', 'open', 'high', 'low'],
                'volume_features': ['volume'],
                'momentum_indicators': ['rsi', 'macd', 'macd_signal', 'macd_hist', 'roc', 'stoch_k', 'stoch_d'],
                'volatility_indicators': ['atr', 'bbands_upper', 'bbands_middle', 'bbands_lower', 'natr'],
                'trend_indicators': ['ema5', 'ema10', 'ema20', 'sma5', 'sma10', 'sma20', 'adx'],
                'custom_features': ['price_change', 'volume_change', 'price_volume_ratio', 'high_low_ratio', 'close_to_high', 'close_to_low']
            }
        }
    
    def calculate_all_technical_indicators(self) -> None:
        """Calculate and cache all possible technical indicators from the base data."""
        data = self.base_data
        
        # Ensure we have the required columns
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Price data
        close = data['close']
        high = data['high']
        low = data['low']
        open_price = data['open']
        volume = data['volume']
        
        # Moving Averages (different periods)
        for period in [5, 10, 20, 50, 100, 200]:
            self.indicator_cache[f'sma{period}'] = ta.trend.sma_indicator(close, window=period)
            self.indicator_cache[f'ema{period}'] = ta.trend.ema_indicator(close, window=period)
        
        # Momentum Indicators
        self.indicator_cache['rsi'] = ta.momentum.rsi(close, window=14)
        
        # MACD
        macd_diff, macd_signal = ta.trend.macd_diff(close), ta.trend.macd_signal(close)
        self.indicator_cache['macd'] = macd_diff
        self.indicator_cache['macd_signal'] = macd_signal
        self.indicator_cache['macd_hist'] = macd_diff - macd_signal
        
        # Rate of Change
        self.indicator_cache['roc'] = ta.momentum.roc(close, window=10)
        
        # Stochastic Oscillator
        self.indicator_cache['stoch_k'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        self.indicator_cache['stoch_d'] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)
        
        # Volatility Indicators
        self.indicator_cache['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
        self.indicator_cache['natr'] = ta.volatility.average_true_range(high, low, close, window=14) / close * 100
        
        # Bollinger Bands
        bb_high = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
        bb_low = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
        bb_mid = ta.volatility.bollinger_mavg(close, window=20)
        self.indicator_cache['bbands_upper'] = bb_high
        self.indicator_cache['bbands_middle'] = bb_mid
        self.indicator_cache['bbands_lower'] = bb_low
        
        # Trend Indicators
        self.indicator_cache['adx'] = ta.trend.adx(high, low, close, window=14)
        
        # Custom Features
        self.indicator_cache['price_change'] = close.diff().fillna(0)
        self.indicator_cache['volume_change'] = volume.diff().fillna(0)
        self.indicator_cache['price_volume_ratio'] = close / (volume + 1)  # Avoid division by zero
        self.indicator_cache['high_low_ratio'] = high / (low + 1e-8)  # Avoid division by zero
        self.indicator_cache['close_to_high'] = (close - low) / (high - low + 1e-8)
        self.indicator_cache['close_to_low'] = (high - close) / (high - low + 1e-8)
    
    def get_features_for_model(self, model_type: str, window_size: int) -> Dict[str, np.ndarray]:
        """
        Get features for a specific model type and window size.
        
        Args:
            model_type: Type of model ('lstm' or 'xgboost')
            window_size: Size of the time window in days (e.g., 30, 60, 90)
            
        Returns:
            Dictionary of feature arrays appropriate for the model type and window
        """
        cache_key = f"{model_type}_{window_size}"
        
        # Return cached results if available
        if cache_key in self.calculated_features:
            return self.calculated_features[cache_key]
        
        # Calculate indicators if not already done
        if not self.indicator_cache:
            self.calculate_all_technical_indicators()
        
        # Prepare features based on model type
        if model_type == "lstm":
            features, scaler = self._prepare_lstm_features(window_size)
        elif model_type == "xgboost":
            features, scaler = self._prepare_xgboost_features(window_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cache results
        self.calculated_features[cache_key] = features
        self.scalers[cache_key] = scaler
        
        return features
    
    def _prepare_lstm_features(self, window_size: int) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
        """
        Prepare features specifically for LSTM models with the given window size.
        
        LSTM models require sequences of data, so this method creates windowed
        sequences of the specified length from all features.
        
        Args:
            window_size: Size of the time window in days
            
        Returns:
            Tuple of (features dictionary, scaler used)
        """
        # Get config for LSTM features
        config = self.feature_configs['lstm']
        
        # Gather all indicators specified in the config
        feature_dict = {}
        for feature_type, feature_list in config.items():
            for feature_name in feature_list:
                if feature_name in self.indicator_cache:
                    feature_dict[feature_name] = self.indicator_cache[feature_name]
                elif feature_name in self.base_data.columns:
                    feature_dict[feature_name] = self.base_data[feature_name]
        
        # Convert to DataFrame for easier handling
        feature_df = pd.DataFrame(feature_dict)
        
        # Handle NaN values
        feature_df = feature_df.ffill().bfill().fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_df)
        
        # Create sequences for LSTM
        X_sequences = []
        for i in range(len(scaled_data) - window_size):
            X_sequences.append(scaled_data[i:i+window_size])
        
        # Convert to numpy array with shape (samples, time steps, features)
        X_sequences = np.array(X_sequences)
        
        # Return the features and scaler
        return {'X': X_sequences, 'feature_names': feature_df.columns.tolist()}, scaler
    
    def _prepare_xgboost_features(self, window_size: int) -> Tuple[Dict[str, np.ndarray], Optional[StandardScaler]]:
        """
        Prepare features specifically for XGBoost models with the given window size.
        
        XGBoost requires tabular data, so this method creates aggregated features
        over the specified window lengths (e.g., min, max, mean of features over the window).
        
        Args:
            window_size: Size of the time window in days
            
        Returns:
            Tuple of (features dictionary, scaler used)
        """
        # Get config for XGBoost features
        config = self.feature_configs['xgboost']
        
        # Gather all indicators specified in the config
        feature_dict = {}
        for feature_type, feature_list in config.items():
            for feature_name in feature_list:
                if feature_name in self.indicator_cache:
                    feature_dict[feature_name] = self.indicator_cache[feature_name]
                elif feature_name in self.base_data.columns:
                    feature_dict[feature_name] = self.base_data[feature_name]
        
        # Convert to DataFrame for easier handling
        feature_df = pd.DataFrame(feature_dict)
        
        # Handle NaN values
        feature_df = feature_df.ffill().bfill().fillna(0)
        
        # Create window aggregations (tabular features for XGBoost)
        agg_features = []
        
        for i in range(window_size, len(feature_df)):
            window = feature_df.iloc[i-window_size:i]
            
            # Current values
            current = feature_df.iloc[i].to_dict()
            
            # Window aggregations
            aggs = {
                f"{col}_mean": window[col].mean() for col in window.columns
            }
            aggs.update({
                f"{col}_std": window[col].std() for col in window.columns
            })
            aggs.update({
                f"{col}_min": window[col].min() for col in window.columns
            })
            aggs.update({
                f"{col}_max": window[col].max() for col in window.columns
            })
            
            # Trend features
            aggs.update({
                f"{col}_trend": (window[col].iloc[-1] - window[col].iloc[0]) / (window[col].iloc[0] + 1e-8)
                for col in window.columns
            })
            
            # Combine current and aggregated
            row = {**current, **aggs}
            agg_features.append(row)
        
        # Convert to DataFrame
        X_tabular = pd.DataFrame(agg_features)
        
        # Handle NaN values again (aggregations might create NaNs)
        X_tabular = X_tabular.fillna(0)
        
        # No need to scale for XGBoost (it handles varying scales well)
        # But return unscaled data and None for scaler for consistency
        return {'X': X_tabular.values, 'feature_names': X_tabular.columns.tolist()}, None
    
    def get_prediction_features(self, model_type: str, window_size: int, latest_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Get features for making a prediction using the latest data.
        
        This method is used during live trading to prepare features for the latest data point.
        
        Args:
            model_type: Type of model ('lstm' or 'xgboost')
            window_size: Size of the time window in days
            latest_data: Optional latest data to append to base_data (for live trading)
            
        Returns:
            Feature array ready for model prediction
        """
        # If latest data is provided, temporarily append it to base_data
        original_data = self.base_data.copy()
        if latest_data is not None:
            self.base_data = pd.concat([self.base_data, latest_data]).reset_index(drop=True)
            
        # Recalculate indicators with the updated data
        self.indicator_cache = {}
        self.calculate_all_technical_indicators()
        
        # Get features
        cache_key = f"{model_type}_{window_size}"
        
        if model_type == "lstm":
            # For LSTM, we need the last sequence of window_size
            # Get the scaler if already cached
            scaler = self.scalers.get(cache_key)
            if scaler is None:
                # If not cached, create a new scaler
                _, scaler = self._prepare_lstm_features(window_size)
                self.scalers[cache_key] = scaler
            
            # Get the config for LSTM features
            config = self.feature_configs['lstm']
            
            # Gather features
            feature_dict = {}
            for feature_type, feature_list in config.items():
                for feature_name in feature_list:
                    if feature_name in self.indicator_cache:
                        feature_dict[feature_name] = self.indicator_cache[feature_name]
                    elif feature_name in self.base_data.columns:
                        feature_dict[feature_name] = self.base_data[feature_name]
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(feature_dict)
            feature_df = feature_df.ffill().bfill().fillna(0)
            
            # Scale
            scaled_data = scaler.transform(feature_df)
            
            # Get the last window
            last_sequence = scaled_data[-window_size:].reshape(1, window_size, -1)
            prediction_features = last_sequence
            
        elif model_type == "xgboost":
            # For XGBoost, we need the aggregated features for the last window
            # XGBoost aggregations are more complex, so we'll recalculate
            features, _ = self._prepare_xgboost_features(window_size)
            
            # Get the last row (most recent aggregated features)
            prediction_features = features['X'][-1:] if len(features['X']) > 0 else None
        
        # Restore original data
        self.base_data = original_data
        
        return prediction_features