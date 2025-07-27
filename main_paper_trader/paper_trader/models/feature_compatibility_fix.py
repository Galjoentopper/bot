#!/usr/bin/env python3
"""
Feature Compatibility Fix Module

This module provides utility functions for handling feature compatibility
between training and inference phases in the paper trading system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def align_features_with_training(
    features_df: pd.DataFrame,
    training_features: List[str],
    fill_missing: bool = True,
    fill_value: float = 0.0,
    feature_type: str = "lstm"
) -> pd.DataFrame:
    """
    Align inference features with training features.
    
    Args:
        features_df: DataFrame with current features
        training_features: List of features expected by the model
        fill_missing: Whether to fill missing features with default values
        fill_value: Value to use for missing features
        feature_type: Type of features ("lstm" or "xgboost")
        
    Returns:
        DataFrame with aligned features
    """
    try:
        aligned_df = features_df.copy()
        
        # For LSTM models, ensure we use the correct feature set based on expected count
        if feature_type == "lstm":
            # Define feature sets
            LSTM_FEATURES = [
                'price_vs_ema_30min', 'price_vs_ema_1h', 'price_vs_ema_2h', 'price_vs_ema_4h',
                'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'volume_change_24h',
                'atr_14', 'adx_14', 'momentum_30min', 'volume_ema_ratio'
            ]
            LSTM_FEATURES_LEGACY = [
                'close', 'volume', 'returns', 'log_returns',
                'volatility_20', 'atr_ratio', 'rsi', 'macd', 'bb_position',
                'volume_ratio', 'price_vs_ema9', 'price_vs_ema21',
                'buying_pressure', 'selling_pressure', 'spread_ratio',
                'momentum_10', 'price_zscore_20'
            ]
            
            # Determine which feature set to use based on expected count
            expected_count = len(training_features)
            if expected_count == 17:
                # Use modern LSTM features
                training_features = LSTM_FEATURES
                logger.info(f"Using modern LSTM features (17 features) for model expecting {expected_count}")
            elif expected_count == 36:
                # Use legacy LSTM features
                training_features = LSTM_FEATURES_LEGACY
                logger.info(f"Using legacy LSTM features (36 features) for model expecting {expected_count}")
            else:
                logger.warning(f"Unexpected feature count {expected_count}, using provided training_features")
        
        # Add missing features
        missing_features = [f for f in training_features if f not in aligned_df.columns]
        if missing_features:
            if fill_missing:
                for feature in missing_features:
                    aligned_df[feature] = fill_value
                logger.debug(f"Added {len(missing_features)} missing features with value {fill_value}")
            else:
                logger.warning(f"Missing features not filled: {missing_features}")
        
        # Reorder columns to match training order
        available_features = [f for f in training_features if f in aligned_df.columns]
        aligned_df = aligned_df[available_features]
        
        return aligned_df
        
    except Exception as e:
        logger.error(f"Error aligning features: {e}")
        return features_df


def validate_scaler_compatibility(
    scaler: StandardScaler,
    features_df: pd.DataFrame,
    expected_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate that scaler is compatible with current features.
    
    Args:
        scaler: Trained StandardScaler
        features_df: Current features DataFrame
        expected_features: Expected feature names
        
    Returns:
        Dictionary with compatibility information
    """
    try:
        result = {
            'compatible': True,
            'scaler_features': getattr(scaler, 'n_features_in_', None),
            'current_features': len(features_df.columns),
            'issues': []
        }
        
        # Check feature count
        if result['scaler_features'] is not None:
            if result['scaler_features'] != result['current_features']:
                result['compatible'] = False
                result['issues'].append(
                    f"Feature count mismatch: scaler expects {result['scaler_features']}, "
                    f"got {result['current_features']}"
                )
        
        # Check for NaN values
        nan_count = features_df.isnull().sum().sum()
        if nan_count > 0:
            result['issues'].append(f"Found {nan_count} NaN values in features")
        
        # Check for infinite values
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            result['issues'].append(f"Found {inf_count} infinite values in features")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating scaler compatibility: {e}")
        return {
            'compatible': False,
            'scaler_features': None,
            'current_features': len(features_df.columns) if features_df is not None else 0,
            'issues': [f"Validation error: {e}"]
        }


def prepare_lstm_sequence_safe(
    features_df: pd.DataFrame,
    sequence_length: int = 96,
    lstm_features: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Safely prepare LSTM sequence from features.
    
    Args:
        features_df: Features DataFrame
        sequence_length: Required sequence length
        lstm_features: List of LSTM feature names
        
    Returns:
        Prepared sequence array or None if insufficient data
    """
    try:
        if features_df is None or len(features_df) < sequence_length:
            logger.warning(
                f"Insufficient data for LSTM sequence: {len(features_df) if features_df is not None else 0}/{sequence_length}"
            )
            return None
        
        # Select LSTM features if specified
        if lstm_features:
            available_lstm_features = [f for f in lstm_features if f in features_df.columns]
            if len(available_lstm_features) != len(lstm_features):
                missing = [f for f in lstm_features if f not in features_df.columns]
                logger.warning(f"Missing LSTM features: {missing}")
            
            if available_lstm_features:
                sequence_data = features_df[available_lstm_features].tail(sequence_length)
            else:
                logger.error("No LSTM features available")
                return None
        else:
            sequence_data = features_df.tail(sequence_length)
        
        # Check for NaN values
        if sequence_data.isnull().any().any():
            logger.warning("NaN values found in LSTM sequence, filling with forward fill")
            sequence_data = sequence_data.fillna(method='ffill').fillna(0)
        
        # Convert to numpy array
        sequence_array = sequence_data.values
        
        # Reshape for LSTM: (1, sequence_length, features)
        sequence_array = sequence_array.reshape(1, sequence_length, -1)
        
        return sequence_array
        
    except Exception as e:
        logger.error(f"Error preparing LSTM sequence: {e}")
        return None


def handle_missing_lstm_delta(
    features_df: pd.DataFrame,
    lstm_prediction: Optional[float] = None,
    default_value: float = 0.0
) -> pd.DataFrame:
    """
    Handle missing lstm_delta feature by adding it if needed.
    
    Args:
        features_df: Features DataFrame
        lstm_prediction: LSTM prediction to use as delta
        default_value: Default value if no prediction available
        
    Returns:
        DataFrame with lstm_delta feature
    """
    try:
        result_df = features_df.copy()
        
        if 'lstm_delta' not in result_df.columns:
            if lstm_prediction is not None:
                # Use the LSTM prediction as delta
                result_df['lstm_delta'] = lstm_prediction
                logger.debug(f"Added lstm_delta feature with prediction value: {lstm_prediction}")
            else:
                # Use default value
                result_df['lstm_delta'] = default_value
                logger.debug(f"Added lstm_delta feature with default value: {default_value}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error handling missing lstm_delta: {e}")
        return features_df


def diagnose_compatibility_issues(
    features_df: pd.DataFrame,
    lstm_features: List[str],
    xgb_features: List[str],
    scaler: Optional[StandardScaler] = None
) -> Dict[str, Any]:
    """
    Diagnose compatibility issues between current features and model requirements.
    
    Args:
        features_df: Current features DataFrame
        lstm_features: Required LSTM features
        xgb_features: Required XGBoost features
        scaler: Optional scaler for additional validation
        
    Returns:
        Dictionary with detailed diagnosis
    """
    try:
        diagnosis = {
            'overall_compatible': True,
            'lstm_diagnosis': {},
            'xgboost_diagnosis': {},
            'scaler_diagnosis': {},
            'recommendations': []
        }
        
        current_features = set(features_df.columns) if features_df is not None else set()
        
        # LSTM diagnosis
        lstm_missing = [f for f in lstm_features if f not in current_features]
        lstm_extra = [f for f in current_features if f not in lstm_features]
        
        diagnosis['lstm_diagnosis'] = {
            'missing_features': lstm_missing,
            'extra_features': lstm_extra,
            'compatible': len(lstm_missing) == 0
        }
        
        if lstm_missing:
            diagnosis['overall_compatible'] = False
            diagnosis['recommendations'].append(
                f"Add missing LSTM features: {lstm_missing[:5]}{'...' if len(lstm_missing) > 5 else ''}"
            )
        
        # XGBoost diagnosis
        xgb_missing = [f for f in xgb_features if f not in current_features]
        xgb_extra = [f for f in current_features if f not in xgb_features]
        
        diagnosis['xgboost_diagnosis'] = {
            'missing_features': xgb_missing,
            'extra_features': xgb_extra,
            'compatible': len(xgb_missing) == 0
        }
        
        if xgb_missing:
            diagnosis['overall_compatible'] = False
            diagnosis['recommendations'].append(
                f"Add missing XGBoost features: {xgb_missing[:5]}{'...' if len(xgb_missing) > 5 else ''}"
            )
        
        # Scaler diagnosis
        if scaler is not None and features_df is not None:
            scaler_validation = validate_scaler_compatibility(scaler, features_df)
            diagnosis['scaler_diagnosis'] = scaler_validation
            
            if not scaler_validation['compatible']:
                diagnosis['overall_compatible'] = False
                diagnosis['recommendations'].extend([
                    f"Scaler issue: {issue}" for issue in scaler_validation['issues']
                ])
        
        # Data quality checks
        if features_df is not None:
            nan_count = features_df.isnull().sum().sum()
            if nan_count > 0:
                diagnosis['recommendations'].append(f"Handle {nan_count} NaN values in features")
            
            inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                diagnosis['recommendations'].append(f"Handle {inf_count} infinite values in features")
        
        return diagnosis
        
    except Exception as e:
        logger.error(f"Error diagnosing compatibility issues: {e}")
        return {
            'overall_compatible': False,
            'lstm_diagnosis': {'error': str(e)},
            'xgboost_diagnosis': {'error': str(e)},
            'scaler_diagnosis': {'error': str(e)},
            'recommendations': [f"Diagnosis failed: {e}"]
        }