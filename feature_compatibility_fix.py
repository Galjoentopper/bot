#!/usr/bin/env python3
"""
Feature Compatibility Fix Utilities

This module provides utility functions to ensure compatibility between 
train_hybrid_models.py and main_paper_trader.py by handling feature 
alignment, scaler compatibility, and model loading issues.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler


def align_features_with_training(
    features_df: pd.DataFrame,
    expected_features: List[str],
    feature_source: str = "inference"
) -> pd.DataFrame:
    """
    Align feature DataFrame with expected training features.
    
    Args:
        features_df: DataFrame with engineered features
        expected_features: List of features expected by the model
        feature_source: Source description for logging
        
    Returns:
        Aligned DataFrame with all expected features
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create a copy to avoid modifying the original
        aligned_df = features_df.copy()
        
        # Check for missing features
        missing_features = [f for f in expected_features if f not in aligned_df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in {feature_source}: {missing_features[:5]}...")
            
            # Fill missing features with appropriate defaults
            for feature in missing_features:
                if 'ratio' in feature.lower() or 'pct' in feature.lower():
                    # Ratio/percentage features default to 1.0 or 0.0
                    default_value = 1.0 if 'ratio' in feature.lower() else 0.0
                elif any(keyword in feature.lower() for keyword in ['surge', 'breakout', 'regime', 'alignment']):
                    # Binary indicator features default to 0
                    default_value = 0
                elif 'price' in feature.lower() and 'vs' in feature.lower():
                    # Price vs MA features default to 0 (at MA level)
                    default_value = 0.0
                elif feature.lower().startswith('returns'):
                    # Returns features default to 0
                    default_value = 0.0
                elif 'volatility' in feature.lower():
                    # Volatility features default to median market volatility
                    default_value = 0.02
                elif 'volume' in feature.lower():
                    # Volume features default to 1.0 (normal volume)
                    default_value = 1.0
                elif any(indicator in feature.lower() for indicator in ['rsi', 'macd', 'bb', 'stoch']):
                    # Technical indicators get neutral defaults
                    if 'rsi' in feature.lower():
                        default_value = 50.0  # Neutral RSI
                    elif 'bb_position' in feature.lower():
                        default_value = 0.5   # Middle of Bollinger Bands
                    elif 'stoch' in feature.lower():
                        default_value = 50.0  # Neutral Stochastic
                    else:
                        default_value = 0.0
                else:
                    # Generic numeric features default to 0
                    default_value = 0.0
                
                aligned_df[feature] = default_value
        
        # Ensure feature order matches expected order
        aligned_df = aligned_df.reindex(columns=expected_features, fill_value=0.0)
        
        # Handle any remaining NaN values
        aligned_df = aligned_df.fillna(0.0)
        
        # Replace infinite values
        aligned_df.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        logger.debug(f"Feature alignment completed: {len(expected_features)} features aligned for {feature_source}")
        return aligned_df
        
    except Exception as e:
        logger.error(f"Error aligning features for {feature_source}: {e}")
        # Return original DataFrame as fallback
        return features_df


def validate_scaler_compatibility(
    scaler: StandardScaler,
    features_df: pd.DataFrame,
    symbol: str,
    window: int = None
) -> Tuple[bool, int, int]:
    """
    Validate if scaler is compatible with the provided features.
    
    Args:
        scaler: StandardScaler object
        features_df: DataFrame with features
        symbol: Symbol name for logging
        window: Window number for logging
        
    Returns:
        Tuple of (is_compatible, expected_features, actual_features)
    """
    logger = logging.getLogger(__name__)
    
    try:
        expected_features = getattr(scaler, 'n_features_in_', None)
        actual_features = len(features_df.columns)
        
        if expected_features is None:
            logger.warning(f"Scaler for {symbol} window {window} has no n_features_in_ attribute")
            return True, 0, actual_features  # Assume compatible
        
        is_compatible = expected_features == actual_features
        
        if not is_compatible:
            logger.warning(
                f"Scaler compatibility issue for {symbol} window {window}: "
                f"expected {expected_features} features, got {actual_features}"
            )
        
        return is_compatible, expected_features, actual_features
        
    except Exception as e:
        logger.error(f"Error validating scaler compatibility for {symbol}: {e}")
        return False, 0, len(features_df.columns)


def prepare_lstm_sequence_safe(
    features_df: pd.DataFrame,
    lstm_features: List[str],
    sequence_length: int = 96,
    symbol: str = "",
    scaler: Optional[StandardScaler] = None
) -> Optional[np.ndarray]:
    """
    Safely prepare LSTM sequence with proper feature alignment and scaling.
    
    Args:
        features_df: DataFrame with all features
        lstm_features: List of features expected by LSTM model
        sequence_length: Required sequence length
        symbol: Symbol name for logging
        scaler: Optional scaler for feature normalization
        
    Returns:
        Prepared sequence array or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Align features with LSTM expectations
        aligned_features = align_features_with_training(
            features_df, lstm_features, f"LSTM-{symbol}"
        )
        
        # Check if we have enough data
        if len(aligned_features) < sequence_length:
            logger.warning(
                f"Insufficient data for LSTM sequence for {symbol}: "
                f"need {sequence_length}, have {len(aligned_features)}"
            )
            # Pad with zeros if needed
            padding_needed = sequence_length - len(aligned_features)
            padding_df = pd.DataFrame(
                np.zeros((padding_needed, len(lstm_features))),
                columns=lstm_features,
                index=range(padding_needed)
            )
            aligned_features = pd.concat([padding_df, aligned_features], ignore_index=True)
        
        # Get the last sequence_length rows
        sequence_data = aligned_features[lstm_features].tail(sequence_length)
        
        # Apply scaling if provided and compatible
        if scaler is not None:
            is_compatible, expected_features, actual_features = validate_scaler_compatibility(
                scaler, sequence_data, symbol
            )
            
            if is_compatible:
                try:
                    sequence_data_scaled = scaler.transform(sequence_data.values)
                    sequence_data = pd.DataFrame(
                        sequence_data_scaled, 
                        columns=lstm_features,
                        index=sequence_data.index
                    )
                    logger.debug(f"Applied scaling to LSTM features for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to apply scaling for {symbol}: {e}")
            else:
                logger.warning(f"Scaler incompatible for {symbol}, skipping scaling")
        
        # Convert to numpy array
        sequence_array = sequence_data.values
        
        # Handle any remaining invalid values
        sequence_array = np.nan_to_num(sequence_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape for LSTM: (1, sequence_length, n_features)
        lstm_sequence = sequence_array.reshape(1, sequence_length, len(lstm_features))
        
        logger.debug(f"Successfully prepared LSTM sequence for {symbol}: {lstm_sequence.shape}")
        return lstm_sequence
        
    except Exception as e:
        logger.error(f"Error preparing LSTM sequence for {symbol}: {e}")
        return None


def create_feature_mapping() -> Dict[str, str]:
    """
    Create mapping between training features and inference feature names.
    Handles cases where feature names might differ slightly between contexts.
    
    Returns:
        Dictionary mapping inference features to training features
    """
    return {
        # Handle common naming variations
        'macd_histogram': 'macd_hist',
        'macd_hist': 'macd_histogram', 
        'bb_percent': 'bb_position',
        'bb_position': 'bb_percent',
        'volume_sma_20': 'volume_sma',
        'rsi_14': 'rsi',
        'ema_20': 'ema_21',  # Close approximations
        'sma_21': 'sma_20',
        # Add more mappings as needed
    }


def handle_missing_lstm_delta(
    features_df: pd.DataFrame,
    lstm_prediction: Optional[float] = None,
    current_price: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing lstm_delta feature by computing it from LSTM prediction.
    
    Args:
        features_df: Features DataFrame
        lstm_prediction: LSTM model prediction
        current_price: Current market price
        
    Returns:
        DataFrame with lstm_delta feature added
    """
    logger = logging.getLogger(__name__)
    
    try:
        features_df = features_df.copy()
        
        if lstm_prediction is not None and current_price is not None:
            # Calculate lstm_delta as percentage change
            lstm_delta = (lstm_prediction - current_price) / current_price
            features_df['lstm_delta'] = lstm_delta
            logger.debug("Added lstm_delta feature from LSTM prediction")
        elif 'lstm_delta' not in features_df.columns:
            # Fill with neutral value if no prediction available
            features_df['lstm_delta'] = 0.0
            logger.debug("Added neutral lstm_delta feature")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error handling lstm_delta feature: {e}")
        return features_df


def diagnose_compatibility_issues(
    symbol: str,
    expected_features: List[str],
    actual_features: List[str],
    scaler: Optional[StandardScaler] = None
) -> Dict[str, Any]:
    """
    Diagnose compatibility issues between expected and actual features.
    
    Args:
        symbol: Symbol being analyzed
        expected_features: Features expected by the model
        actual_features: Features available in the data
        scaler: Optional scaler to check compatibility
        
    Returns:
        Dictionary with diagnosis results
    """
    logger = logging.getLogger(__name__)
    
    diagnosis = {
        'symbol': symbol,
        'total_expected': len(expected_features),
        'total_actual': len(actual_features),
        'missing_features': [],
        'extra_features': [],
        'scaler_compatible': True,
        'scaler_expected_features': None,
        'recommendations': []
    }
    
    try:
        # Find missing and extra features
        expected_set = set(expected_features)
        actual_set = set(actual_features)
        
        diagnosis['missing_features'] = list(expected_set - actual_set)
        diagnosis['extra_features'] = list(actual_set - expected_set)
        
        # Check scaler compatibility
        if scaler is not None:
            scaler_features = getattr(scaler, 'n_features_in_', None)
            diagnosis['scaler_expected_features'] = scaler_features
            
            if scaler_features is not None:
                diagnosis['scaler_compatible'] = scaler_features == len(actual_features)
        
        # Generate recommendations
        if diagnosis['missing_features']:
            diagnosis['recommendations'].append(
                f"Add missing features: {diagnosis['missing_features'][:5]}..."
            )
        
        if not diagnosis['scaler_compatible']:
            diagnosis['recommendations'].append(
                "Scaler dimension mismatch - consider retraining or feature alignment"
            )
        
        if len(diagnosis['missing_features']) > len(diagnosis['extra_features']) * 2:
            diagnosis['recommendations'].append(
                "Significant feature mismatch - verify feature engineering pipeline"
            )
        
        logger.info(f"Compatibility diagnosis for {symbol}: {len(diagnosis['missing_features'])} missing, "
                   f"{len(diagnosis['extra_features'])} extra features")
        
        return diagnosis
        
    except Exception as e:
        logger.error(f"Error diagnosing compatibility issues for {symbol}: {e}")
        diagnosis['error'] = str(e)
        return diagnosis