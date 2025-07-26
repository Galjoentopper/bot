#!/usr/bin/env python3
"""
Model Compatibility Handler

This module provides the ModelCompatibilityHandler class that ensures seamless
compatibility between models trained with train_hybrid_models.py and used in
main_paper_trader.py by handling feature alignment, scaler compatibility,
and robust error handling.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from feature_compatibility_fix import (
    align_features_with_training,
    validate_scaler_compatibility,
    prepare_lstm_sequence_safe,
    handle_missing_lstm_delta,
    diagnose_compatibility_issues
)


class ModelCompatibilityHandler:
    """
    Handles compatibility between training and inference pipelines.
    
    This class ensures that:
    1. Features used during training match those used during inference
    2. Scalers have the correct dimensions for LSTM features  
    3. Sequence length is consistent between training and inference
    4. Robust error handling for missing features
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the compatibility handler.
        
        Args:
            models_dir: Directory containing saved models and metadata
        """
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded metadata
        self._feature_cache: Dict[str, Dict[str, List[str]]] = {}
        self._scaler_cache: Dict[str, StandardScaler] = {}
        self._compatibility_cache: Dict[str, Dict] = {}
        
        self.logger.info(f"ModelCompatibilityHandler initialized with models_dir: {models_dir}")
    
    def load_training_metadata(self, symbol: str, window: int) -> Dict[str, Any]:
        """
        Load metadata saved during training for a specific symbol and window.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCEUR')
            window: Training window number
            
        Returns:
            Dictionary containing training metadata
        """
        try:
            symbol_lower = symbol.lower().replace("-", "")
            cache_key = f"{symbol_lower}_window_{window}"
            
            if cache_key in self._compatibility_cache:
                return self._compatibility_cache[cache_key]
            
            metadata = {
                'symbol': symbol,
                'window': window,
                'lstm_features': [],
                'xgb_features': [],
                'scaler_features': None,
                'sequence_length': 96,  # Default from training
                'feature_columns_loaded': False,
                'scaler_loaded': False
            }
            
            # Load LSTM feature columns
            feature_dir = self.models_dir / "feature_columns"
            lstm_features_file = feature_dir / f"{symbol_lower}_window_{window}.pkl"
            
            if lstm_features_file.exists():
                try:
                    with open(lstm_features_file, 'rb') as f:
                        all_features = pickle.load(f)
                    # Filter out lstm_delta and other dynamic features for LSTM
                    metadata['lstm_features'] = [f for f in all_features if f != 'lstm_delta']
                    metadata['feature_columns_loaded'] = True
                    self.logger.debug(f"Loaded LSTM features for {symbol} window {window}: {len(metadata['lstm_features'])}")
                except Exception as e:
                    self.logger.warning(f"Failed to load LSTM features for {symbol} window {window}: {e}")
            
            # Load XGBoost selected features
            xgb_features_file = feature_dir / f"{symbol_lower}_window_{window}_selected.pkl"
            
            if xgb_features_file.exists():
                try:
                    with open(xgb_features_file, 'rb') as f:
                        metadata['xgb_features'] = pickle.load(f)
                    self.logger.debug(f"Loaded XGBoost features for {symbol} window {window}: {len(metadata['xgb_features'])}")
                except Exception as e:
                    self.logger.warning(f"Failed to load XGBoost features for {symbol} window {window}: {e}")
            
            # Load scaler information
            scaler_dir = self.models_dir / "scalers"
            scaler_file = scaler_dir / f"{symbol_lower}_window_{window}_scaler.pkl"
            
            if scaler_file.exists():
                try:
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                    metadata['scaler_features'] = getattr(scaler, 'n_features_in_', None)
                    metadata['scaler_loaded'] = True
                    # Cache the scaler
                    self._scaler_cache[cache_key] = scaler
                    self.logger.debug(f"Loaded scaler for {symbol} window {window}: {metadata['scaler_features']} features")
                except Exception as e:
                    self.logger.warning(f"Failed to load scaler for {symbol} window {window}: {e}")
            
            # Fallback to default features if nothing loaded
            if not metadata['feature_columns_loaded']:
                # Import default features
                try:
                    from paper_trader.models.feature_engineer import LSTM_FEATURES, TRAINING_FEATURES
                    metadata['lstm_features'] = LSTM_FEATURES.copy()
                    metadata['xgb_features'] = TRAINING_FEATURES.copy()
                    self.logger.info(f"Using default features for {symbol} window {window}")
                except ImportError:
                    self.logger.warning(f"Could not import default features for {symbol}")
            
            # Cache the metadata
            self._compatibility_cache[cache_key] = metadata
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading training metadata for {symbol} window {window}: {e}")
            return {
                'symbol': symbol,
                'window': window,
                'lstm_features': [],
                'xgb_features': [],
                'scaler_features': None,
                'sequence_length': 96,
                'error': str(e)
            }
    
    def align_lstm_features(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        window: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Align features for LSTM model compatibility.
        
        Args:
            features_df: DataFrame with engineered features
            symbol: Trading symbol
            window: Model window number
            
        Returns:
            Tuple of (aligned_features_df, expected_features_list)
        """
        try:
            metadata = self.load_training_metadata(symbol, window)
            expected_features = metadata['lstm_features']
            
            if not expected_features:
                self.logger.warning(f"No LSTM features found for {symbol} window {window}")
                return features_df, []
            
            # Align features with training expectations
            aligned_df = align_features_with_training(
                features_df, 
                expected_features,
                f"LSTM-{symbol}-W{window}"
            )
            
            self.logger.debug(f"Aligned LSTM features for {symbol} window {window}: {len(expected_features)} features")
            return aligned_df, expected_features
            
        except Exception as e:
            self.logger.error(f"Error aligning LSTM features for {symbol} window {window}: {e}")
            return features_df, []
    
    def align_xgboost_features(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        window: int,
        lstm_prediction: Optional[float] = None,
        current_price: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Align features for XGBoost model compatibility.
        
        Args:
            features_df: DataFrame with engineered features
            symbol: Trading symbol
            window: Model window number
            lstm_prediction: LSTM prediction for lstm_delta feature
            current_price: Current market price
            
        Returns:
            Tuple of (aligned_features_df, expected_features_list)
        """
        try:
            metadata = self.load_training_metadata(symbol, window)
            expected_features = metadata['xgb_features']
            
            if not expected_features:
                self.logger.warning(f"No XGBoost features found for {symbol} window {window}")
                return features_df, []
            
            # Handle lstm_delta feature if needed
            features_with_delta = features_df.copy()
            if 'lstm_delta' in expected_features:
                features_with_delta = handle_missing_lstm_delta(
                    features_with_delta, 
                    lstm_prediction, 
                    current_price
                )
            
            # Align features with training expectations
            aligned_df = align_features_with_training(
                features_with_delta,
                expected_features,
                f"XGBoost-{symbol}-W{window}"
            )
            
            self.logger.debug(f"Aligned XGBoost features for {symbol} window {window}: {len(expected_features)} features")
            return aligned_df, expected_features
            
        except Exception as e:
            self.logger.error(f"Error aligning XGBoost features for {symbol} window {window}: {e}")
            return features_df, []
    
    def prepare_lstm_input(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        window: int,
        sequence_length: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Prepare LSTM input with proper scaling and sequence formatting.
        
        Args:
            features_df: DataFrame with features
            symbol: Trading symbol
            window: Model window number
            sequence_length: Override default sequence length
            
        Returns:
            Prepared LSTM input array or None if failed
        """
        try:
            metadata = self.load_training_metadata(symbol, window)
            
            # Use provided sequence length or default from metadata
            seq_len = sequence_length or metadata.get('sequence_length', 96)
            lstm_features = metadata['lstm_features']
            
            if not lstm_features:
                self.logger.error(f"No LSTM features available for {symbol} window {window}")
                return None
            
            # Get scaler if available
            cache_key = f"{symbol.lower().replace('-', '')}_window_{window}"
            scaler = self._scaler_cache.get(cache_key)
            
            # Prepare sequence safely
            lstm_sequence = prepare_lstm_sequence_safe(
                features_df,
                lstm_features,
                seq_len,
                f"{symbol}-W{window}",
                scaler
            )
            
            if lstm_sequence is not None:
                self.logger.debug(f"Prepared LSTM input for {symbol} window {window}: {lstm_sequence.shape}")
            
            return lstm_sequence
            
        except Exception as e:
            self.logger.error(f"Error preparing LSTM input for {symbol} window {window}: {e}")
            return None
    
    def validate_feature_compatibility(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        window: int,
        model_type: str = "both"
    ) -> Dict[str, Any]:
        """
        Validate feature compatibility for models.
        
        Args:
            features_df: DataFrame with available features
            symbol: Trading symbol
            window: Model window number
            model_type: "lstm", "xgboost", or "both"
            
        Returns:
            Validation results dictionary
        """
        try:
            metadata = self.load_training_metadata(symbol, window)
            available_features = list(features_df.columns)
            
            validation = {
                'symbol': symbol,
                'window': window,
                'lstm_compatible': True,
                'xgboost_compatible': True,
                'lstm_diagnosis': {},
                'xgboost_diagnosis': {},
                'recommendations': [],
                'overall_compatible': True
            }
            
            # Validate LSTM compatibility
            if model_type in ["lstm", "both"] and metadata['lstm_features']:
                cache_key = f"{symbol.lower().replace('-', '')}_window_{window}"
                scaler = self._scaler_cache.get(cache_key)
                
                validation['lstm_diagnosis'] = diagnose_compatibility_issues(
                    f"{symbol}-LSTM-W{window}",
                    metadata['lstm_features'],
                    available_features,
                    scaler
                )
                
                validation['lstm_compatible'] = (
                    len(validation['lstm_diagnosis']['missing_features']) == 0 and
                    validation['lstm_diagnosis']['scaler_compatible']
                )
            
            # Validate XGBoost compatibility
            if model_type in ["xgboost", "both"] and metadata['xgb_features']:
                validation['xgboost_diagnosis'] = diagnose_compatibility_issues(
                    f"{symbol}-XGB-W{window}",
                    metadata['xgb_features'],
                    available_features
                )
                
                validation['xgboost_compatible'] = (
                    len(validation['xgboost_diagnosis']['missing_features']) <= 5  # Allow some missing features
                )
            
            # Overall compatibility
            validation['overall_compatible'] = (
                validation['lstm_compatible'] and validation['xgboost_compatible']
            )
            
            # Generate recommendations
            if not validation['lstm_compatible']:
                validation['recommendations'].extend(
                    validation['lstm_diagnosis'].get('recommendations', [])
                )
            
            if not validation['xgboost_compatible']:
                validation['recommendations'].extend(
                    validation['xgboost_diagnosis'].get('recommendations', [])
                )
            
            self.logger.debug(f"Feature compatibility validation for {symbol} window {window}: "
                            f"LSTM={validation['lstm_compatible']}, XGB={validation['xgboost_compatible']}")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating feature compatibility for {symbol} window {window}: {e}")
            return {
                'symbol': symbol,
                'window': window,
                'lstm_compatible': False,
                'xgboost_compatible': False,
                'overall_compatible': False,
                'error': str(e)
            }
    
    def get_model_requirements(self, symbol: str, window: int) -> Dict[str, Any]:
        """
        Get model requirements for a specific symbol and window.
        
        Args:
            symbol: Trading symbol
            window: Model window number
            
        Returns:
            Dictionary with model requirements
        """
        try:
            metadata = self.load_training_metadata(symbol, window)
            
            requirements = {
                'symbol': symbol,
                'window': window,
                'lstm_features_count': len(metadata['lstm_features']),
                'xgboost_features_count': len(metadata['xgb_features']),
                'sequence_length': metadata['sequence_length'],
                'scaler_required': metadata['scaler_loaded'],
                'scaler_features_expected': metadata['scaler_features'],
                'lstm_features': metadata['lstm_features'][:10],  # Show first 10
                'xgboost_features': metadata['xgb_features'][:10],  # Show first 10
                'metadata_complete': metadata['feature_columns_loaded']
            }
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"Error getting model requirements for {symbol} window {window}: {e}")
            return {'error': str(e)}
    
    def diagnose_system_compatibility(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Diagnose compatibility issues across the entire system.
        
        Args:
            symbols: List of trading symbols to check
            
        Returns:
            System-wide compatibility diagnosis
        """
        try:
            diagnosis = {
                'timestamp': pd.Timestamp.now(),
                'symbols_checked': len(symbols),
                'compatible_symbols': [],
                'problematic_symbols': [],
                'common_issues': [],
                'recommendations': [],
                'summary': {}
            }
            
            issue_counts = {}
            
            for symbol in symbols:
                symbol_issues = []
                
                # Check for available windows
                symbol_lower = symbol.lower().replace("-", "")
                
                # Scan for available windows
                windows_found = []
                for model_dir in ['lstm', 'xgboost']:
                    model_path = self.models_dir / model_dir
                    if model_path.exists():
                        for model_file in model_path.glob(f"{symbol_lower}_window_*.{{'keras' if model_dir == 'lstm' else 'json'}}"):
                            try:
                                window_num = int(model_file.stem.split("_window_")[1].split(".")[0])
                                windows_found.append(window_num)
                            except (ValueError, IndexError):
                                continue
                
                windows_found = sorted(set(windows_found))
                
                if not windows_found:
                    symbol_issues.append("No model windows found")
                    issue_counts["no_models"] = issue_counts.get("no_models", 0) + 1
                else:
                    # Check latest window
                    latest_window = max(windows_found)
                    metadata = self.load_training_metadata(symbol, latest_window)
                    
                    if not metadata.get('feature_columns_loaded', False):
                        symbol_issues.append("Missing feature columns")
                        issue_counts["missing_features"] = issue_counts.get("missing_features", 0) + 1
                    
                    if not metadata.get('scaler_loaded', False):
                        symbol_issues.append("Missing scaler")
                        issue_counts["missing_scaler"] = issue_counts.get("missing_scaler", 0) + 1
                    
                    if len(metadata.get('lstm_features', [])) == 0:
                        symbol_issues.append("No LSTM features")
                        issue_counts["no_lstm_features"] = issue_counts.get("no_lstm_features", 0) + 1
                    
                    if len(metadata.get('xgb_features', [])) == 0:
                        symbol_issues.append("No XGBoost features")
                        issue_counts["no_xgb_features"] = issue_counts.get("no_xgb_features", 0) + 1
                
                if symbol_issues:
                    diagnosis['problematic_symbols'].append({
                        'symbol': symbol,
                        'windows_found': len(windows_found),
                        'issues': symbol_issues
                    })
                else:
                    diagnosis['compatible_symbols'].append(symbol)
            
            # Identify common issues
            for issue, count in issue_counts.items():
                if count >= len(symbols) * 0.5:  # Affects 50%+ of symbols
                    diagnosis['common_issues'].append(f"{issue}: {count}/{len(symbols)} symbols")
            
            # Generate recommendations
            if issue_counts.get("missing_features", 0) > 0:
                diagnosis['recommendations'].append(
                    "Re-run training with proper feature column saving"
                )
            
            if issue_counts.get("missing_scaler", 0) > 0:
                diagnosis['recommendations'].append(
                    "Ensure scalers are saved during training"
                )
            
            if issue_counts.get("no_models", 0) > 0:
                diagnosis['recommendations'].append(
                    "Train models for symbols with no saved models"
                )
            
            # Summary
            diagnosis['summary'] = {
                'compatible_count': len(diagnosis['compatible_symbols']),
                'problematic_count': len(diagnosis['problematic_symbols']),
                'compatibility_rate': len(diagnosis['compatible_symbols']) / len(symbols) if symbols else 0,
                'most_common_issue': max(issue_counts.keys(), key=issue_counts.get) if issue_counts else None
            }
            
            self.logger.info(f"System compatibility diagnosis: {diagnosis['summary']['compatibility_rate']:.1%} compatible")
            
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"Error diagnosing system compatibility: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear all cached data."""
        self._feature_cache.clear()
        self._scaler_cache.clear()
        self._compatibility_cache.clear()
        self.logger.debug("Cleared compatibility handler cache")