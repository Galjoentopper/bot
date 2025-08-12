#!/usr/bin/env python3
"""
DatasetBuilder - Centralized Dataset Assembly for ML Training
============================================================

This module provides a centralized approach to dataset creation with:
1. Feature cache per symbol and time range for massive speed-up
2. Feature metadata validation to prevent training issues
3. Consistent feature engineering across all models
4. Hash-based cache invalidation when features change
"""

import os
import hashlib
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for cached features"""
    feature_names: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    min_values: Dict[str, float]
    max_values: Dict[str, float]
    signature: str
    label_spec: Dict[str, Any]
    created_at: str
    n_samples: int
    symbol: str
    interval: str


class DatasetBuilder:
    """
    Centralized dataset assembly with feature caching and validation.
    
    Features:
    - Loads raw data once per symbol
    - Runs feature engineering once with caching
    - Validates feature metadata before training
    - Computes feature signature for cache invalidation
    - Handles time-series specific concerns (leakage prevention)
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 cache_dir: str = None,
                 feature_config: Dict[str, Any] = None):
        """
        Initialize DatasetBuilder.
        
        Args:
            data_dir: Directory containing raw market data
            cache_dir: Directory for feature cache (data/features)
            feature_config: Configuration for feature engineering
        """
        # Set up directories
        self.repo_root = self._find_repo_root()
        self.data_dir = Path(data_dir) if data_dir else Path(self.repo_root) / "data"
        self.cache_dir = Path(cache_dir) if cache_dir else Path(self.repo_root) / "data" / "features"
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default feature configuration
        self.feature_config = feature_config or self._get_default_feature_config()
        
        # Feature signature (hash of config for cache invalidation)
        self.feature_signature = self._compute_feature_signature()
        
        logger.info(f"DatasetBuilder initialized: cache_dir={self.cache_dir}, signature={self.feature_signature[:8]}")
    
    def _find_repo_root(self) -> Path:
        """Find repository root by looking for key files."""
        current_file = Path(__file__).absolute()
        current_dir = current_file.parent
        
        while current_dir != current_dir.parent:
            if (current_dir / "data").exists() or (current_dir / "requirements.txt").exists():
                return current_dir
            current_dir = current_dir.parent
        
        return Path.cwd()
    
    def _get_default_feature_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            "technical_indicators": {
                "rsi_periods": [9, 14, 21],
                "ema_periods": [9, 21, 50, 100],
                "bb_period": 20,
                "bb_std": 2.0,
                "atr_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            },
            "price_features": {
                "return_periods": [1, 2, 3, 5, 10, 20],
                "volatility_periods": [5, 10, 20, 50],
                "price_change_periods": [2, 4, 16, 96]  # 30min, 1h, 4h, 24h
            },
            "volume_features": {
                "volume_ma_periods": [10, 20, 50],
                "volume_spike_threshold": 2.0
            },
            "market_microstructure": {
                "enable_order_flow": True,
                "enable_pressure_features": True,
                "enable_candle_patterns": True
            },
            "target_config": {
                "prediction_horizon": 4,  # 1 hour = 4x15min
                "price_change_threshold": 0.005,  # 0.5%
                "target_type": "binary_jump"  # binary classification for jumps
            }
        }
    
    def _compute_feature_signature(self) -> str:
        """Compute hash of feature configuration for cache invalidation."""
        config_str = json.dumps(self.feature_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def get_dataset(self, 
                   symbol: str, 
                   interval: str = "15m",
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   force_rebuild: bool = False) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """
        Get dataset with features for a symbol, using cache when possible.
        
        Args:
            symbol: Trading symbol (e.g., "BTCEUR")
            interval: Time interval (e.g., "15m")
            start_date: Start date (optional, for filtering)
            end_date: End date (optional, for filtering)
            force_rebuild: Force rebuild cache even if valid
            
        Returns:
            Tuple of (features_df, metadata)
        """
        # Create cache key
        cache_key = f"{symbol}_{interval}_{self.feature_signature[:12]}"
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
        
        # Check if cached version exists and is valid
        if not force_rebuild and cache_path.exists() and metadata_path.exists():
            try:
                # Load cached features
                logger.info(f"Loading cached features for {symbol} from {cache_path}")
                features_df = pd.read_parquet(cache_path)
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = DatasetMetadata(**metadata_dict)
                
                # Validate cache integrity
                if self._validate_cached_features(features_df, metadata):
                    # Apply date filtering if requested
                    if start_date or end_date:
                        features_df = self._filter_by_date(features_df, start_date, end_date)
                    return features_df, metadata
                else:
                    logger.warning(f"Cache validation failed for {symbol}, rebuilding...")
            except Exception as e:
                logger.warning(f"Failed to load cached features for {symbol}: {e}, rebuilding...")
        
        # Build features from scratch
        logger.info(f"Building features for {symbol} (interval: {interval})")
        features_df = self._build_features(symbol, interval)
        
        # Create metadata
        metadata = self._create_metadata(features_df, symbol, interval)
        
        # Save to cache
        self._save_to_cache(features_df, metadata, cache_path, metadata_path)
        
        # Apply date filtering if requested
        if start_date or end_date:
            features_df = self._filter_by_date(features_df, start_date, end_date)
        
        return features_df, metadata
    
    def _load_raw_data(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        """Load raw OHLCV data from SQLite database."""
        db_path = self.data_dir / f"{symbol.lower()}_{interval}.db"
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        logger.info(f"Loaded {len(df):,} candles for {symbol} from {db_path}")
        return df
    
    def _build_features(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        """Build comprehensive technical features."""
        # Load raw data
        df = self._load_raw_data(symbol, interval)
        
        # Start with raw data
        features = df.copy()
        
        # Add technical indicators
        features = self._add_technical_indicators(features)
        
        # Add price features
        features = self._add_price_features(features)
        
        # Add volume features  
        features = self._add_volume_features(features)
        
        # Add market microstructure features
        features = self._add_microstructure_features(features)
        
        # Add time-based features
        features = self._add_time_features(features)
        
        # Add target variable
        features = self._add_target_variable(features)
        
        # Clean features (handle NaN/inf)
        features = self._clean_features(features)
        
        logger.info(f"Built {len(features.columns)} features for {symbol}")
        return features
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        try:
            import pandas_ta as ta
        except ImportError:
            logger.warning("pandas_ta not available, using fallback indicators")
            ta = self._get_fallback_ta()
        
        data = df.copy()
        config = self.feature_config["technical_indicators"]
        
        # RSI with multiple periods
        for period in config["rsi_periods"]:
            data[f"rsi_{period}"] = ta.rsi(data["close"], length=period)
        
        # EMA with multiple periods
        for period in config["ema_periods"]:
            data[f"ema_{period}"] = ta.ema(data["close"], length=period)
            data[f"price_vs_ema_{period}"] = (data["close"] - data[f"ema_{period}"]) / data[f"ema_{period}"]
        
        # Bollinger Bands
        bb = ta.bbands(data["close"], length=config["bb_period"], std=config["bb_std"])
        if bb is not None and not bb.empty:
            data["bb_upper"] = bb.iloc[:, 0]  # Upper band
            data["bb_middle"] = bb.iloc[:, 1]  # Middle band  
            data["bb_lower"] = bb.iloc[:, 2]  # Lower band
            data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
            data["bb_position"] = (data["close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])
        
        # ATR
        data["atr"] = ta.atr(data["high"], data["low"], data["close"], length=config["atr_period"])
        data["atr_ratio"] = data["atr"] / data["close"]
        
        # MACD
        macd = ta.macd(data["close"], fast=config["macd_fast"], slow=config["macd_slow"], signal=config["macd_signal"])
        if macd is not None and not macd.empty:
            data["macd"] = macd.iloc[:, 0]
            data["macd_histogram"] = macd.iloc[:, 1] 
            data["macd_signal"] = macd.iloc[:, 2]
        
        return data
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        data = df.copy()
        config = self.feature_config["price_features"]
        
        # Returns with multiple periods
        for period in config["return_periods"]:
            data[f"returns_{period}"] = data["close"].pct_change(period)
            data[f"log_returns_{period}"] = np.log(data["close"] / data["close"].shift(period))
        
        # Volatility with multiple periods
        for period in config["volatility_periods"]:
            data[f"volatility_{period}"] = data["returns_1"].rolling(period).std()
        
        # Price changes for different timeframes
        for period in config["price_change_periods"]:
            data[f"price_change_{period}p"] = data["close"].pct_change(period)
        
        # Price normalization features
        data["price_zscore_20"] = (data["close"] - data["close"].rolling(20).mean()) / data["close"].rolling(20).std()
        data["price_zscore_50"] = (data["close"] - data["close"].rolling(50).mean()) / data["close"].rolling(50).std()
        
        return data
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        data = df.copy()
        config = self.feature_config["volume_features"]
        
        # Volume moving averages
        for period in config["volume_ma_periods"]:
            data[f"volume_ma_{period}"] = data["volume"].rolling(period).mean()
            data[f"volume_ratio_{period}"] = data["volume"] / data[f"volume_ma_{period}"]
        
        # Volume spike detection
        data["volume_spike"] = (data["volume_ratio_20"] > config["volume_spike_threshold"]).astype(int)
        
        # Volume-price features
        data["volume_price_trend"] = data["volume"] * data["returns_1"]
        data["vwap"] = (data["close"] * data["volume"]).cumsum() / data["volume"].cumsum()
        data["price_vs_vwap"] = (data["close"] - data["vwap"]) / data["vwap"]
        
        return data
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        data = df.copy()
        config = self.feature_config["market_microstructure"]
        
        if config["enable_order_flow"]:
            # Order flow approximation
            data["buying_pressure"] = (data["close"] - data["low"]) / (data["high"] - data["low"])
            data["selling_pressure"] = (data["high"] - data["close"]) / (data["high"] - data["low"])
            data["net_pressure"] = data["buying_pressure"] - data["selling_pressure"]
        
        if config["enable_pressure_features"]:
            # Pressure indicators
            data["spread"] = (data["high"] - data["low"]) / data["close"]
            data["spread_ma"] = data["spread"].rolling(20).mean()
            data["spread_ratio"] = data["spread"] / data["spread_ma"]
        
        if config["enable_candle_patterns"]:
            # Candle pattern features
            data["candle_body"] = abs(data["close"] - data["open"]) / data["open"]
            data["upper_wick"] = (data["high"] - np.maximum(data["open"], data["close"])) / data["open"]
            data["lower_wick"] = (np.minimum(data["open"], data["close"]) - data["low"]) / data["open"]
        
        return data
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        data = df.copy()
        
        # Time components
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
        data["month"] = data.index.month
        data["quarter"] = data.index.quarter
        
        return data
    
    def _add_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable based on configuration."""
        data = df.copy()
        config = self.feature_config["target_config"]
        
        if config["target_type"] == "binary_jump":
            # Binary classification: predict if price will increase by threshold within horizon
            horizon = config["prediction_horizon"]
            threshold = config["price_change_threshold"]
            
            # Calculate maximum price reached within horizon
            future_prices = data["close"].shift(-horizon).rolling(window=horizon, min_periods=1).max()
            price_change = (future_prices - data["close"]) / data["close"]
            
            data["target"] = (price_change >= threshold).astype(int)
        
        elif config["target_type"] == "regression":
            # Regression: predict actual price change
            horizon = config["prediction_horizon"]
            data["target"] = data["close"].pct_change(horizon).shift(-horizon)
        
        return data
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling NaN and infinite values."""
        data = df.copy()
        
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill first to preserve time series structure
        data = data.fillna(method='ffill')
        
        # Backward fill for any remaining NaN at the beginning
        data = data.fillna(method='bfill')
        
        # For any still remaining NaN, fill with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isna().any():
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
        
        # Drop any rows with NaN in target
        if "target" in data.columns:
            data = data.dropna(subset=["target"])
        
        logger.info(f"Cleaned features: {len(data)} samples remaining")
        return data
    
    def _create_metadata(self, df: pd.DataFrame, symbol: str, interval: str) -> DatasetMetadata:
        """Create metadata for the dataset."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        metadata = DatasetMetadata(
            feature_names=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            null_counts={col: int(df[col].isnull().sum()) for col in df.columns},
            min_values={col: float(numeric_df[col].min()) for col in numeric_df.columns if not numeric_df[col].isna().all()},
            max_values={col: float(numeric_df[col].max()) for col in numeric_df.columns if not numeric_df[col].isna().all()},
            signature=self.feature_signature,
            label_spec=self.feature_config["target_config"],
            created_at=datetime.now().isoformat(),
            n_samples=len(df),
            symbol=symbol,
            interval=interval
        )
        
        return metadata
    
    def _save_to_cache(self, df: pd.DataFrame, metadata: DatasetMetadata, 
                      cache_path: Path, metadata_path: Path):
        """Save features and metadata to cache."""
        try:
            # Save features to parquet
            df.to_parquet(cache_path, index=True, engine='auto')
            
            # Save metadata to JSON
            with open(metadata_path, 'w') as f:
                json.dump(metadata.__dict__, f, indent=2, default=str)
            
            logger.info(f"Cached features saved: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _validate_cached_features(self, df: pd.DataFrame, metadata: DatasetMetadata) -> bool:
        """Validate cached features against metadata."""
        try:
            # Check signature matches
            if metadata.signature != self.feature_signature:
                logger.warning(f"Feature signature mismatch: {metadata.signature[:8]} vs {self.feature_signature[:8]}")
                return False
            
            # Check feature names match
            if set(df.columns) != set(metadata.feature_names):
                logger.warning("Feature names mismatch in cache")
                return False
            
            # Check for unexpected NaN/inf values
            if df.isnull().sum().sum() > sum(metadata.null_counts.values()) * 1.1:  # Allow 10% tolerance
                logger.warning("Excessive NaN values in cached features")
                return False
            
            if np.isinf(df.select_dtypes(include=[np.number]).values).any():
                logger.warning("Infinite values detected in cached features")
                return False
            
            logger.info(f"Cache validation passed for {metadata.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False
    
    def _filter_by_date(self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Filter dataframe by date range."""
        filtered_df = df.copy()
        
        if start_date:
            start_ts = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df.index >= start_ts]
        
        if end_date:
            end_ts = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df.index <= end_ts]
        
        logger.info(f"Date filtering: {len(filtered_df)} samples remaining")
        return filtered_df
    
    def _get_fallback_ta(self):
        """Fallback technical indicators implementation."""
        class FallbackTA:
            @staticmethod
            def rsi(prices, length=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            @staticmethod
            def ema(prices, length=14):
                return prices.ewm(span=length).mean()
            
            @staticmethod
            def bbands(prices, length=20, std=2):
                sma = prices.rolling(window=length).mean()
                std_dev = prices.rolling(window=length).std()
                upper = sma + (std_dev * std)
                lower = sma - (std_dev * std)
                return pd.DataFrame({'upper': upper, 'middle': sma, 'lower': lower})
            
            @staticmethod
            def atr(high, low, close, length=14):
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                return true_range.rolling(window=length).mean()
            
            @staticmethod
            def macd(prices, fast=12, slow=26, signal=9):
                ema_fast = prices.ewm(span=fast).mean()
                ema_slow = prices.ewm(span=slow).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal).mean()
                histogram = macd_line - signal_line
                return pd.DataFrame({'macd': macd_line, 'histogram': histogram, 'signal': signal_line})
        
        return FallbackTA()
    
    def validate_dataset(self, df: pd.DataFrame, metadata: DatasetMetadata) -> Dict[str, Any]:
        """
        Comprehensive dataset validation before training.
        
        Returns validation report with warnings/errors.
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {}
        }
        
        try:
            # Check for NaN/inf values
            nan_counts = df.isnull().sum()
            inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
            
            if nan_counts.sum() > 0:
                report["warnings"].append(f"Found {nan_counts.sum()} NaN values")
                report["stats"]["nan_by_column"] = nan_counts[nan_counts > 0].to_dict()
            
            if inf_counts.sum() > 0:
                report["errors"].append(f"Found {inf_counts.sum()} infinite values")
                report["stats"]["inf_by_column"] = inf_counts[inf_counts > 0].to_dict()
                report["valid"] = False
            
            # Check target distribution
            if "target" in df.columns:
                target_dist = df["target"].value_counts()
                target_balance = target_dist.min() / target_dist.max()
                
                if target_balance < 0.1:  # Very imbalanced
                    report["warnings"].append(f"Highly imbalanced target: {target_dist.to_dict()}")
                
                report["stats"]["target_distribution"] = target_dist.to_dict()
                report["stats"]["target_balance_ratio"] = float(target_balance)
            
            # Check feature consistency
            if set(df.columns) != set(metadata.feature_names):
                missing = set(metadata.feature_names) - set(df.columns)
                extra = set(df.columns) - set(metadata.feature_names)
                if missing:
                    report["errors"].append(f"Missing expected features: {missing}")
                    report["valid"] = False
                if extra:
                    report["warnings"].append(f"Extra unexpected features: {extra}")
            
            report["stats"]["n_samples"] = len(df)
            report["stats"]["n_features"] = len(df.columns)
            
        except Exception as e:
            report["errors"].append(f"Validation failed: {str(e)}")
            report["valid"] = False
        
        return report