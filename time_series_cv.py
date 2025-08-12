#!/usr/bin/env python3
"""
Time Series Cross-Validation with Leakage Prevention
==================================================

This module provides time-series specific cross-validation methods with:
1. Purged K-Fold to prevent leakage
2. Embargo periods to handle serial correlation
3. Consistent validation across all models
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FoldConfig:
    """Configuration for time series folds."""
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    embargo_start: int
    embargo_end: int


class TimeSeriesCV:
    """
    Time Series Cross-Validation with purging and embargo.
    
    This implements Purged K-Fold Cross-Validation from "Advances in Financial Machine Learning"
    by Marcos López de Prado, which prevents data leakage in time series models.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 embargo_pct: float = 0.01,
                 purge_pct: float = 0.02,
                 min_train_size: float = 0.3):
        """
        Initialize TimeSeriesCV.
        
        Args:
            n_splits: Number of cross-validation folds
            embargo_pct: Embargo period as percentage of total data
            purge_pct: Purge period as percentage of total data  
            min_train_size: Minimum training size as percentage of total data
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        self.min_train_size = min_train_size
        
        logger.info(f"TimeSeriesCV initialized: {n_splits} splits, embargo={embargo_pct:.1%}, purge={purge_pct:.1%}")
    
    def split(self, 
              X: pd.DataFrame, 
              y: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation splits with purging and embargo.
        
        Args:
            X: Feature matrix with datetime index
            y: Target vector (optional)
            
        Yields:
            Tuples of (train_indices, val_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate sizes
        min_train_samples = int(self.min_train_size * n_samples)
        embargo_samples = int(self.embargo_pct * n_samples)
        purge_samples = int(self.purge_pct * n_samples)
        
        # Calculate fold boundaries
        folds = self._calculate_fold_boundaries(n_samples, min_train_samples, embargo_samples, purge_samples)
        
        logger.info(f"Generated {len(folds)} folds for {n_samples} samples")
        
        for i, fold in enumerate(folds):
            train_idx = indices[fold.train_start:fold.train_end]
            val_idx = indices[fold.val_start:fold.val_end]
            
            # Log fold information
            train_pct = len(train_idx) / n_samples * 100
            val_pct = len(val_idx) / n_samples * 100
            logger.debug(f"Fold {i+1}: train={len(train_idx)} ({train_pct:.1f}%), val={len(val_idx)} ({val_pct:.1f}%)")
            
            yield train_idx, val_idx
    
    def _calculate_fold_boundaries(self, 
                                  n_samples: int, 
                                  min_train_samples: int,
                                  embargo_samples: int, 
                                  purge_samples: int) -> List[FoldConfig]:
        """Calculate fold boundaries with proper purging and embargo."""
        folds = []
        
        # Calculate validation fold size
        total_available = n_samples - embargo_samples
        val_fold_size = total_available // self.n_splits
        
        for i in range(self.n_splits):
            # Validation period
            val_start = i * val_fold_size
            val_end = min(val_start + val_fold_size, total_available)
            
            # Skip if validation period is too small
            if val_end - val_start < 10:
                continue
            
            # Training period (before validation)
            train_end = max(0, val_start - purge_samples)
            train_start = max(0, train_end - min_train_samples)
            
            # Skip if training period is too small
            if train_end - train_start < min_train_samples:
                continue
            
            # Embargo period (after validation)
            embargo_start = val_end
            embargo_end = min(embargo_start + embargo_samples, n_samples)
            
            fold = FoldConfig(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end
            )
            
            folds.append(fold)
        
        return folds
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits."""
        return self.n_splits


def get_time_series_folds(timestamps: pd.DatetimeIndex, 
                         n_splits: int = 5,
                         embargo_pct: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to get time series folds.
    
    Args:
        timestamps: DatetimeIndex of the data
        n_splits: Number of cross-validation folds
        embargo_pct: Embargo period as percentage of data
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Create dummy DataFrame with timestamps as index
    dummy_df = pd.DataFrame(index=timestamps)
    
    # Use TimeSeriesCV to generate folds
    tscv = TimeSeriesCV(n_splits=n_splits, embargo_pct=embargo_pct)
    
    # Convert generator to list
    folds = list(tscv.split(dummy_df))
    
    logger.info(f"Generated {len(folds)} time series folds with embargo={embargo_pct:.1%}")
    return folds


class WalkForwardValidation:
    """
    Walk-forward validation for time series models.
    
    This is a specialized form of time series validation where the model
    is retrained on each expanding window of data.
    """
    
    def __init__(self, 
                 initial_window: int,
                 step_size: int = 1,
                 expanding_window: bool = True):
        """
        Initialize WalkForwardValidation.
        
        Args:
            initial_window: Initial training window size
            step_size: Number of samples to step forward each iteration
            expanding_window: Whether to use expanding window (True) or rolling window (False)
        """
        self.initial_window = initial_window
        self.step_size = step_size
        self.expanding_window = expanding_window
    
    def split(self, X: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Feature matrix with datetime index
            
        Yields:
            Tuples of (train_indices, val_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Start with initial window
        current_pos = self.initial_window
        
        while current_pos < n_samples:
            # Training indices
            if self.expanding_window:
                train_start = 0
            else:
                train_start = max(0, current_pos - self.initial_window)
            
            train_end = current_pos
            train_idx = indices[train_start:train_end]
            
            # Validation indices (next step_size samples)
            val_start = current_pos
            val_end = min(current_pos + self.step_size, n_samples)
            val_idx = indices[val_start:val_end]
            
            if len(val_idx) == 0:
                break
            
            yield train_idx, val_idx
            
            # Move forward
            current_pos += self.step_size


def validate_time_series_split(train_idx: np.ndarray, 
                              val_idx: np.ndarray,
                              timestamps: pd.DatetimeIndex) -> dict:
    """
    Validate time series split to ensure no leakage.
    
    Args:
        train_idx: Training indices
        val_idx: Validation indices  
        timestamps: Timestamps for the data
        
    Returns:
        Validation report
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check that validation comes after training
    if len(train_idx) > 0 and len(val_idx) > 0:
        max_train_time = timestamps[train_idx].max()
        min_val_time = timestamps[val_idx].min()
        
        if max_train_time >= min_val_time:
            report["errors"].append("Time leakage detected: validation data overlaps with training data")
            report["valid"] = False
    
    # Check for gaps (which are okay)
    if len(train_idx) > 0 and len(val_idx) > 0:
        max_train_idx = train_idx.max()
        min_val_idx = val_idx.min()
        gap_size = min_val_idx - max_train_idx - 1
        
        if gap_size > 0:
            report["warnings"].append(f"Gap of {gap_size} samples between training and validation")
    
    # Add statistics
    report["stats"] = {
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "train_period": (timestamps[train_idx].min(), timestamps[train_idx].max()) if len(train_idx) > 0 else None,
        "val_period": (timestamps[val_idx].min(), timestamps[val_idx].max()) if len(val_idx) > 0 else None
    }
    
    return report