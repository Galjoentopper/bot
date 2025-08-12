#!/usr/bin/env python3
"""
Probability Calibration Utilities
=================================

This module provides utilities for calibrating model probabilities:
1. Isotonic regression calibration
2. Platt scaling calibration  
3. Calibration persistence and loading
4. Calibration quality assessment
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Utility class for calibrating model probabilities.
    
    Supports both isotonic regression and Platt scaling methods
    for improving probability estimates from classifiers.
    """
    
    def __init__(self, method: str = "isotonic", n_bins: int = 10):
        """
        Initialize probability calibrator.
        
        Args:
            method: Calibration method ("isotonic" or "platt")
            n_bins: Number of bins for reliability diagram
        """
        if method not in ["isotonic", "platt"]:
            raise ValueError("Method must be 'isotonic' or 'platt'")
        
        self.method = method
        self.n_bins = n_bins
        self.calibrator = None
        self.is_fitted = False
        self.calibration_curve = None
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibration model.
        
        Args:
            y_true: True binary labels (0/1)
            y_prob: Uncalibrated probabilities
            
        Returns:
            Self for method chaining
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == "platt":
            self.calibrator = LogisticRegression()
            # Reshape for sklearn
            y_prob = y_prob.reshape(-1, 1)
        
        # Fit calibrator
        self.calibrator.fit(y_prob, y_true)
        self.is_fitted = True
        
        # Generate calibration curve for diagnostics
        self.calibration_curve = self._compute_calibration_curve(y_true, y_prob)
        
        logger.info(f"Calibration fitted using {self.method} method")
        return self
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            y_prob: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == "platt":
            y_prob = y_prob.reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(y_prob)[:, 1]
        else:  # isotonic
            calibrated = self.calibrator.predict(y_prob)
        
        # Ensure probabilities are in valid range
        calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
        
        return calibrated
    
    def evaluate_calibration(self, 
                           y_true: np.ndarray, 
                           y_prob_raw: np.ndarray,
                           y_prob_cal: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate calibration quality.
        
        Args:
            y_true: True binary labels
            y_prob_raw: Raw (uncalibrated) probabilities  
            y_prob_cal: Calibrated probabilities (computed if None)
            
        Returns:
            Dictionary of calibration metrics
        """
        if y_prob_cal is None:
            y_prob_cal = self.calibrate(y_prob_raw)
        
        # Calibration metrics
        brier_raw = brier_score_loss(y_true, y_prob_raw)
        brier_cal = brier_score_loss(y_true, y_prob_cal)
        
        logloss_raw = log_loss(y_true, y_prob_raw)
        logloss_cal = log_loss(y_true, y_prob_cal)
        
        # Reliability (calibration error)
        reliability_raw = self._compute_expected_calibration_error(y_true, y_prob_raw)
        reliability_cal = self._compute_expected_calibration_error(y_true, y_prob_cal)
        
        # Resolution (ability to discriminate)
        resolution_raw = self._compute_resolution(y_true, y_prob_raw)
        resolution_cal = self._compute_resolution(y_true, y_prob_cal)
        
        return {
            'brier_score_raw': float(brier_raw),
            'brier_score_calibrated': float(brier_cal),
            'brier_improvement': float(brier_raw - brier_cal),
            'log_loss_raw': float(logloss_raw),
            'log_loss_calibrated': float(logloss_cal),
            'log_loss_improvement': float(logloss_raw - logloss_cal),
            'ece_raw': float(reliability_raw),
            'ece_calibrated': float(reliability_cal),
            'ece_improvement': float(reliability_raw - reliability_cal),
            'resolution_raw': float(resolution_raw),
            'resolution_calibrated': float(resolution_cal)
        }
    
    def _compute_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute calibration curve (reliability diagram data)."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        observed_frequencies = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_center = (bin_lower + bin_upper) / 2
                observed_freq = y_true[in_bin].mean()
                
                bin_centers.append(bin_center)
                observed_frequencies.append(observed_freq)
                bin_sizes.append(in_bin.sum())
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                observed_frequencies.append(0.0)
                bin_sizes.append(0)
        
        return {
            'bin_centers': np.array(bin_centers),
            'observed_frequencies': np.array(observed_frequencies),
            'bin_sizes': np.array(bin_sizes),
            'bin_boundaries': bin_boundaries
        }
    
    def _compute_expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute Expected Calibration Error (ECE)."""
        curve = self._compute_calibration_curve(y_true, y_prob)
        
        # Weight by bin size
        bin_weights = curve['bin_sizes'] / len(y_true)
        
        # Calibration error per bin
        calibration_errors = np.abs(curve['bin_centers'] - curve['observed_frequencies'])
        
        # Weighted average
        ece = np.average(calibration_errors, weights=bin_weights)
        return float(ece)
    
    def _compute_resolution(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute resolution (discrimination ability)."""
        base_rate = y_true.mean()
        
        curve = self._compute_calibration_curve(y_true, y_prob)
        bin_weights = curve['bin_sizes'] / len(y_true)
        
        # Resolution is weighted variance of conditional expectations
        resolution = np.average(
            (curve['observed_frequencies'] - base_rate) ** 2,
            weights=bin_weights
        )
        return float(resolution)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save calibrator to disk."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Cannot save.")
        
        calibrator_data = {
            'method': self.method,
            'n_bins': self.n_bins,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted,
            'calibration_curve': self.calibration_curve
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(calibrator_data, f)
        
        logger.info(f"Saved calibrator to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ProbabilityCalibrator':
        """Load calibrator from disk."""
        with open(filepath, 'rb') as f:
            calibrator_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            method=calibrator_data['method'],
            n_bins=calibrator_data['n_bins']
        )
        
        # Restore state
        instance.calibrator = calibrator_data['calibrator']
        instance.is_fitted = calibrator_data['is_fitted']
        instance.calibration_curve = calibrator_data['calibration_curve']
        
        logger.info(f"Loaded calibrator from {filepath}")
        return instance


def calibrate_model_probabilities(model,
                                 X_cal: np.ndarray,
                                 y_cal: np.ndarray,
                                 method: str = "isotonic") -> Tuple[Any, ProbabilityCalibrator]:
    """
    Calibrate a model's probabilities using validation data.
    
    Args:
        model: Fitted classifier with predict_proba method
        X_cal: Calibration features
        y_cal: Calibration labels
        method: Calibration method ("isotonic" or "platt")
        
    Returns:
        Tuple of (calibrated_model, calibrator)
    """
    # Get uncalibrated probabilities
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_cal)[:, 1]
    else:
        # Assume predict returns probabilities
        y_prob = model.predict(X_cal)
    
    # Fit calibrator
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(y_cal, y_prob)
    
    # Create calibrated model wrapper
    calibrated_model = CalibratedModelWrapper(model, calibrator)
    
    logger.info(f"Model probabilities calibrated using {method}")
    return calibrated_model, calibrator


class CalibratedModelWrapper:
    """
    Wrapper for models with calibrated probabilities.
    
    This wrapper applies probability calibration transparently
    while maintaining the original model interface.
    """
    
    def __init__(self, model, calibrator: ProbabilityCalibrator):
        """
        Initialize calibrated model wrapper.
        
        Args:
            model: Original fitted model
            calibrator: Fitted probability calibrator
        """
        self.model = model
        self.calibrator = calibrator
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using original model."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate calibrated probabilities."""
        if hasattr(self.model, 'predict_proba'):
            raw_proba = self.model.predict_proba(X)
            if raw_proba.ndim == 2:
                # Binary classification - calibrate positive class
                calibrated_pos = self.calibrator.calibrate(raw_proba[:, 1])
                calibrated_proba = np.column_stack([1 - calibrated_pos, calibrated_pos])
            else:
                # Single probability output
                calibrated_proba = self.calibrator.calibrate(raw_proba)
        else:
            # Assume predict returns probabilities
            raw_proba = self.model.predict(X)
            calibrated_proba = self.calibrator.calibrate(raw_proba)
        
        return calibrated_proba
    
    def __getattr__(self, name):
        """Delegate other methods to original model."""
        return getattr(self.model, name)


def assess_calibration_quality(y_true: np.ndarray, 
                              y_prob: np.ndarray,
                              n_bins: int = 10) -> Dict[str, Any]:
    """
    Assess calibration quality of probability predictions.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary containing calibration assessment
    """
    # Create temporary calibrator for curve computation
    calibrator = ProbabilityCalibrator(n_bins=n_bins)
    curve = calibrator._compute_calibration_curve(y_true, y_prob)
    
    # Compute metrics
    ece = calibrator._compute_expected_calibration_error(y_true, y_prob)
    resolution = calibrator._compute_resolution(y_true, y_prob)
    
    # Additional statistics
    base_rate = y_true.mean()
    brier_score = brier_score_loss(y_true, y_prob)
    
    # Decompose Brier score
    reliability = ece  # Approximation
    uncertainty = base_rate * (1 - base_rate)
    
    return {
        'expected_calibration_error': float(ece),
        'resolution': float(resolution),
        'brier_score': float(brier_score),
        'reliability': float(reliability),
        'uncertainty': float(uncertainty),
        'base_rate': float(base_rate),
        'calibration_curve': {
            'bin_centers': curve['bin_centers'].tolist(),
            'observed_frequencies': curve['observed_frequencies'].tolist(),
            'bin_sizes': curve['bin_sizes'].tolist()
        }
    }


def create_reliability_diagram_data(y_true: np.ndarray, 
                                   y_prob: np.ndarray,
                                   n_bins: int = 10) -> pd.DataFrame:
    """
    Create data for plotting reliability diagrams.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        DataFrame with reliability diagram data
    """
    calibrator = ProbabilityCalibrator(n_bins=n_bins)
    curve = calibrator._compute_calibration_curve(y_true, y_prob)
    
    return pd.DataFrame({
        'predicted_probability': curve['bin_centers'],
        'observed_frequency': curve['observed_frequencies'],
        'bin_size': curve['bin_sizes'],
        'bin_weight': curve['bin_sizes'] / len(y_true)
    })