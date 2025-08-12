#!/usr/bin/env python3
"""
LightGBM Trainer using Centralized DatasetBuilder
=================================================

This module provides a LightGBM trainer that uses the centralized DatasetBuilder
for consistent data processing and features advanced capabilities like:
- Automated hyperparameter tuning
- Feature importance analysis
- Cost-aware evaluation
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from dataset_builder import DatasetBuilder
from model_adapter import ModelAdapter
from time_series_cv import get_time_series_folds
from cost_aware_evaluation import CostAwareEvaluator
from calibration_utils import calibrate_model_probabilities

logger = logging.getLogger(__name__)


class LightGBMModelAdapter(ModelAdapter):
    """LightGBM model adapter with advanced features."""
    
    def __init__(self, name: str = "LightGBM", config: Optional[Dict[str, Any]] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LightGBM models")
        
        super().__init__(name, config)
        self.feature_importance_ = None
        self.feature_names_ = None
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            feature_names: Optional[List[str]] = None,
            **kwargs) -> 'LightGBMModelAdapter':
        """Train LightGBM model."""
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Store feature names
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Default parameters
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        # Merge with user config
        params = {**default_params, **self.config}
        
        # Extract training-specific parameters
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        n_estimators = params.pop('n_estimators', 100)
        
        # Create model
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **params
        )
        
        # Train with early stopping
        eval_set = [(X_val, y_val)]
        eval_names = ['validation']
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)  # Suppress output
            ]
        )
        
        # Store feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        self.is_fitted = True
        self.metadata['best_iteration'] = getattr(self.model, 'best_iteration', n_estimators)
        self.metadata['feature_importance'] = dict(zip(self.feature_names_, self.feature_importance_))
        
        logger.info(f"LightGBM training completed: {self.metadata['best_iteration']} iterations")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate LightGBM predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate LightGBM prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if importance_type == 'gain':
            importance = self.model.feature_importances_
        elif importance_type == 'split':
            importance = self.model.booster_.feature_importance(importance_type='split')
        else:
            importance = self.model.booster_.feature_importance(importance_type=importance_type)
        
        return dict(zip(self.feature_names_, importance))
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get LightGBM model artifacts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return {
            'name': self.name,
            'config': self.config,
            'model': self.model,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
    
    def _restore_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore LightGBM from artifacts."""
        self.config = artifacts['config']
        self.model = artifacts['model']
        self.feature_names_ = artifacts['feature_names']
        self.feature_importance_ = artifacts['feature_importance']
        self.metadata = artifacts['metadata']
        self.is_fitted = artifacts['is_fitted']


class LightGBMTrainer:
    """
    LightGBM trainer using centralized DatasetBuilder.
    
    Features:
    - Centralized dataset assembly
    - Time-series cross-validation
    - Feature importance analysis
    - Cost-aware evaluation
    - Hyperparameter optimization
    """
    
    def __init__(self, 
                 dataset_builder: DatasetBuilder,
                 config: Optional[Dict[str, Any]] = None,
                 cost_model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM trainer.
        
        Args:
            dataset_builder: Centralized dataset builder
            config: LightGBM model configuration
            cost_model_config: Cost model configuration
        """
        self.dataset_builder = dataset_builder
        self.config = config or self._get_default_config()
        self.cost_evaluator = CostAwareEvaluator()
        
        # Configure cost model if provided
        if cost_model_config:
            from cost_aware_evaluation import CostModel
            cost_model = CostModel(**cost_model_config)
            self.cost_evaluator = CostAwareEvaluator(cost_model)
    
    def train_symbol(self, 
                    symbol: str,
                    interval: str = "15m",
                    n_splits: int = 5,
                    calibrate: bool = True,
                    save_artifacts: bool = True,
                    optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train LightGBM model for a single symbol.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            n_splits: Number of CV splits
            calibrate: Whether to calibrate probabilities
            save_artifacts: Whether to save model artifacts
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training LightGBM model for {symbol}")
        
        # Get dataset
        features_df, metadata = self.dataset_builder.get_dataset(
            symbol=symbol, 
            interval=interval
        )
        
        # Validate dataset
        validation_report = self.dataset_builder.validate_dataset(features_df, metadata)
        if not validation_report['valid']:
            raise ValueError(f"Dataset validation failed: {validation_report['errors']}")
        
        # Prepare features and target
        X, y, feature_names = self._prepare_features(features_df)
        
        if len(X) == 0:
            raise ValueError("No valid samples found")
        
        # Time-series cross-validation
        cv_folds = get_time_series_folds(
            timestamps=features_df.index,
            n_splits=n_splits,
            embargo_pct=0.02
        )
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            logger.info("Optimizing hyperparameters...")
            self.config = self._optimize_hyperparameters(X, y, cv_folds, feature_names)
        
        # Train on each fold
        fold_results = []
        models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            logger.info(f"Training fold {fold_idx + 1}/{len(cv_folds)}")
            
            # Create model adapter
            model_adapter = LightGBMModelAdapter(
                name=f"LightGBM_{symbol}_fold_{fold_idx}",
                config=self.config
            )
            
            # Train
            model_adapter.fit(X, y, train_idx, val_idx, feature_names=feature_names)
            
            # Evaluate
            fold_metrics = self._evaluate_fold(
                model_adapter, X, y, train_idx, val_idx, features_df
            )
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)
            
            models.append(model_adapter)
        
        # Aggregate results
        results = self._aggregate_fold_results(fold_results)
        results['symbol'] = symbol
        results['metadata'] = metadata.__dict__
        results['validation_report'] = validation_report
        results['feature_names'] = feature_names
        
        # Analyze feature importance across folds
        results['feature_importance'] = self._aggregate_feature_importance(models)
        
        # Train final model on full dataset if requested
        if save_artifacts:
            final_model = self._train_final_model(X, y, feature_names, symbol)
            
            # Calibrate probabilities if requested
            if calibrate:
                final_model, calibrator = self._calibrate_model(final_model, X, y)
                results['calibrated'] = True
            else:
                calibrator = None
                results['calibrated'] = False
            
            # Save artifacts
            self._save_artifacts(final_model, calibrator, symbol, results)
        
        logger.info(f"LightGBM training completed for {symbol}: {results['avg_net_sharpe']:.4f} net Sharpe")
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for LightGBM."""
        # Select features (exclude target and non-numeric columns)
        feature_columns = [col for col in df.columns 
                          if col != 'target' and df[col].dtype in ['float64', 'int64']]
        
        if len(feature_columns) == 0:
            raise ValueError("No numeric features found")
        
        # Extract features and targets
        X = df[feature_columns].values
        y = df['target'].values if 'target' in df.columns else None
        
        if y is None:
            raise ValueError("No target column found")
        
        # Remove samples with NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X, y, feature_columns
    
    def _evaluate_fold(self, 
                      model: LightGBMModelAdapter,
                      X: np.ndarray, 
                      y: np.ndarray,
                      train_idx: np.ndarray,
                      val_idx: np.ndarray,
                      features_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on validation fold."""
        
        # Get predictions
        val_proba = model.predict_proba(X[val_idx])
        
        # Get returns for cost-aware evaluation
        returns_col = 'returns_1' if 'returns_1' in features_df.columns else None
        if returns_col is None:
            # Calculate returns if not available
            returns = features_df['close'].pct_change().fillna(0).values
        else:
            returns = features_df[returns_col].fillna(0).values
        
        # Align returns with validation indices
        val_returns = returns[val_idx]
        
        # Find optimal threshold and evaluate
        optimal_threshold, metrics = self.cost_evaluator.find_optimal_threshold(
            y_true=y[val_idx],
            y_pred_proba=val_proba,
            returns=val_returns,
            position_size=self.config.get('position_size', 1000.0)
        )
        
        return {
            'optimal_threshold': optimal_threshold,
            'net_sharpe': metrics.net_sharpe_ratio,
            'gross_return': metrics.gross_return,
            'net_return': metrics.net_return,
            'num_trades': metrics.num_trades,
            'win_rate': metrics.win_rate,
            'max_drawdown': metrics.max_drawdown,
            'total_costs': metrics.total_costs,
            'feature_importance': model.get_feature_importance()
        }
    
    def _aggregate_fold_results(self, fold_results: list) -> Dict[str, Any]:
        """Aggregate results across folds."""
        if not fold_results:
            return {}
        
        # Calculate averages
        metrics = ['net_sharpe', 'gross_return', 'net_return', 'win_rate', 'max_drawdown']
        aggregated = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        # Sum total metrics
        sum_metrics = ['num_trades', 'total_costs']
        for metric in sum_metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'total_{metric.replace("total_", "")}'] = np.sum(values)
        
        # Average optimal thresholds
        thresholds = [fold['optimal_threshold'] for fold in fold_results if 'optimal_threshold' in fold]
        if thresholds:
            aggregated['avg_optimal_threshold'] = np.mean(thresholds)
            aggregated['std_optimal_threshold'] = np.std(thresholds)
        
        aggregated['n_folds'] = len(fold_results)
        aggregated['fold_results'] = fold_results
        
        return aggregated
    
    def _aggregate_feature_importance(self, models: List[LightGBMModelAdapter]) -> Dict[str, Any]:
        """Aggregate feature importance across models."""
        if not models:
            return {}
        
        # Collect importance from all models
        importance_by_feature = {}
        for model in models:
            importance = model.get_feature_importance()
            for feature, score in importance.items():
                if feature not in importance_by_feature:
                    importance_by_feature[feature] = []
                importance_by_feature[feature].append(score)
        
        # Calculate statistics
        feature_stats = {}
        for feature, scores in importance_by_feature.items():
            feature_stats[feature] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        # Sort by mean importance
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        return {
            'feature_stats': feature_stats,
            'top_features': [f[0] for f in sorted_features[:20]],  # Top 20
            'feature_ranking': [f[0] for f in sorted_features]
        }
    
    def _train_final_model(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          feature_names: List[str],
                          symbol: str) -> LightGBMModelAdapter:
        """Train final model on full dataset."""
        # Use 80/20 split for final training
        split_idx = int(0.8 * len(X))
        train_idx = np.arange(split_idx)
        val_idx = np.arange(split_idx, len(X))
        
        final_model = LightGBMModelAdapter(
            name=f"LightGBM_{symbol}_final",
            config=self.config
        )
        
        final_model.fit(X, y, train_idx, val_idx, feature_names=feature_names)
        return final_model
    
    def _calibrate_model(self, 
                        model: LightGBMModelAdapter, 
                        X: np.ndarray, 
                        y: np.ndarray) -> Tuple[Any, Any]:
        """Calibrate model probabilities."""
        # Use last 20% for calibration
        split_idx = int(0.8 * len(X))
        X_cal = X[split_idx:]
        y_cal = y[split_idx:]
        
        calibrated_model, calibrator = calibrate_model_probabilities(
            model=model,
            X_cal=X_cal,
            y_cal=y_cal,
            method="isotonic"
        )
        
        return calibrated_model, calibrator
    
    def _optimize_hyperparameters(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray,
                                 cv_folds: List[Tuple[np.ndarray, np.ndarray]],
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Optimize hyperparameters using cross-validation."""
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not available, skipping hyperparameter optimization")
            return self.config
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            }
            
            # Merge with base config
            config = {**self.config, **params}
            
            # Cross-validate
            fold_scores = []
            for train_idx, val_idx in cv_folds[:3]:  # Use first 3 folds for speed
                model = LightGBMModelAdapter(config=config)
                model.fit(X, y, train_idx, val_idx, feature_names=feature_names)
                
                val_proba = model.predict_proba(X[val_idx])
                
                # Use AUC as optimization metric
                from sklearn.metrics import roc_auc_score
                try:
                    score = roc_auc_score(y[val_idx], val_proba)
                    fold_scores.append(score)
                except:
                    fold_scores.append(0.5)  # Fallback score
            
            return np.mean(fold_scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, timeout=600)  # 10 minutes max
        
        # Get best parameters
        best_params = study.best_params
        optimized_config = {**self.config, **best_params}
        
        logger.info(f"Hyperparameter optimization completed. Best AUC: {study.best_value:.4f}")
        return optimized_config
    
    def _save_artifacts(self, 
                       model: LightGBMModelAdapter,
                       calibrator: Optional[Any],
                       symbol: str,
                       results: Dict[str, Any]) -> None:
        """Save model artifacts."""
        # Create artifact directory
        artifacts_dir = Path("models") / "lightgbm" / symbol.lower()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / "model.pkl"
        model.save(model_path)
        
        # Save calibrator if available
        if calibrator:
            calibrator_path = artifacts_dir / "calibrator.pkl"
            calibrator.save(calibrator_path)
        
        # Save feature importance
        importance_path = artifacts_dir / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(results['feature_importance'], f, indent=2, default=str)
        
        # Save training results
        results_path = artifacts_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create latest symlink
        latest_dir = artifacts_dir.parent / "latest"
        if latest_dir.is_symlink():
            latest_dir.unlink()
        elif latest_dir.exists():
            import shutil
            shutil.rmtree(latest_dir)
        
        latest_dir.symlink_to(artifacts_dir.name, target_is_directory=True)
        
        logger.info(f"Artifacts saved to {artifacts_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LightGBM configuration."""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 200,
            'early_stopping_rounds': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'position_size': 1000.0
        }