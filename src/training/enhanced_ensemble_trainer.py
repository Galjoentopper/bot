"""
Enhanced Ensemble Training System
=================================

Comprehensive training system that integrates all enhanced components:
- Trading-optimized models (GRU, LightGBM, PPO)
- Advanced feature engineering
- Trading-specific loss functions and metrics
- Intelligent ensemble construction
- Comprehensive validation framework
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data_pipeline.features import FeatureEngine
from ..ensemble.trading_ensemble import TradingEnsemble
from ..evaluation.trading_evaluator import TradingModelEvaluator
from ..models.enhanced_ppo_trainer import EnhancedPPOTrainer
from ..models.trading_gru_trainer import TradingGRUTrainer
from ..models.trading_lgbm import TradingLightGBM
from ..utils.trading_metrics import TradingMetricsCalculator
from ..validation.data_validator import TradingDataValidator
from ..validation.ensemble_validator import EnsembleValidator

logger = logging.getLogger(__name__)


class EnhancedEnsembleTrainer:
    """
    Comprehensive trainer that integrates all enhanced trading components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced ensemble trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get("training", {})

        # Initialize components
        self.feature_engine = FeatureEngine(config)
        self.data_validator = TradingDataValidator(config)
        self.ensemble_validator = EnsembleValidator(config.get("ensemble_validation", {}))
        self.model_evaluator = TradingModelEvaluator(config.get("evaluation", {}))
        self.metrics_calc = TradingMetricsCalculator()

        # Model trainers
        self.trainers = {
            "trading_gru": TradingGRUTrainer(config),
            "trading_lgbm": TradingLightGBM(config),
            "enhanced_ppo": EnhancedPPOTrainer(config),
        }

        # Training parameters
        self.enabled_models = config.get("enabled_models", list(self.trainers.keys()))
        self.ensemble_optimization = config.get("ensemble_optimization", True)
        self.comprehensive_validation = config.get("comprehensive_validation", True)

        # Results storage
        self.training_results = {}
        self.ensemble_results = {}
        self.validation_results = {}

        logger.info(f"Enhanced ensemble trainer initialized with models: {self.enabled_models}")

    def train_complete_system(
        self, data: pd.DataFrame, symbol: str, target_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Train the complete enhanced trading system.

        Args:
            data: Market data with OHLCV
            symbol: Trading symbol
            target_column: Target column for prediction

        Returns:
            Complete training results
        """
        logger.info(f"Starting complete system training for {symbol}")

        try:
            # Step 1: Data validation and preparation
            logger.info("Step 1: Data validation and feature engineering")
            processed_data = self._prepare_data(data, symbol, target_column)

            if processed_data is None:
                logger.error("Data preparation failed")
                return {}

            X, y, prices, feature_names = processed_data

            # Step 2: Train individual models
            logger.info("Step 2: Training individual models")
            trained_models = self._train_individual_models(X, y, prices, symbol)

            if not trained_models:
                logger.error("No models trained successfully")
                return {}

            # Step 3: Create and optimize ensemble
            logger.info("Step 3: Creating and optimizing ensemble")
            ensemble_results = self._create_optimized_ensemble(trained_models, X, y, prices, symbol)

            # Step 4: Comprehensive validation
            if self.comprehensive_validation:
                logger.info("Step 4: Comprehensive validation")
                validation_results = self._comprehensive_validation(
                    trained_models, ensemble_results.get("ensemble"), X, y, prices, symbol
                )
            else:
                validation_results = {}

            # Step 5: Generate final results
            final_results = self._compile_final_results(
                trained_models, ensemble_results, validation_results, symbol
            )

            logger.info(f"Complete system training finished for {symbol}")
            return final_results

        except Exception as e:
            logger.error(f"Error in complete system training: {e}")
            return {}

    def _prepare_data(
        self, data: pd.DataFrame, symbol: str, target_column: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]]:
        """Prepare and validate data for training."""
        try:
            # Validate raw data
            validation_result = self.data_validator.validate_for_training(data, symbol)

            if not validation_result["is_valid"]:
                logger.error(f"Data validation failed: {validation_result.get('errors', [])}")
                return None

            # Generate enhanced features and targets
            logger.info("Generating enhanced features and trading targets")
            enhanced_data = self.feature_engine.generate_features_and_targets(data, symbol)

            # Extract features and targets
            feature_columns = [
                col
                for col in enhanced_data.columns
                if col not in ["open", "high", "low", "close", "volume"]
                and not col.startswith("target_")
            ]

            X = enhanced_data[feature_columns].values

            # Use trading-optimized target if available
            if "target_1h" in enhanced_data.columns:
                y = enhanced_data["target_1h"].values
                logger.info("Using trading-optimized 1h target")
            elif "target" in enhanced_data.columns:
                y = enhanced_data["target"].values
                logger.info("Using default target")
            else:
                # Fallback: use price changes
                y = enhanced_data[target_column].pct_change().fillna(0).values
                logger.info("Using fallback price change target")

            # Extract prices for trading metrics
            prices = enhanced_data[target_column].values

            # Handle NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            prices = prices[valid_mask]

            logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")

            return X, y, prices, feature_columns

        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            return None

    def _train_individual_models(
        self, X: np.ndarray, y: np.ndarray, prices: np.ndarray, symbol: str
    ) -> Dict[str, Any]:
        """Train individual models."""
        trained_models = {}

        # Create train/validation split
        train_size = int(0.8 * len(X))
        train_idx = np.arange(train_size)
        valid_idx = np.arange(train_size, len(X))

        for model_name in self.enabled_models:
            if model_name not in self.trainers:
                logger.warning(f"Trainer not found for {model_name}")
                continue

            logger.info(f"Training {model_name} for {symbol}")

            try:
                trainer = self.trainers[model_name]

                # Train model
                training_result = trainer.train(X, y, train_idx, valid_idx, symbol)

                if training_result and "model" in training_result:
                    # For PPO, we need to extract the policy network or use the trainer itself
                    if model_name == "enhanced_ppo":
                        trained_models[model_name] = trainer  # Use trainer as model for PPO
                    elif hasattr(training_result, "get") and training_result.get("model"):
                        trained_models[model_name] = training_result["model"]
                    elif hasattr(training_result, "get") and "models" in training_result:
                        # For ensemble models like trading_lgbm
                        trained_models[model_name] = training_result["models"]["main"]
                    else:
                        # Use the trainer itself
                        trained_models[model_name] = trainer

                    # Store training results
                    self.training_results[model_name] = training_result

                    logger.info(f"Successfully trained {model_name}")
                else:
                    logger.error(f"Training failed for {model_name}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models

    def _create_optimized_ensemble(
        self,
        trained_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        symbol: str,
    ) -> Dict[str, Any]:
        """Create and optimize ensemble."""
        ensemble_results = {}

        try:
            # Create ensemble with default configuration
            ensemble_config = self.config.get(
                "ensemble",
                {
                    "weighting_method": "dynamic",
                    "static_weights": {
                        name: 1.0 / len(trained_models) for name in trained_models.keys()
                    },
                },
            )

            ensemble = TradingEnsemble(ensemble_config)

            # Add models to ensemble
            for name, model in trained_models.items():
                ensemble.add_model(name, model)

            ensemble_results["ensemble"] = ensemble
            ensemble_results["config"] = ensemble_config

            # Optimize ensemble if requested
            if self.ensemble_optimization and len(trained_models) > 1:
                logger.info("Optimizing ensemble weights")

                try:
                    optimization_results = self.ensemble_validator.optimize_ensemble_weights(
                        trained_models, X, y, prices, symbol
                    )

                    ensemble_results["optimization"] = optimization_results

                    # Create optimized ensemble
                    if optimization_results and "best_params" in optimization_results:
                        optimized_config = ensemble_config.copy()
                        optimized_config.update(optimization_results["best_params"])

                        optimized_ensemble = TradingEnsemble(optimized_config)
                        for name, model in trained_models.items():
                            optimized_ensemble.add_model(name, model)

                        ensemble_results["optimized_ensemble"] = optimized_ensemble
                        ensemble_results["optimized_config"] = optimized_config

                        logger.info("Ensemble optimization completed")

                except Exception as e:
                    logger.error(f"Ensemble optimization failed: {e}")

            # Test ensemble predictions
            test_size = min(100, len(X) // 4)
            test_X = X[-test_size:]
            test_prices = prices[-test_size:]

            ensemble_pred = ensemble.predict(test_X, test_prices)
            ensemble_results["test_predictions"] = ensemble_pred

            logger.info("Ensemble creation completed")

        except Exception as e:
            logger.error(f"Error in ensemble creation: {e}")

        return ensemble_results

    def _comprehensive_validation(
        self,
        trained_models: Dict[str, Any],
        ensemble: Optional[TradingEnsemble],
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        symbol: str,
    ) -> Dict[str, Any]:
        """Perform comprehensive validation."""
        validation_results = {}

        try:
            # Individual model validation
            logger.info("Validating individual models")
            for name, model in trained_models.items():
                try:
                    model_validation = self.model_evaluator.walk_forward_validation(
                        model, X, y, prices, f"{symbol}_{name}"
                    )
                    validation_results[f"model_{name}"] = model_validation
                except Exception as e:
                    logger.error(f"Validation failed for {name}: {e}")

            # Ensemble validation
            if ensemble is not None:
                logger.info("Validating ensemble")
                try:
                    ensemble_validation = self.model_evaluator.walk_forward_validation(
                        ensemble, X, y, prices, f"{symbol}_ensemble"
                    )
                    validation_results["ensemble"] = ensemble_validation

                    # Robustness testing
                    robustness_results = self.ensemble_validator.test_ensemble_robustness(
                        ensemble, X, y, prices
                    )
                    validation_results["robustness"] = robustness_results

                    # Performance attribution
                    attribution_results = self.ensemble_validator.performance_attribution(
                        ensemble, X, y, prices
                    )
                    validation_results["attribution"] = attribution_results

                except Exception as e:
                    logger.error(f"Ensemble validation failed: {e}")

            # Benchmarking
            logger.info("Performing benchmark comparison")
            try:
                if ensemble is not None:
                    ensemble_pred = ensemble.predict(X, prices)
                    benchmark_results = self.model_evaluator.benchmark_comparison(
                        ensemble_pred, prices
                    )
                    validation_results["benchmarks"] = benchmark_results
            except Exception as e:
                logger.error(f"Benchmark comparison failed: {e}")

            logger.info("Comprehensive validation completed")

        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")

        return validation_results

    def _compile_final_results(
        self,
        trained_models: Dict[str, Any],
        ensemble_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        symbol: str,
    ) -> Dict[str, Any]:
        """Compile final training results."""
        final_results = {
            "symbol": symbol,
            "trained_models": list(trained_models.keys()),
            "individual_training_results": self.training_results,
            "ensemble_results": ensemble_results,
            "validation_results": validation_results,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Extract key performance metrics
        performance_summary = {}

        # Individual model performance
        for name in trained_models.keys():
            if f"model_{name}" in validation_results:
                model_val = validation_results[f"model_{name}"]
                if "overall" in model_val:
                    performance_summary[name] = {
                        "sharpe_ratio": model_val["overall"].get("trading_sharpe_ratio", 0),
                        "total_return": model_val["overall"].get("trading_net_return", 0),
                        "max_drawdown": model_val["overall"].get("trading_max_drawdown", 0),
                        "directional_accuracy": model_val["overall"].get("directional_accuracy", 0),
                    }

        # Ensemble performance
        if "ensemble" in validation_results and "overall" in validation_results["ensemble"]:
            ensemble_val = validation_results["ensemble"]["overall"]
            performance_summary["ensemble"] = {
                "sharpe_ratio": ensemble_val.get("trading_sharpe_ratio", 0),
                "total_return": ensemble_val.get("trading_net_return", 0),
                "max_drawdown": ensemble_val.get("trading_max_drawdown", 0),
                "directional_accuracy": ensemble_val.get("directional_accuracy", 0),
            }

        final_results["performance_summary"] = performance_summary

        # Generate summary report
        final_results["summary_report"] = self._generate_summary_report(final_results)

        logger.info("Final results compiled successfully")
        return final_results

    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary report."""
        symbol = results.get("symbol", "Unknown")

        report = f"""
Enhanced Trading System Training Report
=====================================

Symbol: {symbol}
Training Date: {results.get('timestamp', 'Unknown')}
Models Trained: {', '.join(results.get('trained_models', []))}

Performance Summary:
-------------------
"""

        perf_summary = results.get("performance_summary", {})

        for model_name, metrics in perf_summary.items():
            report += f"""
{model_name.upper()}:
  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
  Total Return: {metrics.get('total_return', 0):.2%}
  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
  Directional Accuracy: {metrics.get('directional_accuracy', 0):.3f}
"""

        # Add ensemble-specific information
        if "ensemble" in perf_summary:
            report += f"""
ENSEMBLE BENEFITS:
  Best Individual Sharpe: {max([m.get('sharpe_ratio', 0) for m in perf_summary.values() if isinstance(m, dict)]):.3f}
  Ensemble Sharpe: {perf_summary['ensemble'].get('sharpe_ratio', 0):.3f}
  Improvement: {perf_summary['ensemble'].get('sharpe_ratio', 0) - max([m.get('sharpe_ratio', 0) for m in perf_summary.values() if isinstance(m, dict)]):.3f}
"""

        # Add validation insights
        validation_results = results.get("validation_results", {})
        if "robustness" in validation_results:
            robustness = validation_results["robustness"]
            overall_robustness = robustness.get("overall_robustness_score", 0)
            report += f"""
ROBUSTNESS ANALYSIS:
  Overall Robustness Score: {overall_robustness:.3f}/1.0
  Model Stability: {'High' if overall_robustness > 0.8 else 'Medium' if overall_robustness > 0.6 else 'Low'}
"""

        return report

    def save_complete_results(self, results: Dict[str, Any], save_path: str):
        """Save complete training results."""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save models
            symbol = results.get("symbol", "unknown")

            for model_name in results.get("trained_models", []):
                if model_name in self.training_results:
                    model_path = save_path.replace(".pkl", f"_{model_name}.pkl")

                    # Save model using appropriate method
                    if model_name in self.trainers:
                        try:
                            self.trainers[model_name].save_model(model_path)
                        except Exception as e:
                            logger.error(f"Failed to save {model_name}: {e}")

            # Save ensemble
            ensemble_results = results.get("ensemble_results", {})
            if "ensemble" in ensemble_results:
                ensemble_path = save_path.replace(".pkl", "_ensemble.pkl")
                # Note: TradingEnsemble doesn't have save method implemented yet
                # This would need to be added to the TradingEnsemble class

            # Save metadata and results
            metadata_path = save_path.replace(".pkl", "_metadata.json")
            import json

            # Prepare serializable results
            serializable_results = {
                "symbol": results.get("symbol"),
                "trained_models": results.get("trained_models"),
                "performance_summary": results.get("performance_summary"),
                "ensemble_config": ensemble_results.get("config"),
                "timestamp": results.get("timestamp"),
                "summary_report": results.get("summary_report"),
            }

            with open(metadata_path, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Complete results saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
