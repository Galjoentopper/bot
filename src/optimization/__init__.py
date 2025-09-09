"""
Optimization Module
==================

Contains hyperparameter optimization and financial optimization components:
- BayesianOptimizer: Bayesian optimization for hyperparameter tuning
- FinancialHyperopt: Financial-specific hyperparameter optimization
"""

from .bayesian_optimizer import BayesianOptimizationResult, FinancialBayesianOptimizer
from .financial_hyperopt import AssetClass, MarketRegime

__all__ = [
    "FinancialBayesianOptimizer",
    "BayesianOptimizationResult",
    "MarketRegime",
    "AssetClass",
]
