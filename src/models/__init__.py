"""
Models Module
=============

Contains all machine learning models for the crypto trading bot:
- GRU sequence model for price prediction
- LightGBM feature model for refined predictions
- PPO reinforcement learning agent for trading decisions
- StableGRU: Enhanced GRU with stability improvements
- ModelManager: Unified model management interface
- Adapters: Model adaptation layers
"""

from .adapters import GRUAdapter, LightGBMAdapter
from .base_adapter import BaseModelAdapter
from .gru_trainer import GRUTrainer
from .lgbm_trainer import LightGBMTrainer
from .model_manager import ModelManager
from .ppo_trainer import PPOTrainer
from .stable_gru_trainer import StableGRUTrainer

__all__ = [
    "GRUTrainer",
    "LightGBMTrainer",
    "PPOTrainer",
    "StableGRUTrainer",
    "ModelManager",
    "BaseModelAdapter",
    "GRUAdapter",
    "LightGBMAdapter",
]
