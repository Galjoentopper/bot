"""Adapter layer for legacy components."""

from .config_adapter import ConfigAdapter
from .feature_adapter import FeatureAdapter
from .trader_adapter import TraderAdapter

__all__ = ["ConfigAdapter", "FeatureAdapter", "TraderAdapter"]
