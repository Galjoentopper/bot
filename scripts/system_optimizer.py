#!/usr/bin/env python3
"""
System Optimizer
===============

Production-ready system optimization without external dependencies.
Optimizes memory, threading, and system performance for trading bot.
"""

import gc
import hashlib
import logging
import os
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LightweightSystemOptimizer:
    """Lightweight system optimizer without external dependencies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize system optimizer."""
        self.config = config

        perf_config = config.get("performance_optimization", {})
        self.gc_interval = perf_config.get("garbage_collection_interval", 300)
        self.enable_caching = perf_config.get("enable_prediction_caching", True)
        self.cache_size = perf_config.get("cache_size", 1000)
        self.cache_ttl = perf_config.get("cache_ttl_seconds", 600)

        # Initialize caches
        if self.enable_caching:
            self.prediction_cache = OrderedDict()
            self.cache_timestamps = {}
            self.cache_hits = 0
            self.cache_misses = 0

        # Optimization tracking
        self.last_gc_time = 0
        self.optimizations_run = 0

        # Threading optimization
        self.optimal_threads = min(8, os.cpu_count() or 4)
        self._setup_thread_optimization()

        logger.info(
            f"System Optimizer initialized (caching={'enabled' if self.enable_caching else 'disabled'})"
        )

    def _setup_thread_optimization(self):
        """Setup optimal threading configuration."""
        try:
            # Set environment variables for optimal threading
            os.environ["OMP_NUM_THREADS"] = str(self.optimal_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.optimal_threads)
            os.environ["MKL_NUM_THREADS"] = str(self.optimal_threads)

            logger.info(f"Threading optimized for {self.optimal_threads} threads")

        except Exception as e:
            logger.debug(f"Thread optimization setup failed: {e}")

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage and run garbage collection."""
        try:
            # Force garbage collection
            collected_objects = gc.collect()

            # Clear various caches
            self._clear_system_caches()

            # Update tracking
            self.last_gc_time = time.time()
            self.optimizations_run += 1

            logger.info(f"Memory optimization completed: {collected_objects} objects collected")

            return {
                "objects_collected": collected_objects,
                "timestamp": datetime.now().isoformat(),
                "optimizations_run": self.optimizations_run,
            }

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {"error": str(e)}

    def _clear_system_caches(self):
        """Clear various system caches."""
        try:
            # Clear import caches
            if hasattr(sys, "_clear_type_cache"):
                sys._clear_type_cache()

            # Clear regex cache
            import re

            re.purge()

            # Clear our own expired cache entries
            if self.enable_caching:
                self._clear_expired_cache()

        except Exception as e:
            logger.debug(f"Cache clearing failed: {e}")

    def _clear_expired_cache(self):
        """Clear expired prediction cache entries."""
        try:
            current_time = time.time()
            expired_keys = [
                key
                for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > self.cache_ttl
            ]

            for key in expired_keys:
                self.prediction_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)

            if expired_keys:
                logger.debug(f"Cleared {len(expired_keys)} expired cache entries")

        except Exception as e:
            logger.debug(f"Cache cleanup failed: {e}")

    def get_cached_prediction(
        self, symbol: str, features_hash: str, model_type: str
    ) -> Optional[float]:
        """Get cached prediction if available."""
        if not self.enable_caching:
            return None

        try:
            cache_key = f"{symbol}:{model_type}:{features_hash}"

            # Check if cached and not expired
            if cache_key in self.prediction_cache:
                timestamp = self.cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp < self.cache_ttl:
                    # Move to end (LRU)
                    prediction = self.prediction_cache.pop(cache_key)
                    self.prediction_cache[cache_key] = prediction
                    self.cache_hits += 1
                    return prediction
                else:
                    # Expired - remove
                    self.prediction_cache.pop(cache_key, None)
                    self.cache_timestamps.pop(cache_key, None)

            self.cache_misses += 1
            return None

        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
            self.cache_misses += 1
            return None

    def cache_prediction(self, symbol: str, features_hash: str, model_type: str, prediction: float):
        """Cache a prediction result."""
        if not self.enable_caching:
            return

        try:
            cache_key = f"{symbol}:{model_type}:{features_hash}"

            # Remove oldest if at capacity
            if (
                len(self.prediction_cache) >= self.cache_size
                and cache_key not in self.prediction_cache
            ):
                oldest_key = next(iter(self.prediction_cache))
                self.prediction_cache.pop(oldest_key)
                self.cache_timestamps.pop(oldest_key, None)

            self.prediction_cache[cache_key] = prediction
            self.cache_timestamps[cache_key] = time.time()

        except Exception as e:
            logger.debug(f"Cache put failed: {e}")

    def hash_features(self, features_df) -> str:
        """Generate hash for feature DataFrame."""
        try:
            # Simple hash based on DataFrame shape and some values
            shape_str = f"{features_df.shape[0]}x{features_df.shape[1]}"

            # Sample some values for hash
            if not features_df.empty:
                sample_values = str(features_df.iloc[0].sum()) + str(features_df.iloc[-1].sum())
            else:
                sample_values = "empty"

            hash_input = f"{shape_str}:{sample_values}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:12]

        except Exception as e:
            logger.debug(f"Feature hashing failed: {e}")
            return str(int(time.time()))

    def should_run_gc(self) -> bool:
        """Check if garbage collection should run."""
        return (time.time() - self.last_gc_time) > self.gc_interval

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.enable_caching:
            return {"caching_enabled": False}

        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "caching_enabled": True,
            "cache_size": len(self.prediction_cache),
            "cache_capacity": self.cache_size,
            "hit_rate_pct": hit_rate,
            "total_hits": self.cache_hits,
            "total_misses": self.cache_misses,
            "total_requests": total_requests,
        }

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization system status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "optimizations_run": self.optimizations_run,
            "last_gc_time": self.last_gc_time,
            "gc_interval_seconds": self.gc_interval,
            "optimal_threads": self.optimal_threads,
            "cache_statistics": self.get_cache_statistics(),
            "active_threads": threading.active_count(),
            "memory_info": {"gc_counts": gc.get_count(), "gc_thresholds": gc.get_threshold()},
        }

    def optimize_system(self) -> Dict[str, Any]:
        """Run comprehensive system optimization."""
        try:
            logger.info("Running system optimization...")
            start_time = time.time()

            results = {}

            # Memory optimization
            memory_results = self.optimize_memory()
            results["memory_optimization"] = memory_results

            # Cache cleanup
            if self.enable_caching:
                self._clear_expired_cache()
                results["cache_cleanup"] = True

            # Update statistics
            results["optimization_duration"] = time.time() - start_time
            results["status"] = self.get_optimization_status()

            logger.info(f"System optimization completed in {results['optimization_duration']:.2f}s")

            return results

        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            return {"error": str(e)}


class PredictionCacheManager:
    """Manages prediction caching for the trading system."""

    def __init__(self, optimizer: LightweightSystemOptimizer):
        """Initialize prediction cache manager."""
        self.optimizer = optimizer

    def get_prediction_with_cache(
        self, symbol: str, features_df, model_type: str, prediction_func
    ) -> Optional[float]:
        """Get prediction with caching support."""
        try:
            # Generate features hash
            features_hash = self.optimizer.hash_features(features_df)

            # Try cache first
            cached_prediction = self.optimizer.get_cached_prediction(
                symbol, features_hash, model_type
            )
            if cached_prediction is not None:
                return cached_prediction

            # Cache miss - compute prediction
            prediction = prediction_func()

            # Cache the result
            if prediction is not None:
                self.optimizer.cache_prediction(symbol, features_hash, model_type, prediction)

            return prediction

        except Exception as e:
            logger.error(f"Cached prediction failed for {symbol}:{model_type}: {e}")
            # Fall back to direct prediction
            return prediction_func()


def create_system_optimizer(config: Dict[str, Any]) -> LightweightSystemOptimizer:
    """Factory function to create system optimizer."""
    return LightweightSystemOptimizer(config)


if __name__ == "__main__":
    # Test the system optimizer
    test_config = {
        "performance_optimization": {
            "enable_prediction_caching": True,
            "cache_size": 1000,
            "cache_ttl_seconds": 300,
            "garbage_collection_interval": 60,
        }
    }

    optimizer = create_system_optimizer(test_config)

    # Test optimization
    results = optimizer.optimize_system()
    print("Optimization results:")
    print(f"- Objects collected: {results['memory_optimization']['objects_collected']}")
    print(f"- Duration: {results['optimization_duration']:.3f}s")

    # Test caching
    import numpy as np
    import pandas as pd

    test_df = pd.DataFrame(np.random.randn(100, 10))
    features_hash = optimizer.hash_features(test_df)

    optimizer.cache_prediction("BTCEUR", features_hash, "gru", 0.5)
    cached_value = optimizer.get_cached_prediction("BTCEUR", features_hash, "gru")

    print(f"Cache test: {cached_value}")

    stats = optimizer.get_cache_statistics()
    print(f"Cache stats: {stats}")
