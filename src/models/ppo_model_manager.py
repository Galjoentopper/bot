"""
PPO Model Manager
=================

Lazy-loading, singleton manager for Stable-Baselines3 PPO models used in
inference. Prevents loading multiple 1GB+ models simultaneously and applies
saved VecNormalize statistics to observations when available.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PPOModelManager:
    """LRU-cached PPO model manager with optional VecNormalize support."""

    def __init__(self, max_cached: int = 1, device: str = "auto") -> None:
        self.max_cached = max_cached
        self.device = self._resolve_device(device)
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _evict_if_needed(self) -> None:
        if len(self._cache) <= self.max_cached:
            return
        # Evict least-recently used
        path, entry = self._cache.popitem(last=False)
        try:
            # Proactively free GPU memory if applicable
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info(f"Evicted PPO model from cache: {path}")

    def _load_vec_stats(self, metadata_path: str) -> Optional[Dict[str, Any]]:
        try:
            if not os.path.exists(metadata_path):
                return None
            with open(metadata_path, "r", encoding="utf-8") as f:
                md = json.load(f)
            vec_path = md.get("vecnormalize_path")
            if not vec_path or not os.path.exists(vec_path):
                return None

            # Try to load VecNormalize object to extract obs stats
            try:
                import cloudpickle  # type: ignore

                with open(vec_path, "rb") as vf:
                    obj = cloudpickle.load(vf)
                # Expected attributes on VecNormalize
                obs_rms = getattr(obj, "obs_rms", None)
                clip_obs = getattr(obj, "clip_obs", 10.0)
                if obs_rms is not None and hasattr(obs_rms, "mean") and hasattr(obs_rms, "var"):
                    mean = np.asarray(obs_rms.mean, dtype=np.float32)
                    var = np.asarray(obs_rms.var, dtype=np.float32)
                    eps = 1e-8
                    return {"mean": mean, "var": var, "eps": eps, "clip": float(clip_obs)}
            except Exception as e:
                logger.debug(f"VecNormalize stats load failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to parse PPO metadata for VecNormalize stats: {e}")
            return None

    def load_model(self, model_path: str):
        """Load PPO model lazily and cache it. Returns the loaded SB3 model."""
        with self._lock:
            if model_path in self._cache:
                entry = self._cache.pop(model_path)
                self._cache[model_path] = entry  # mark as most recently used
                return entry.get("model")

            # Load model
            try:
                from stable_baselines3 import PPO as SB3_PPO  # type: ignore
            except Exception as e:
                logger.error(f"stable-baselines3 not available for PPO inference: {e}")
                raise

            logger.info(f"Loading PPO model: {model_path} ({self.device})")
            model = SB3_PPO.load(model_path, device=self.device)

            # Load vec stats if available
            base, _ = os.path.splitext(model_path)
            metadata_path = f"{base}_metadata.json"
            vec_stats = self._load_vec_stats(metadata_path)

            self._cache[model_path] = {"model": model, "vec_stats": vec_stats}
            self._evict_if_needed()
            return model

    def _apply_vecnormalize(self, obs: np.ndarray, vec_stats: Optional[Dict[str, Any]]) -> np.ndarray:
        if not isinstance(obs, np.ndarray) or vec_stats is None:
            return obs
        try:
            mean = vec_stats["mean"]
            var = vec_stats["var"]
            eps = float(vec_stats.get("eps", 1e-8))
            clip = float(vec_stats.get("clip", 10.0))
            # Broadcast mean/var over observation
            norm = (obs - mean) / np.sqrt(var + eps)
            return np.clip(norm, -clip, clip)
        except Exception:
            return obs

    def predict(
        self, model_path: str, observation: Any, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with self._lock:
            model = self.load_model(model_path)
            entry = self._cache.get(model_path, {})
            vec_stats = entry.get("vec_stats")

        # Convert input to numpy and apply normalization if available
        obs = observation
        try:
            if hasattr(obs, "numpy"):
                obs = obs.numpy()
            obs = np.asarray(obs, dtype=np.float32)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            obs = self._apply_vecnormalize(obs, vec_stats)
        except Exception:
            pass

        try:
            return model.predict(obs, deterministic=deterministic)
        except Exception as e:
            logger.error(f"PPO predict failed: {e}")
            # Fallback neutral action
            actions = np.zeros((1,), dtype=np.float32)
            return actions, None


_GLOBAL_MANAGER: Optional[PPOModelManager] = None


def get_ppo_manager() -> PPOModelManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = PPOModelManager(max_cached=1)
    return _GLOBAL_MANAGER

