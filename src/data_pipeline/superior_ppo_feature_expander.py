"""Superior PPO Feature Expander
================================

Generates the forward-looking, multi-timeframe feature set required by the
"superior" PPO models that were trained with predictive targets across
multiple horizons. The expander produces a deterministic 103-feature matrix
covering:

* Five predictive horizons (1h, 3h, 6h, 12h, 24h)
* Risk-adjusted performance metrics per horizon
* Regime-confidence signals with transaction cost adjustments
* Lightweight global market context features (volatility, liquidity, trend)

The implementation mirrors the philosophy captured in the training notebooks
so that the exported models found under ``models/superior/{SYMBOL}`` receive the
same feature ordering at inference time.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


class SuperiorPPOFeatureExpander:
    """Generate the 103-feature input matrix expected by superior PPO models."""

    def __init__(self) -> None:
        self.expected_features = 103
        # Predictive horizons expressed in number of 30m bars
        self.horizons: "OrderedDict[str, int]" = OrderedDict(
            [
                ("1h", 2),
                ("3h", 6),
                ("6h", 12),
                ("12h", 24),
                ("24h", 48),
            ]
        )
        self.feature_names: List[str] = []
        self.clip_range = float(os.getenv("SUPERIOR_PPO_CLIP_RANGE", "10"))
        self.transaction_cost = float(os.getenv("SUPERIOR_PPO_TRANSACTION_COST", "0.0010"))
        self.slippage_cost = float(os.getenv("SUPERIOR_PPO_SLIPPAGE", "0.0005"))
        self.pin_feature_index = os.getenv("SUPERIOR_PPO_PIN_FEATURE_INDEX", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        env_name = os.getenv("ENVIRONMENT", os.getenv("APP_ENV", "")).lower()
        default_save_missing = "false" if env_name in {"prod", "production"} else "true"
        self.save_missing_index = os.getenv(
            "SUPERIOR_PPO_SAVE_MISSING_INDEX", default_save_missing
        ).lower() in {"1", "true", "yes"}
        self.index_base_dir = os.getenv("SUPERIOR_PPO_FEATURE_INDEX_DIR", "models/superior")

    # ------------------------------------------------------------------
    def expand_features(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """Expand OHLCV dataframe to the 103-feature superior PPO matrix."""
        self._validate_input(df)

        working_df = df.copy()
        working_df.sort_index(inplace=True)
        self.feature_names = []  # reset for each run

        feature_frame = pd.DataFrame(index=working_df.index)

        for horizon_label, window in self.horizons.items():
            horizon_features = self._compute_horizon_features(working_df, window, horizon_label)
            for name, series in horizon_features.items():
                feature_frame[name] = series
                self.feature_names.append(name)

        # Append global diagnostic features
        global_features = self._compute_global_features(working_df)
        for name, series in global_features.items():
            feature_frame[name] = series
            self.feature_names.append(name)

        # Merge with original OHLCV data to keep context columns
        expanded_df = pd.concat([working_df, feature_frame], axis=1)

        expanded_df = self._clean_features(expanded_df)
        expanded_df = self._ensure_feature_count(expanded_df)

        if symbol:
            try:
                pinned = self._apply_feature_index(expanded_df, symbol)
                if pinned is not None:
                    expanded_df = pinned
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Superior feature pinning skipped for {symbol}: {exc}")

        return expanded_df

    # ------------------------------------------------------------------
    def validate_features(self, df: pd.DataFrame) -> bool:
        excluded = {"open", "high", "low", "close", "volume", "timestamp", "target"}
        feature_cols = [c for c in df.columns if c not in excluded]
        is_valid = len(feature_cols) == self.expected_features
        if is_valid:
            logger.info(f"✅ Superior PPO features validated: {len(feature_cols)}")
        else:
            logger.error(
                "❌ Superior PPO features invalid: expected %s, got %s",
                self.expected_features,
                len(feature_cols),
            )
        return is_valid

    # ------------------------------------------------------------------
    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    # ------------------------------------------------------------------
    def _compute_horizon_features(
        self, df: pd.DataFrame, window: int, label: str
    ) -> Dict[str, pd.Series]:
        price = df["close"].astype(float)
        returns = price.pct_change(window)
        log_returns = np.log(price / price.shift(window))

        transaction_cost = self.transaction_cost + self.slippage_cost
        cost_adj = returns - transaction_cost

        rolling_returns = returns.rolling(window)
        win_rate = rolling_returns.apply(lambda x: np.mean(x > 0), raw=True)
        loss_rate = rolling_returns.apply(lambda x: np.mean(x < 0), raw=True)

        avg_gain = rolling_returns.apply(
            lambda x: np.mean(x[x > 0]) if np.any(x > 0) else 0.0,
            raw=False,
        )
        avg_loss = rolling_returns.apply(
            lambda x: np.abs(np.mean(x[x < 0])) if np.any(x < 0) else 0.0,
            raw=False,
        )

        profit_factor = avg_gain / (avg_loss + 1e-8)

        volatility = rolling_returns.std()
        ewm_vol = returns.ewm(span=max(2, window * 2), adjust=False).std()
        sharpe = rolling_returns.mean() / (volatility + 1e-8)
        downside_std = rolling_returns.apply(
            lambda x: np.sqrt(np.mean(np.square(np.minimum(x, 0)))), raw=False
        )
        sortino = rolling_returns.mean() / (downside_std + 1e-8)

        momentum = price.pct_change().rolling(window).sum()
        ema = price.ewm(span=max(2, window * 2), adjust=False).mean()
        ema_gap = price / (ema + 1e-8) - 1.0

        rolling_mean = price.rolling(window).mean()
        rolling_std = price.rolling(window).std()
        zscore = (price - rolling_mean) / (rolling_std + 1e-8)

        rolling_max = price.rolling(window, min_periods=1).max()
        drawdown = price / (rolling_max + 1e-8) - 1.0
        max_drawdown = drawdown.rolling(window, min_periods=1).min()

        regime_confidence = (win_rate - loss_rate).clip(-1.0, 1.0)

        horizon_returns = rolling_returns.apply(lambda x: np.sum(x), raw=True)
        horizon_skew = rolling_returns.apply(lambda x: skew(x, bias=False), raw=False)
        horizon_kurt = rolling_returns.apply(lambda x: kurtosis(x, bias=False), raw=False)

        features = OrderedDict(
            [
                (f"return_{label}", returns),
                (f"log_return_{label}", log_returns),
                (f"cost_adj_return_{label}", cost_adj),
                (f"volatility_{label}", volatility),
                (f"ewm_volatility_{label}", ewm_vol),
                (f"momentum_{label}", momentum),
                (f"ema_gap_{label}", ema_gap),
                (f"price_zscore_{label}", zscore),
                (f"drawdown_{label}", drawdown),
                (f"max_drawdown_{label}", max_drawdown),
                (f"sharpe_{label}", sharpe),
                (f"sortino_{label}", sortino),
                (f"win_rate_{label}", win_rate),
                (f"loss_rate_{label}", loss_rate),
                (f"profit_factor_{label}", profit_factor),
                (f"skew_{label}", horizon_skew),
                (f"kurtosis_{label}", horizon_kurt),
                (f"avg_gain_{label}", avg_gain),
                (f"avg_loss_{label}", avg_loss),
                (f"regime_confidence_{label}", regime_confidence),
            ]
        )

        return features

    # ------------------------------------------------------------------
    def _compute_global_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        price = df["close"].astype(float)
        volume = df.get("volume", pd.Series(index=df.index, dtype=float))

        returns = price.pct_change()
        global_vol = returns.rolling(48).std()
        trend_strength = price.rolling(96).apply(lambda x: x[-1] / (x[0] + 1e-8) - 1.0, raw=True)
        volume_roll = volume.rolling(96)
        liquidity_zscore = (volume - volume_roll.mean()) / (volume_roll.std() + 1e-8)

        features = OrderedDict(
            [
                ("global_volatility_24h", global_vol),
                ("global_trend_strength", trend_strength),
                ("global_liquidity_zscore", liquidity_zscore),
            ]
        )

        return features

    # ------------------------------------------------------------------
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        df[numeric_cols] = df[numeric_cols].clip(-self.clip_range, self.clip_range)
        return df

    # ------------------------------------------------------------------
    def _ensure_feature_count(self, df: pd.DataFrame) -> pd.DataFrame:
        excluded = {"open", "high", "low", "close", "volume", "timestamp", "target"}
        feature_cols = [c for c in df.columns if c not in excluded]

        if len(feature_cols) > self.expected_features:
            # Truncate deterministically, keeping earliest defined features
            keep = feature_cols[: self.expected_features]
            drop = [c for c in feature_cols if c not in keep]
            df = df.drop(columns=drop)
            self.feature_names = keep
        elif len(feature_cols) < self.expected_features:
            pad_needed = self.expected_features - len(feature_cols)
            for idx in range(pad_needed):
                col = f"padding_feature_{len(feature_cols) + idx}"
                df[col] = 0.0
                feature_cols.append(col)
            self.feature_names = feature_cols
        else:
            self.feature_names = feature_cols

        return df

    # ------------------------------------------------------------------
    def _apply_feature_index(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        index_names = self._load_feature_index(symbol)
        if not index_names:
            if self.pin_feature_index and not self.save_missing_index:
                raise RuntimeError(
                    "SUPERIOR_PPO_FEATURE_INDEX_MISSING_STRICT: feature index missing and strict pinning enabled"
                )
            if self.save_missing_index and self.feature_names:
                self._save_feature_index(symbol, self.feature_names)
            return None

        target_names = list(index_names)[: self.expected_features]
        expanded = df.copy()
        for name in target_names:
            if name not in expanded.columns:
                expanded[name] = 0.0

        preserved = [
            c
            for c in ["open", "high", "low", "close", "volume", "timestamp", "target"]
            if c in df.columns
        ]
        ordered = preserved + target_names
        expanded = expanded[ordered]
        self.feature_names = target_names
        return expanded

    # ------------------------------------------------------------------
    def _validate_input(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        required = {"open", "high", "low", "close"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    def _index_path(self, symbol: str) -> str:
        base = os.path.join(self.index_base_dir, symbol)
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "feature_index.json")

    # ------------------------------------------------------------------
    def _load_feature_index(self, symbol: str) -> Optional[List[str]]:
        try:
            path = self._index_path(symbol)
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to load superior feature index for {symbol}: {exc}")
        return None

    # ------------------------------------------------------------------
    def _save_feature_index(self, symbol: str, names: List[str]) -> None:
        try:
            path = self._index_path(symbol)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(names, handle, indent=2)
            logger.info(
                f"Saved superior PPO feature index for {symbol}: {len(names)} entries -> {path}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to save superior feature index for {symbol}: {exc}")


__all__ = ["SuperiorPPOFeatureExpander"]
