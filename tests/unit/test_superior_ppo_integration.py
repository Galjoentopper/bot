import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.model_feature_router import ModelFeatureRouter
from src.data_pipeline.superior_ppo_feature_expander import SuperiorPPOFeatureExpander
from src.data_pipeline.trading_features import TradingFeatureEngine, generate_trading_features


def _load_trader_module():
    module_name = "trader_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    trader_path = Path(__file__).resolve().parents[2] / "bin" / "trader"
    loader = importlib.machinery.SourceFileLoader(module_name, str(trader_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def test_superior_expander_generates_expected_features(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERIOR_PPO_FEATURE_INDEX_DIR", str(tmp_path / "indices"))

    idx = pd.date_range("2024-01-01", periods=256, freq="30T")
    base_price = 25000 + np.cumsum(np.random.normal(scale=100, size=len(idx)))
    df = pd.DataFrame(
        {
            "open": base_price * np.random.uniform(0.995, 1.005, size=len(idx)),
            "high": base_price * np.random.uniform(1.000, 1.010, size=len(idx)),
            "low": base_price * np.random.uniform(0.990, 1.000, size=len(idx)),
            "close": base_price,
            "volume": np.random.uniform(10_000, 50_000, size=len(idx)),
        },
        index=idx,
    )

    expander = SuperiorPPOFeatureExpander()
    expanded = expander.expand_features(df, symbol="BTCEUR")

    excluded = {"open", "high", "low", "close", "volume", "timestamp", "target"}
    feature_cols = [c for c in expanded.columns if c not in excluded]

    assert len(feature_cols) == expander.expected_features
    assert feature_cols[0].startswith("return_1h")
    assert feature_cols[-1] == "global_liquidity_zscore"

    index_path = tmp_path / "indices" / "BTCEUR" / "feature_index.json"
    assert index_path.exists(), "Feature index should be persisted for reuse"


def test_model_feature_router_reports_superior_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("PPO_FEATURE_MODE", "superior")
    monkeypatch.setenv("SUPERIOR_PPO_FEATURE_INDEX_DIR", str(tmp_path / "indices"))

    router = ModelFeatureRouter()

    assert router.ppo_mode == "superior"
    assert router.ppo_expected_features == 103


def test_superior_loader_prefers_best_model(tmp_path, monkeypatch):
    EnhancedUnifiedPaperTrader = _load_trader_module().EnhancedUnifiedPaperTrader

    models_root = tmp_path / "models"
    superior_dir = models_root / "superior" / "BTCEUR" / "best"
    superior_dir.mkdir(parents=True)
    best_path = superior_dir / "best_model.zip"
    best_path.write_bytes(b"")

    trader = EnhancedUnifiedPaperTrader.__new__(EnhancedUnifiedPaperTrader)
    trader.models_dir = models_root
    trader.logger = type("L", (), {"logger": logging.getLogger("test")})()

    captured = {}

    def fake_loader(path, model_type):
        captured["path"] = path
        captured["type"] = model_type
        return "model"

    trader._load_model_file = fake_loader  # type: ignore[attr-defined]

    result = EnhancedUnifiedPaperTrader._load_from_superior_models(trader, "BTCEUR", "ppo")

    assert result is not None
    model, path = result
    assert model == "model"
    assert Path(path) == best_path
    assert captured["path"] == best_path
    assert captured["type"] == "ppo"


def test_superior_loader_returns_none_for_missing_symbol(tmp_path):
    EnhancedUnifiedPaperTrader = _load_trader_module().EnhancedUnifiedPaperTrader

    trader = EnhancedUnifiedPaperTrader.__new__(EnhancedUnifiedPaperTrader)
    trader.models_dir = tmp_path / "models"
    trader.logger = type("L", (), {"logger": logging.getLogger("test")})()

    trader._load_model_file = lambda *args, **kwargs: "model"  # type: ignore[attr-defined]

    result = EnhancedUnifiedPaperTrader._load_from_superior_models(trader, "BTCEUR", "ppo")
    assert result is None


def test_trading_features_risk_metrics(tmp_path):
    idx = pd.date_range("2024-01-01", periods=256, freq="30T")
    base_price = 20000 + np.cumsum(np.random.normal(scale=50, size=len(idx)))
    frame = pd.DataFrame(
        {
            "open": base_price * np.random.uniform(0.999, 1.001, size=len(idx)),
            "high": base_price * np.random.uniform(1.000, 1.005, size=len(idx)),
            "low": base_price * np.random.uniform(0.995, 1.000, size=len(idx)),
            "close": base_price,
            "volume": np.random.uniform(5_000, 15_000, size=len(idx)),
        },
        index=idx,
    )

    engine = TradingFeatureEngine()
    engine.config["remove_outliers"] = False
    engine.config["min_periods_ratio"] = 0.0

    features = engine.generate_trading_features(frame)

    for window in [20, 50, 100]:
        assert f"sortino_{window}" in features.columns
        assert f"calmar_{window}" in features.columns
