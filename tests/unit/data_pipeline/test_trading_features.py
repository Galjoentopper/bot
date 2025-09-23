import numpy as np
import pandas as pd

from src.data_pipeline.trading_features import TradingFeatureEngine


def test_add_risk_features_uses_defined_mean_return():
    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="30min")
    base = np.linspace(100.0, 110.0, periods)
    prices = base + np.sin(np.linspace(0, 6, periods))

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(periods, 1000.0),
        },
        index=index,
    )

    engine = TradingFeatureEngine(
        config={
            "sharpe_windows": [20],
            "max_drawdown_windows": [20],
            "var_windows": [20],
            "var_confidence": 0.1,
        }
    )

    result = engine._add_risk_features(df.copy())

    assert "sortino_20" in result.columns
    assert "calmar_20" in result.columns
    assert result["sortino_20"].notna().sum() > 0
    assert result["calmar_20"].notna().sum() > 0


def test_generate_trading_features_converts_timestamp_column():
    periods = 256
    rng = pd.Index(range(periods))
    dt_index = pd.date_range("2024-01-01", periods=periods, freq="30min")
    np.random.seed(0)
    prices = 100 + np.cumsum(np.random.normal(scale=0.5, size=periods))

    df = pd.DataFrame(
        {
            "timestamp": (dt_index.view("int64") // 10**9),
            "open": prices * np.random.uniform(0.999, 1.001, size=periods),
            "high": prices * np.random.uniform(1.000, 1.005, size=periods),
            "low": prices * np.random.uniform(0.995, 1.000, size=periods),
            "close": prices,
            "volume": np.random.uniform(1_000, 5_000, size=periods),
        },
        index=rng,
    )

    engine = TradingFeatureEngine()
    engine.config["remove_outliers"] = False
    engine.config["min_periods_ratio"] = 0.0

    features = engine.generate_trading_features(df)

    assert isinstance(features.index, pd.DatetimeIndex)
    assert features.index[0] == dt_index[0]
