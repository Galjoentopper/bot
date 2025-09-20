import pytest

from src.trading.profit_optimizer import ProfitOptimizer


def test_rebalancing_ignores_positions_within_buffer() -> None:
    optimizer = ProfitOptimizer(
        {
            "max_position_pct": 0.15,
            "rebalance_buffer_pct": 0.05,
            "rebalance_target_buffer_pct": 0.01,
            "min_rebalance_notional": 50.0,
        }
    )

    signals = optimizer.generate_rebalancing_signals(
        current_positions={"ETHEUR": 0.5},
        current_prices={"ETHEUR": 3_000.0},
        current_balance=8_500.0,
    )

    assert signals == {}


def test_rebalancing_triggers_when_above_buffer() -> None:
    optimizer = ProfitOptimizer(
        {
            "max_position_pct": 0.15,
            "rebalance_buffer_pct": 0.05,
            "rebalance_target_buffer_pct": 0.01,
            "min_rebalance_notional": 50.0,
        }
    )

    signals = optimizer.generate_rebalancing_signals(
        current_positions={"ETHEUR": 2.0},
        current_prices={"ETHEUR": 3_000.0},
        current_balance=5_000.0,
    )

    assert "ETHEUR" in signals
    signal = signals["ETHEUR"]
    assert signal.action == "SELL"
    assert signal.quantity_pct == pytest.approx(0.5)  # capped at 50%


def test_rebalancing_skips_when_below_min_notional() -> None:
    optimizer = ProfitOptimizer(
        {
            "max_position_pct": 0.15,
            "rebalance_buffer_pct": 0.05,
            "rebalance_target_buffer_pct": 0.01,
            "min_rebalance_notional": 500.0,
        }
    )

    signals = optimizer.generate_rebalancing_signals(
        current_positions={"ETHEUR": 0.25},
        current_prices={"ETHEUR": 3_000.0},
        current_balance=8_500.0,
    )

    assert signals == {}
