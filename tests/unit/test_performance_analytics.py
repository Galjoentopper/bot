import pytest

from src.trading.performance_analytics import PerformanceAnalyzer


def _record_snapshot(
    analyzer: PerformanceAnalyzer, portfolio_value: float, cash_balance: float
) -> None:
    analyzer.record_portfolio_snapshot(
        {
            "portfolio_value": portfolio_value,
            "cash_balance": cash_balance,
        }
    )


def test_performance_analyzer_unrealized_pnl_only() -> None:
    analyzer = PerformanceAnalyzer({}, initial_capital=10_000.0)

    analyzer.record_trade(
        {
            "symbol": "ETHEUR",
            "action": "BUY",
            "quantity": 1.0,
            "price": 3_000.0,
            "fee": 0.0,
            "cost": 3_000.0,
            "reasoning": "test",
            "confidence": 0.0,
            "timestamp": 0.0,
        }
    )

    _record_snapshot(analyzer, portfolio_value=10_000.0, cash_balance=10_000.0)
    _record_snapshot(analyzer, portfolio_value=10_200.0, cash_balance=7_000.0)

    metrics = analyzer.calculate_comprehensive_metrics(
        current_positions={"ETHEUR": 1.0},
        current_prices={"ETHEUR": 3_200.0},
        current_balance=7_000.0,
    )

    assert metrics.total_pnl == pytest.approx(200.0)
    assert metrics.realized_pnl == pytest.approx(0.0)
    assert metrics.unrealized_pnl == pytest.approx(200.0)


def test_performance_analyzer_realized_and_unrealized_split() -> None:
    analyzer = PerformanceAnalyzer({}, initial_capital=10_000.0)

    analyzer.record_trade(
        {
            "symbol": "ETHEUR",
            "action": "BUY",
            "quantity": 1.0,
            "price": 3_000.0,
            "fee": 0.0,
            "cost": 3_000.0,
            "reasoning": "test",
            "confidence": 0.0,
            "timestamp": 0.0,
        }
    )

    analyzer.record_trade(
        {
            "symbol": "ETHEUR",
            "action": "SELL",
            "quantity": 0.5,
            "price": 3_500.0,
            "fee": 0.0,
            "proceeds": 1_750.0,
            "realized_pnl": 250.0,
            "reasoning": "test",
            "confidence": 0.0,
            "timestamp": 1.0,
        }
    )

    _record_snapshot(analyzer, portfolio_value=10_000.0, cash_balance=10_000.0)
    _record_snapshot(analyzer, portfolio_value=10_350.0, cash_balance=8_750.0)

    metrics = analyzer.calculate_comprehensive_metrics(
        current_positions={"ETHEUR": 0.5},
        current_prices={"ETHEUR": 3_200.0},
        current_balance=8_750.0,
    )

    assert metrics.total_pnl == pytest.approx(350.0)
    assert metrics.realized_pnl == pytest.approx(250.0)
    assert metrics.unrealized_pnl == pytest.approx(100.0)
