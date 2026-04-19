import pytest
from backtest.reporter import Reporter
from backtest.models import Trade


@pytest.fixture
def sample_result():
    trades = [
        Trade(id="t1", order_id="o1", symbol="BTCUSDT", side="buy",
              price=42000.0, quantity=1000.0, pnl=0.0, commission=0.4, timestamp=1704067200000),
        Trade(id="t2", order_id="o2", symbol="BTCUSDT", side="sell",
              price=42600.0, quantity=1000.0, pnl=14.29, commission=0.4, timestamp=1704153600000),
        Trade(id="t3", order_id="o3", symbol="BTCUSDT", side="sell",
              price=42800.0, quantity=1000.0, pnl=0.0, commission=0.4, timestamp=1704240000000),
        Trade(id="t4", order_id="o4", symbol="BTCUSDT", side="buy",
              price=43200.0, quantity=1000.0, pnl=-9.35, commission=0.4, timestamp=1704326400000),
    ]
    equity_curve = [
        (1704067200000, 10000.0), (1704153600000, 10013.49),
        (1704240000000, 10013.49), (1704326400000, 10002.54),
        (1704412800000, 10020.0),
    ]
    return {
        "trades": trades, "trades_count": 4,
        "equity_curve": equity_curve, "final_equity": 10020.0,
        "initial_balance": 10000.0,
    }


def test_reporter_basic_metrics(sample_result):
    report = Reporter.generate(sample_result)
    assert report["net_return"] == pytest.approx(0.002, abs=0.001)
    assert report["total_trades"] == 4
    assert report["total_commission"] == pytest.approx(1.6)
    assert "max_drawdown" in report
    assert "sharpe_ratio" in report
    assert "win_rate" in report
    assert "profit_factor" in report


def test_reporter_win_rate(sample_result):
    report = Reporter.generate(sample_result)
    assert report["win_rate"] == pytest.approx(0.5)


def test_reporter_max_drawdown(sample_result):
    report = Reporter.generate(sample_result)
    expected_dd = (10013.49 - 10002.54) / 10013.49
    assert report["max_drawdown"] == pytest.approx(expected_dd, rel=0.01)
