import sqlite3
import pytest
from backtest.engine import BacktestEngine
from backtest.strategy import BaseStrategy
from backtest.models import Bar


class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._entered = False

    def on_bar(self, bar: Bar):
        if not self._entered:
            self.buy(1000.0)
            self._entered = True


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE klines (
            symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, exchange TEXT,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    rows = [
        ("BTCUSDT", "1h", 1704067200000 + i * 3600000,
         42000 + i * 100, 42000 + i * 100 + 200,
         42000 + i * 100 - 100, 42000 + i * 100 + 50,
         1000, "binance")
        for i in range(10)
    ]
    conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(path)


def test_engine_runs_and_produces_result(db_path):
    engine = BacktestEngine(
        db_path=db_path, symbol="BTCUSDT", interval="1h", exchange="binance",
        strategy_class=BuyAndHoldStrategy, balance=10000.0, leverage=10,
        commission_rate=0.0004, funding_rate=0.0001, maintenance_margin=0.005,
    )
    result = engine.run()
    assert result["trades_count"] > 0
    assert len(result["equity_curve"]) == 10
    assert result["final_equity"] > 0


def test_engine_with_time_range(db_path):
    engine = BacktestEngine(
        db_path=db_path, symbol="BTCUSDT", interval="1h", exchange="binance",
        strategy_class=BuyAndHoldStrategy, balance=10000.0, leverage=10,
        start="2024-01-01 01:00:00", end="2024-01-01 05:00:00",
    )
    result = engine.run()
    assert len(result["equity_curve"]) == 5
