import sqlite3
import pytest
from backtest.data_feed import DataFeed
from backtest.models import Bar


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test_klines.db"
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
        ("BTCUSDT", "1h", 1704067200000, 42000, 42500, 41800, 42300, 1500, "binance"),
        ("BTCUSDT", "1h", 1704070800000, 42300, 42800, 42100, 42600, 1200, "binance"),
        ("BTCUSDT", "1h", 1704074400000, 42600, 42900, 42400, 42700, 1100, "binance"),
    ]
    conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(path)


def test_iterate_all_bars(db_path):
    feed = DataFeed(db_path=db_path, symbol="BTCUSDT", interval="1h", exchange="binance")
    bars = list(feed)
    assert len(bars) == 3
    assert all(isinstance(b, Bar) for b in bars)
    assert bars[0].timestamp < bars[1].timestamp < bars[2].timestamp


def test_filter_by_time_range(db_path):
    feed = DataFeed(db_path=db_path, symbol="BTCUSDT", interval="1h", exchange="binance",
                    start_ts=1704070800000, end_ts=1704070800000)
    bars = list(feed)
    assert len(bars) == 1
    assert bars[0].timestamp == 1704070800000


def test_empty_result(db_path):
    feed = DataFeed(db_path=db_path, symbol="ETHUSDT", interval="1h", exchange="binance")
    bars = list(feed)
    assert len(bars) == 0
