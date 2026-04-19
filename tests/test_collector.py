import sqlite3
import pytest
from backtest.collector.binance import BinanceCollector
from backtest.models import Bar


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


class TestBaseCollector:
    def test_init_db_creates_table(self, db_path):
        collector = BinanceCollector(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor]
        assert "klines" in tables
        conn.close()

    def test_save_bars(self, db_path):
        collector = BinanceCollector(db_path)
        bars = [
            Bar("BTCUSDT", 1704067200000, 42000, 42500, 41800, 42300, 1500, "1h"),
            Bar("BTCUSDT", 1704070800000, 42300, 42800, 42100, 42600, 1200, "1h"),
        ]
        collector._save_bars(bars)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT count(*) FROM klines").fetchone()[0]
        assert count == 2
        conn.close()

    def test_get_latest_timestamp(self, db_path):
        collector = BinanceCollector(db_path)
        bars = [
            Bar("BTCUSDT", 1704067200000, 42000, 42500, 41800, 42300, 1500, "1h"),
            Bar("BTCUSDT", 1704070800000, 42300, 42800, 42100, 42600, 1200, "1h"),
        ]
        collector._save_bars(bars)
        ts = collector._get_latest_timestamp("BTCUSDT", "1h")
        assert ts == 1704070800000

    def test_get_latest_timestamp_empty(self, db_path):
        collector = BinanceCollector(db_path)
        ts = collector._get_latest_timestamp("BTCUSDT", "1h")
        assert ts is None
