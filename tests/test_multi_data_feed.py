import sqlite3
import pytest
from backtest.multi_data_feed import MultiSymbolDataFeed


@pytest.fixture
def two_sym_db(tmp_path):
    db = str(tmp_path / "klines.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    rows = [
        # ts=1000: both symbols present
        ("binance", "AAAUSDT", "1h", 1000, 100.0, 105.0, 99.0, 102.0, 1000.0),
        ("binance", "BBBUSDT", "1h", 1000, 200.0, 205.0, 199.0, 203.0, 2000.0),
        # ts=2000: only AAAUSDT (BBBUSDT missing — gap)
        ("binance", "AAAUSDT", "1h", 2000, 102.0, 108.0, 101.0, 106.0, 1100.0),
        # ts=3000: both symbols present
        ("binance", "AAAUSDT", "1h", 3000, 106.0, 110.0, 105.0, 109.0, 1200.0),
        ("binance", "BBBUSDT", "1h", 3000, 203.0, 208.0, 202.0, 206.0, 2100.0),
    ]
    conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


def test_yields_three_timestamps(two_sym_db):
    feed = MultiSymbolDataFeed(two_sym_db, ["AAAUSDT", "BBBUSDT"], "1h", "binance")
    results = list(feed)
    assert len(results) == 3


def test_timestamps_are_sorted(two_sym_db):
    feed = MultiSymbolDataFeed(two_sym_db, ["AAAUSDT", "BBBUSDT"], "1h", "binance")
    tss = [ts for ts, _ in feed]
    assert tss == [1000, 2000, 3000]


def test_both_symbols_at_ts1000(two_sym_db):
    feed = MultiSymbolDataFeed(two_sym_db, ["AAAUSDT", "BBBUSDT"], "1h", "binance")
    ts, bars = next(iter(feed))
    assert ts == 1000
    assert set(bars.keys()) == {"AAAUSDT", "BBBUSDT"}
    assert bars["AAAUSDT"].close == 102.0
    assert bars["BBBUSDT"].close == 203.0


def test_only_one_symbol_at_ts2000(two_sym_db):
    feed = MultiSymbolDataFeed(two_sym_db, ["AAAUSDT", "BBBUSDT"], "1h", "binance")
    results = list(feed)
    ts, bars = results[1]
    assert ts == 2000
    assert set(bars.keys()) == {"AAAUSDT"}


def test_start_end_filter(two_sym_db):
    feed = MultiSymbolDataFeed(
        two_sym_db, ["AAAUSDT", "BBBUSDT"], "1h", "binance",
        start_ts=2000, end_ts=2000,
    )
    results = list(feed)
    assert len(results) == 1
    assert results[0][0] == 2000


def test_bar_fields_are_correct(two_sym_db):
    feed = MultiSymbolDataFeed(two_sym_db, ["AAAUSDT", "BBBUSDT"], "1h", "binance")
    _, bars = next(iter(feed))
    bar = bars["AAAUSDT"]
    assert bar.symbol == "AAAUSDT"
    assert bar.interval == "1h"
    assert bar.open == 100.0
    assert bar.high == 105.0
    assert bar.low == 99.0
    assert bar.volume == 1000.0
