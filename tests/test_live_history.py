# tests/test_live_history.py
import json
import sqlite3
import tempfile
import os
import pytest
from backtest.live_history import LiveHistoryDB


@pytest.fixture
def db(tmp_path):
    return LiveHistoryDB(str(tmp_path / "history.db"))


def test_tables_created_on_init(db, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "history.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"state_snapshots", "orders", "positions", "trades"} <= tables


def test_record_state_snapshot(db, tmp_path):
    db.record_state_snapshot("acc1", "binance", "MyStrat", "BTCUSDT", "15m", {"x": 1}, 1000)
    conn = sqlite3.connect(str(tmp_path / "history.db"))
    rows = conn.execute("SELECT account_id, exchange, strategy, symbol, interval, state_json, ts FROM state_snapshots").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "acc1"
    assert json.loads(rows[0][5]) == {"x": 1}
    assert rows[0][6] == 1000


def test_record_state_snapshot_appends(db):
    db.record_state_snapshot("acc1", "binance", "S", "BTCUSDT", "1h", {}, 1000)
    db.record_state_snapshot("acc1", "binance", "S", "BTCUSDT", "1h", {}, 2000)
    conn = sqlite3.connect(db._db_path)
    assert conn.execute("SELECT COUNT(*) FROM state_snapshots").fetchone()[0] == 2


def test_record_position_with_open_position(db):
    from backtest.models import Position
    pos = Position("BTCUSDT", "long", 5000.0, 50000.0, 10, 200.0, 500.0)
    db.record_position("acc1", "binance", "BTCUSDT", pos, 1000.0, 1700.0, 9000)
    conn = sqlite3.connect(db._db_path)
    row = conn.execute("SELECT side, quantity, entry_price, balance, equity, ts FROM positions").fetchone()
    assert row[0] == "long"
    assert row[1] == 5000.0
    assert row[3] == 1000.0
    assert row[5] == 9000


def test_record_position_flat(db):
    db.record_position("acc1", "binance", "BTCUSDT", None, 1000.0, 1000.0, 9000)
    conn = sqlite3.connect(db._db_path)
    row = conn.execute("SELECT side, quantity FROM positions").fetchone()
    assert row[0] is None
    assert row[1] == 0.0


def test_upsert_order_insert(db):
    order = {
        "order_id": "ORD001", "symbol": "BTCUSDT", "side": "buy", "type": "market",
        "quantity": 1000.0, "price": None, "status": "pending",
        "filled_price": None, "filled_qty": None, "commission": None,
        "ts": 5000, "filled_at": None,
    }
    db.upsert_order("acc1", "binance", order)
    conn = sqlite3.connect(db._db_path)
    row = conn.execute("SELECT order_id, status FROM orders").fetchone()
    assert row[0] == "ORD001"
    assert row[1] == "pending"


def test_upsert_order_updates_on_fill(db):
    order = {
        "order_id": "ORD002", "symbol": "BTCUSDT", "side": "buy", "type": "market",
        "quantity": 1000.0, "price": None, "status": "pending",
        "filled_price": None, "filled_qty": None, "commission": None,
        "ts": 5000, "filled_at": None,
    }
    db.upsert_order("acc1", "binance", order)
    filled = {**order, "status": "filled", "filled_price": 50000.0, "filled_qty": 0.02, "commission": 0.4, "filled_at": 5100}
    db.upsert_order("acc1", "binance", filled)
    conn = sqlite3.connect(db._db_path)
    rows = conn.execute("SELECT order_id, status, filled_price FROM orders").fetchall()
    assert len(rows) == 1  # no duplicate
    assert rows[0][1] == "filled"
    assert rows[0][2] == 50000.0


def test_upsert_order_different_accounts_isolated(db):
    order = {
        "order_id": "ORD003", "symbol": "BTCUSDT", "side": "buy", "type": "market",
        "quantity": 100.0, "price": None, "status": "pending",
        "filled_price": None, "filled_qty": None, "commission": None,
        "ts": 1000, "filled_at": None,
    }
    db.upsert_order("acc1", "binance", order)
    db.upsert_order("acc2", "binance", order)  # same order_id, different account
    conn = sqlite3.connect(db._db_path)
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 2


def test_upsert_trades_bulk_dedup(db):
    trades = [
        {"trade_id": "T1", "order_id": "O1", "symbol": "BTCUSDT", "side": "buy",
         "price": 50000.0, "qty": 0.02, "notional": 1000.0, "commission": 0.4,
         "realized_pnl": 0.0, "ts": 6000},
        {"trade_id": "T2", "order_id": "O1", "symbol": "BTCUSDT", "side": "buy",
         "price": 50001.0, "qty": 0.01, "notional": 500.0, "commission": 0.2,
         "realized_pnl": 0.0, "ts": 6001},
    ]
    db.upsert_trades("acc1", "binance", trades)
    db.upsert_trades("acc1", "binance", trades)  # duplicate insert
    conn = sqlite3.connect(db._db_path)
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 2


def test_latest_ts_empty_returns_zero(db):
    assert db.latest_ts("acc1", "binance", "BTCUSDT", "orders") == 0


def test_latest_ts_returns_max(db):
    for ts in [1000, 5000, 3000]:
        order = {
            "order_id": f"O{ts}", "symbol": "BTCUSDT", "side": "buy", "type": "market",
            "quantity": 100.0, "price": None, "status": "filled",
            "filled_price": 50000.0, "filled_qty": 0.002, "commission": 0.04,
            "ts": ts, "filled_at": ts + 100,
        }
        db.upsert_order("acc1", "binance", order)
    assert db.latest_ts("acc1", "binance", "BTCUSDT", "orders") == 5000


def test_latest_ts_invalid_table_raises(db):
    with pytest.raises(ValueError):
        db.latest_ts("acc1", "binance", "BTCUSDT", "evil_table; DROP TABLE orders--")
