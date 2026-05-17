# Live Trading History & Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record live trading state snapshots, orders, positions, and trades to SQLite for multi-account multi-exchange replay, with a background sync thread that pulls history from the exchange API.

**Architecture:** New `live_history.py` (append-only DB) and `live_connector.py` (abstract exchange API) sit alongside the existing live modules. `LiveExchange` and `LiveFeed` are refactored to take a `BaseExchangeConnector` instead of a raw `UMFutures` client. `LiveEngine` gains a background daemon thread that periodically syncs order/trade history from the exchange.

**Tech Stack:** Python 3.11+, SQLite (stdlib `sqlite3`), `threading.Event`, `abc.ABC`, `binance-futures-connector`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/backtest/live_history.py` | **Create** | `LiveHistoryDB` — all writes to `live_history.db` |
| `src/backtest/live_connector.py` | **Create** | `BaseExchangeConnector` ABC + `BinanceConnector` |
| `tests/test_live_history.py` | **Create** | Tests for `LiveHistoryDB` |
| `tests/test_live_connector.py` | **Create** | Tests for `BinanceConnector` normalization |
| `src/backtest/live_exchange.py` | **Modify** | Take connector + history_db; remove raw UMFutures |
| `src/backtest/live_feed.py` | **Modify** | Take connector instead of UMFutures client |
| `tests/test_live_exchange.py` | **Modify** | Update mocks to use connector interface |
| `tests/test_live_feed.py` | **Modify** | Rename `client` → `connector` in tests |
| `src/backtest/live_engine.py` | **Modify** | Use connector; add history writes; bg sync thread |
| `tests/test_live_engine.py` | **Modify** | Update mocks to connector interface |
| `src/backtest/__main__.py` | **Modify** | Add `--exchange`, `--sync-interval`, `--history-db` |

---

## Task 1: LiveHistoryDB

**Files:**
- Create: `src/backtest/live_history.py`
- Create: `tests/test_live_history.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/airwolf/Github/crypto_backtest
pytest tests/test_live_history.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'backtest.live_history'`

- [ ] **Step 3: Implement LiveHistoryDB**

```python
# src/backtest/live_history.py
import json
import sqlite3
import threading
from pathlib import Path

from backtest.models import Position


class LiveHistoryDB:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id  TEXT    NOT NULL,
                    exchange    TEXT    NOT NULL,
                    strategy    TEXT    NOT NULL,
                    symbol      TEXT    NOT NULL,
                    interval    TEXT    NOT NULL,
                    state_json  TEXT    NOT NULL,
                    ts          INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id   TEXT    NOT NULL,
                    exchange     TEXT    NOT NULL,
                    symbol       TEXT    NOT NULL,
                    order_id     TEXT    NOT NULL,
                    side         TEXT    NOT NULL,
                    type         TEXT    NOT NULL,
                    quantity     REAL    NOT NULL,
                    price        REAL,
                    status       TEXT    NOT NULL,
                    filled_price REAL,
                    filled_qty   REAL,
                    commission   REAL,
                    ts           INTEGER NOT NULL,
                    filled_at    INTEGER,
                    UNIQUE (exchange, account_id, order_id)
                );
                CREATE TABLE IF NOT EXISTS positions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id     TEXT    NOT NULL,
                    exchange       TEXT    NOT NULL,
                    symbol         TEXT    NOT NULL,
                    side           TEXT,
                    quantity       REAL    NOT NULL,
                    entry_price    REAL,
                    leverage       INTEGER,
                    unrealized_pnl REAL,
                    margin         REAL,
                    balance        REAL    NOT NULL,
                    equity         REAL    NOT NULL,
                    ts             INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id   TEXT    NOT NULL,
                    exchange     TEXT    NOT NULL,
                    symbol       TEXT    NOT NULL,
                    trade_id     TEXT    NOT NULL,
                    order_id     TEXT    NOT NULL,
                    side         TEXT    NOT NULL,
                    price        REAL    NOT NULL,
                    qty          REAL    NOT NULL,
                    notional     REAL    NOT NULL,
                    commission   REAL    NOT NULL,
                    realized_pnl REAL,
                    ts           INTEGER NOT NULL,
                    UNIQUE (exchange, account_id, trade_id)
                );
            """)

    def record_state_snapshot(self, account_id: str, exchange: str, strategy: str,
                               symbol: str, interval: str, state: dict, ts: int) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO state_snapshots "
                "(account_id, exchange, strategy, symbol, interval, state_json, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (account_id, exchange, strategy, symbol, interval, json.dumps(state), ts),
            )

    def record_position(self, account_id: str, exchange: str, symbol: str,
                        position: Position | None, balance: float, equity: float, ts: int) -> None:
        if position is not None:
            row = (account_id, exchange, symbol, position.side, position.quantity,
                   position.entry_price, position.leverage, position.unrealized_pnl,
                   position.margin, balance, equity, ts)
        else:
            row = (account_id, exchange, symbol, None, 0.0, None, None, None, None, balance, equity, ts)
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO positions "
                "(account_id, exchange, symbol, side, quantity, entry_price, leverage, "
                "unrealized_pnl, margin, balance, equity, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )

    def upsert_order(self, account_id: str, exchange: str, order: dict) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO orders "
                "(account_id, exchange, symbol, order_id, side, type, quantity, price, "
                "status, filled_price, filled_qty, commission, ts, filled_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (account_id, exchange, order["symbol"], order["order_id"],
                 order["side"], order["type"], order["quantity"], order.get("price"),
                 order["status"], order.get("filled_price"), order.get("filled_qty"),
                 order.get("commission"), order["ts"], order.get("filled_at")),
            )
            conn.execute(
                "UPDATE orders SET status=?, filled_price=?, filled_qty=?, "
                "commission=?, filled_at=? "
                "WHERE exchange=? AND account_id=? AND order_id=? AND status != 'filled'",
                (order["status"], order.get("filled_price"), order.get("filled_qty"),
                 order.get("commission"), order.get("filled_at"),
                 exchange, account_id, order["order_id"]),
            )

    def upsert_trades(self, account_id: str, exchange: str, trades: list[dict]) -> None:
        if not trades:
            return
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO trades "
                "(account_id, exchange, symbol, trade_id, order_id, side, price, "
                "qty, notional, commission, realized_pnl, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(account_id, exchange, t["symbol"], t["trade_id"], t["order_id"],
                  t["side"], t["price"], t["qty"], t["notional"],
                  t["commission"], t.get("realized_pnl"), t["ts"]) for t in trades],
            )

    def latest_ts(self, account_id: str, exchange: str, symbol: str, table: str) -> int:
        if table not in {"orders", "trades", "positions", "state_snapshots"}:
            raise ValueError(f"Invalid table: {table!r}")
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT MAX(ts) FROM {table} WHERE account_id=? AND exchange=? AND symbol=?",
                (account_id, exchange, symbol),
            ).fetchone()
        return row[0] if row and row[0] is not None else 0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_live_history.py -v
```
Expected: all 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/live_history.py tests/test_live_history.py
git commit -m "feat: add LiveHistoryDB with state/order/position/trade tables"
```

---

## Task 2: BaseExchangeConnector + BinanceConnector

**Files:**
- Create: `src/backtest/live_connector.py`
- Create: `tests/test_live_connector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_live_connector.py
import pytest
from unittest.mock import MagicMock, patch


def _make_binance_client():
    client = MagicMock()
    client.time.return_value = {"serverTime": 1_000_000_000_000}
    return client


class TestBinanceConnectorNormalization:
    def test_normalize_order_filled(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.020",
            "executedQty": "0.020",
            "avgPrice": "50000.0",
            "price": "0",
            "time": 1_700_000_000_000,
            "updateTime": 1_700_000_001_000,
        }
        result = BinanceConnector._normalize_order(raw)
        assert result["order_id"] == "12345"
        assert result["symbol"] == "BTCUSDT"
        assert result["status"] == "filled"
        assert result["side"] == "buy"
        assert result["type"] == "market"
        assert result["filled_price"] == 50000.0
        assert result["ts"] == 1_700_000_000_000
        assert result["filled_at"] == 1_700_000_001_000

    def test_normalize_order_pending(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "orderId": 99,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "0.01",
            "executedQty": "0",
            "avgPrice": "0",
            "price": "52000",
            "time": 1_700_000_000_000,
            "updateTime": 1_700_000_000_000,
        }
        result = BinanceConnector._normalize_order(raw)
        assert result["status"] == "new"
        assert result["filled_price"] is None
        assert result["price"] == 52000.0

    def test_normalize_trade(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "id": 555,
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "buyer": False,
            "price": "50000.0",
            "qty": "0.020",
            "quoteQty": "1000.0",
            "commission": "0.4",
            "realizedPnl": "50.0",
            "time": 1_700_000_001_000,
        }
        result = BinanceConnector._normalize_trade(raw)
        assert result["trade_id"] == "555"
        assert result["order_id"] == "12345"
        assert result["side"] == "sell"
        assert result["price"] == 50000.0
        assert result["qty"] == 0.020
        assert result["notional"] == 1000.0
        assert result["commission"] == 0.4
        assert result["realized_pnl"] == 50.0
        assert result["ts"] == 1_700_000_001_000

    def test_normalize_trade_buyer_side(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "id": 556, "orderId": 1, "symbol": "BTCUSDT", "buyer": True,
            "price": "50000", "qty": "0.01", "quoteQty": "500",
            "commission": "0.2", "realizedPnl": "0", "time": 1000,
        }
        result = BinanceConnector._normalize_trade(raw)
        assert result["side"] == "buy"


class TestBinanceConnectorInterface:
    def _make_connector(self):
        with patch("backtest.live_connector.UMFutures") as MockUMFutures:
            mock_client = _make_binance_client()
            mock_client.exchange_info.return_value = {
                "symbols": [{"symbol": "BTCUSDT", "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"}
                ]}]
            }
            MockUMFutures.return_value = mock_client
            from backtest.live_connector import BinanceConnector
            connector = BinanceConnector(api_key="key", secret="secret", testnet=True)
            connector._client = mock_client
            return connector, mock_client

    def test_exchange_name(self):
        connector, _ = self._make_connector()
        assert connector.exchange_name == "binance"

    def test_exchange_info_returns_symbol_dict(self):
        connector, client = self._make_connector()
        client.exchange_info.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}]}]
        }
        result = connector.exchange_info("BTCUSDT")
        assert result["symbol"] == "BTCUSDT"

    def test_fetch_balance(self):
        connector, client = self._make_connector()
        client.balance.return_value = [
            {"asset": "BNB", "availableBalance": "1.0"},
            {"asset": "USDT", "availableBalance": "1234.56"},
        ]
        assert connector.fetch_balance() == 1234.56

    def test_fetch_position_flat_returns_none(self):
        connector, client = self._make_connector()
        client.get_position_risk.return_value = [{"positionAmt": "0", "entryPrice": "0", "unrealizedProfit": "0"}]
        assert connector.fetch_position("BTCUSDT") is None

    def test_fetch_position_long(self):
        connector, client = self._make_connector()
        client.get_position_risk.return_value = [
            {"positionAmt": "0.1", "entryPrice": "50000", "unrealizedProfit": "200"}
        ]
        pos = connector.fetch_position("BTCUSDT")
        assert pos is not None
        assert pos["side"] == "long"
        assert pos["qty"] == 0.1
        assert pos["entry_price"] == 50000.0

    def test_fetch_orders_passes_since_ms(self):
        connector, client = self._make_connector()
        client.get_all_orders.return_value = []
        connector.fetch_orders("BTCUSDT", since_ms=1_700_000_000_000)
        client.get_all_orders.assert_called_once_with(symbol="BTCUSDT", startTime=1_700_000_000_000)

    def test_fetch_trades_no_since_ms(self):
        connector, client = self._make_connector()
        client.get_account_trades.return_value = []
        connector.fetch_trades("BTCUSDT")
        client.get_account_trades.assert_called_once_with(symbol="BTCUSDT")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_live_connector.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'backtest.live_connector'`

- [ ] **Step 3: Implement live_connector.py**

```python
# src/backtest/live_connector.py
import time
from abc import ABC, abstractmethod

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

_TESTNET_URL = "https://testnet.binancefuture.com"
_MAINNET_URL = "https://fapi.binance.com"


def _retry(fn, attempts: int = 3, backoff: float = 2.0):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff * (2 ** i))
    raise last_exc


class BaseExchangeConnector(ABC):
    @property
    @abstractmethod
    def exchange_name(self) -> str: ...

    @abstractmethod
    def server_time(self) -> int: ...

    @abstractmethod
    def exchange_info(self, symbol: str) -> dict: ...

    @abstractmethod
    def klines(self, symbol: str, interval: str, **kwargs) -> list: ...

    @abstractmethod
    def fetch_balance(self) -> float: ...

    @abstractmethod
    def fetch_position(self, symbol: str) -> dict | None:
        """Returns {"side", "qty", "entry_price", "unrealized_pnl"} or None if flat."""
        ...

    @abstractmethod
    def fetch_mark_price(self, symbol: str) -> float: ...

    @abstractmethod
    def fetch_orders(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        """Returns list of normalized order dicts."""
        ...

    @abstractmethod
    def fetch_trades(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        """Returns list of normalized trade dicts."""
        ...

    @abstractmethod
    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None) -> dict:
        """Returns raw exchange response dict."""
        ...

    @abstractmethod
    def query_order(self, symbol: str, order_id: str) -> dict:
        """Returns normalized order dict with current status."""
        ...

    @abstractmethod
    def change_leverage(self, symbol: str, leverage: int) -> None: ...


class BinanceConnector(BaseExchangeConnector):
    def __init__(self, api_key: str = "", secret: str = "", testnet: bool = True):
        base_url = _TESTNET_URL if testnet else _MAINNET_URL
        probe = UMFutures(base_url=base_url)
        server_ms = probe.time()["serverTime"]
        local_ms = int(time.time() * 1000)
        offset_ms = server_ms - local_ms
        if abs(offset_ms) > 500:
            import binance.api as _binance_api
            print(f"[INFO] Clock skew detected: {offset_ms:+d}ms, applying correction")
            _binance_api.get_timestamp = lambda: int(time.time() * 1000) + offset_ms
        self._client = UMFutures(key=api_key, secret=secret, base_url=base_url)

    @property
    def exchange_name(self) -> str:
        return "binance"

    def server_time(self) -> int:
        return self._client.time()["serverTime"]

    def exchange_info(self, symbol: str) -> dict:
        info = _retry(lambda: self._client.exchange_info())
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return s
        raise ValueError(f"Symbol {symbol!r} not found in exchange_info")

    def klines(self, symbol: str, interval: str, **kwargs) -> list:
        return _retry(lambda: self._client.klines(symbol=symbol, interval=interval, **kwargs))

    def fetch_balance(self) -> float:
        balances = _retry(lambda: self._client.balance())
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0

    def fetch_position(self, symbol: str) -> dict | None:
        positions = _retry(lambda: self._client.get_position_risk(symbol=symbol))
        for p in positions:
            qty = float(p["positionAmt"])
            if abs(qty) < 1e-8:
                continue
            return {
                "side": "long" if qty > 0 else "short",
                "qty": abs(qty),
                "entry_price": float(p["entryPrice"]),
                "unrealized_pnl": float(p["unrealizedProfit"]),
            }
        return None

    def fetch_mark_price(self, symbol: str) -> float:
        result = _retry(lambda: self._client.mark_price(symbol=symbol))
        return float(result["markPrice"])

    def fetch_orders(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        kwargs: dict = {"symbol": symbol}
        if since_ms:
            kwargs["startTime"] = since_ms
        raw = _retry(lambda: self._client.get_all_orders(**kwargs))
        return [self._normalize_order(o) for o in raw]

    def fetch_trades(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        kwargs: dict = {"symbol": symbol}
        if since_ms:
            kwargs["startTime"] = since_ms
        raw = _retry(lambda: self._client.get_account_trades(**kwargs))
        return [self._normalize_trade(t) for t in raw]

    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None) -> dict:
        params: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET" if type_ == "market" else "LIMIT",
            "quantity": quantity,
        }
        if type_ == "limit" and price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"
        return _retry(lambda: self._client.new_order(**params))

    def query_order(self, symbol: str, order_id: str) -> dict:
        resp = _retry(lambda: self._client.query_order(symbol=symbol, orderId=int(order_id)))
        return self._normalize_order(resp)

    def change_leverage(self, symbol: str, leverage: int) -> None:
        _retry(lambda: self._client.change_leverage(symbol=symbol, leverage=leverage))

    @staticmethod
    def _normalize_order(o: dict) -> dict:
        avg = float(o.get("avgPrice", 0))
        status = o["status"].lower()
        return {
            "order_id": str(o["orderId"]),
            "symbol": o["symbol"],
            "side": o["side"].lower(),
            "type": "market" if o["type"] == "MARKET" else "limit",
            "quantity": float(o["origQty"]),
            "price": float(o["price"]) if float(o["price"]) > 0 else None,
            "status": status,
            "filled_price": avg if avg > 0 else None,
            "filled_qty": float(o["executedQty"]),
            "commission": None,
            "ts": int(o["time"]),
            "filled_at": int(o["updateTime"]) if status == "filled" else None,
        }

    @staticmethod
    def _normalize_trade(t: dict) -> dict:
        return {
            "trade_id": str(t["id"]),
            "order_id": str(t["orderId"]),
            "symbol": t["symbol"],
            "side": "buy" if t["buyer"] else "sell",
            "price": float(t["price"]),
            "qty": float(t["qty"]),
            "notional": float(t["quoteQty"]),
            "commission": float(t["commission"]),
            "realized_pnl": float(t.get("realizedPnl", 0)),
            "ts": int(t["time"]),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_live_connector.py -v
```
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/live_connector.py tests/test_live_connector.py
git commit -m "feat: add BaseExchangeConnector ABC and BinanceConnector"
```

---

## Task 3: Refactor LiveExchange

**Files:**
- Modify: `src/backtest/live_exchange.py`
- Modify: `tests/test_live_exchange.py`

- [ ] **Step 1: Rewrite test file with connector interface**

Replace entire `tests/test_live_exchange.py`:

```python
# tests/test_live_exchange.py
from unittest.mock import MagicMock


def _make_connector(
    balance=1000.0,
    position=None,   # None → flat; or dict {"side","qty","entry_price","unrealized_pnl"}
    mark_price=50000.0,
    step_size="0.001",
):
    connector = MagicMock()
    connector.exchange_name = "binance"
    connector.exchange_info.return_value = {
        "symbol": "BTCUSDT",
        "filters": [{"filterType": "LOT_SIZE", "stepSize": step_size}],
    }
    connector.fetch_balance.return_value = balance
    connector.fetch_position.return_value = position
    connector.fetch_mark_price.return_value = mark_price
    return connector


def _make_history_db():
    return MagicMock()


def _make_ex(connector=None, history_db=None, **kwargs):
    from backtest.live_exchange import LiveExchange
    return LiveExchange(
        connector=connector or _make_connector(),
        history_db=history_db or _make_history_db(),
        account_id="testacc",
        symbol="BTCUSDT",
        leverage=kwargs.get("leverage", 10),
        commission_rate=kwargs.get("commission_rate", 0.0004),
        dry_run=kwargs.get("dry_run", False),
    )


class TestLiveExchangeSync:
    def test_balance_after_sync(self):
        ex = _make_ex(connector=_make_connector(balance=2500.0))
        ex.sync()
        assert ex.balance == 2500.0

    def test_no_position_when_flat(self):
        ex = _make_ex(connector=_make_connector(position=None))
        ex.sync()
        assert ex.get_position("BTCUSDT") is None

    def test_long_position_parsed(self):
        pos_data = {"side": "long", "qty": 0.1, "entry_price": 50000.0, "unrealized_pnl": 200.0}
        ex = _make_ex(connector=_make_connector(position=pos_data))
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.unrealized_pnl == 200.0
        assert abs(pos.quantity - 5000.0) < 0.01   # 0.1 * 50000
        assert abs(pos.margin - 500.0) < 0.01       # 5000 / 10

    def test_short_position_parsed(self):
        pos_data = {"side": "short", "qty": 0.05, "entry_price": 48000.0, "unrealized_pnl": -100.0}
        ex = _make_ex(connector=_make_connector(position=pos_data))
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

    def test_equity_no_position(self):
        ex = _make_ex(connector=_make_connector(balance=1000.0, position=None))
        ex.sync()
        assert ex.equity == 1000.0

    def test_equity_with_position(self):
        pos_data = {"side": "long", "qty": 0.1, "entry_price": 50000.0, "unrealized_pnl": 200.0}
        ex = _make_ex(connector=_make_connector(balance=500.0, position=pos_data))
        ex.sync()
        # equity = balance(500) + margin(500) + unrealized(200) = 1200
        assert abs(ex.equity - 1200.0) < 0.01

    def test_get_position_wrong_symbol_returns_none(self):
        pos_data = {"side": "long", "qty": 0.1, "entry_price": 50000.0, "unrealized_pnl": 0.0}
        ex = _make_ex(connector=_make_connector(position=pos_data))
        ex.sync()
        assert ex.get_position("ETHUSDT") is None

    def test_sync_records_position_in_history(self):
        history_db = _make_history_db()
        ex = _make_ex(connector=_make_connector(balance=1000.0), history_db=history_db)
        ex.sync()
        history_db.record_position.assert_called_once()


class TestLiveExchangeSubmitOrder:
    def test_dry_run_does_not_call_api(self):
        connector = _make_connector(mark_price=50000.0)
        history_db = _make_history_db()
        ex = _make_ex(connector=connector, history_db=history_db, dry_run=True)
        ex.sync()
        order = ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        connector.submit_order.assert_not_called()
        assert order.status == "filled"

    def test_dry_run_records_order_in_history(self):
        history_db = _make_history_db()
        ex = _make_ex(connector=_make_connector(), history_db=history_db, dry_run=True)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        history_db.upsert_order.assert_called()

    def test_dry_run_prints_log(self, capsys):
        ex = _make_ex(dry_run=True)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        captured = capsys.readouterr()
        assert "dry-run" in captured.out.lower()

    def test_market_order_converts_usdt_to_contracts(self):
        connector = _make_connector(mark_price=50000.0)
        connector.submit_order.return_value = {"orderId": 1}
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        call_kwargs = connector.submit_order.call_args
        # quantity arg is contracts: 1000/50000 = 0.02
        assert call_kwargs[0][3] == pytest.approx(0.020, abs=0.001)

    def test_order_id_tracked_in_pending(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 42}
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "42" in ex._pending_order_ids

    def test_sell_side_passed_correctly(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 7}
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        assert connector.submit_order.call_args[0][1] == "sell"


class TestLiveExchangeWaitFills:
    def test_clears_pending_when_filled(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 99}
        connector.query_order.return_value = {
            "order_id": "99", "symbol": "BTCUSDT", "side": "buy", "type": "market",
            "quantity": 500.0, "price": None, "status": "filled",
            "filled_price": 50000.0, "filled_qty": 0.01, "commission": 0.2,
            "ts": 5000, "filled_at": 5100,
        }
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "99" in ex._pending_order_ids
        ex.wait_fills(timeout=5.0)
        assert ex._pending_order_ids == []

    def test_fill_upserted_to_history(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 99}
        connector.query_order.return_value = {
            "order_id": "99", "symbol": "BTCUSDT", "side": "buy", "type": "market",
            "quantity": 500.0, "price": None, "status": "filled",
            "filled_price": 50000.0, "filled_qty": 0.01, "commission": 0.2,
            "ts": 5000, "filled_at": 5100,
        }
        history_db = _make_history_db()
        ex = _make_ex(connector=connector, history_db=history_db)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        ex.wait_fills(timeout=5.0)
        # upsert_order called: once on submit, once on fill
        assert history_db.upsert_order.call_count == 2

    def test_warns_when_order_never_fills(self, capsys):
        connector = _make_connector()
        connector.query_order.return_value = {
            "order_id": "77", "symbol": "BTCUSDT", "side": "buy", "type": "market",
            "quantity": 500.0, "price": None, "status": "new",
            "filled_price": None, "filled_qty": None, "commission": None,
            "ts": 1000, "filled_at": None,
        }
        ex = _make_ex(connector=connector)
        ex._pending_order_ids = ["77"]
        ex.wait_fills(timeout=0.01)
        captured = capsys.readouterr()
        assert "77" in captured.out or "WARN" in captured.out

    def test_no_api_call_when_no_pending(self):
        connector = _make_connector()
        ex = _make_ex(connector=connector)
        ex.wait_fills()
        connector.query_order.assert_not_called()
```

- [ ] **Step 2: Run tests to see current failures**

```bash
pytest tests/test_live_exchange.py -v 2>&1 | head -30
```
Expected: import errors or signature mismatches — confirms tests are driving the refactor.

- [ ] **Step 3: Rewrite live_exchange.py**

```python
# src/backtest/live_exchange.py
import math
import time
import uuid

from backtest.live_connector import BaseExchangeConnector
from backtest.live_history import LiveHistoryDB
from backtest.models import Order, Position


class LiveExchange:
    def __init__(self, connector: BaseExchangeConnector, history_db: LiveHistoryDB,
                 account_id: str, symbol: str, leverage: int,
                 commission_rate: float, dry_run: bool = False):
        self._connector = connector
        self._history_db = history_db
        self._account_id = account_id
        self._symbol = symbol
        self._leverage = leverage
        self._commission_rate = commission_rate
        self._dry_run = dry_run
        self._balance: float = 0.0
        self._position: Position | None = None
        self._current_price: float = 0.0
        self._lot_step: float = 0.001
        self._pending_order_ids: list[str] = []
        self._fetch_lot_step()

    def _fetch_lot_step(self) -> None:
        info = self._connector.exchange_info(self._symbol)
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                self._lot_step = float(f["stepSize"])
                return
        raise ValueError(f"LOT_SIZE filter not found for {self._symbol!r}")

    def _round_qty(self, qty: float) -> float:
        if self._lot_step <= 0:
            return qty
        precision = max(0, -int(round(math.log10(self._lot_step))))
        return round(qty, precision)

    def sync(self) -> None:
        if not self._dry_run:
            self._balance = self._connector.fetch_balance()
            pos_data = self._connector.fetch_position(self._symbol)
            if pos_data is None:
                self._position = None
            else:
                notional = pos_data["qty"] * pos_data["entry_price"]
                margin = notional / self._leverage
                self._position = Position(
                    symbol=self._symbol,
                    side=pos_data["side"],
                    quantity=notional,
                    entry_price=pos_data["entry_price"],
                    leverage=self._leverage,
                    unrealized_pnl=pos_data["unrealized_pnl"],
                    margin=margin,
                )
        self._current_price = self._connector.fetch_mark_price(self._symbol)
        self._history_db.record_position(
            self._account_id, self._connector.exchange_name,
            self._symbol, self._position, self._balance, self.equity,
            int(time.time() * 1000),
        )

    def get_position(self, symbol: str) -> Position | None:
        if self._position and self._position.symbol == symbol:
            return self._position
        return None

    @property
    def leverage(self) -> int:
        return self._leverage

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        if self._position is None:
            return self._balance
        return self._balance + self._position.margin + self._position.unrealized_pnl

    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None) -> Order:
        order_id = uuid.uuid4().hex[:8]
        ts = int(time.time() * 1000)

        if self._dry_run:
            order_dict = {
                "order_id": order_id, "symbol": symbol, "side": side, "type": type_,
                "quantity": quantity, "price": price, "status": "filled",
                "filled_price": self._current_price,
                "filled_qty": quantity / self._current_price if self._current_price else 0,
                "commission": quantity * self._commission_rate,
                "ts": ts, "filled_at": ts,
            }
            self._history_db.upsert_order(self._account_id, self._connector.exchange_name, order_dict)
            print(f"[dry-run] {side.upper()} {type_} {quantity:.2f} USDT @ {price or 'market'}")
            return Order(
                id=order_id, symbol=symbol, side=side, type=type_,
                quantity=quantity, price=price, status="filled",
                filled_price=self._current_price, filled_at=ts,
                commission=quantity * self._commission_rate,
            )

        contract_qty = self._round_qty(quantity / self._current_price)
        if contract_qty <= 0:
            return Order(id=order_id, symbol=symbol, side=side, type=type_,
                         quantity=quantity, price=price, status="canceled")

        resp = self._connector.submit_order(symbol, side, type_, contract_qty, price)
        exchange_id = str(resp["orderId"])

        order_dict = {
            "order_id": exchange_id, "symbol": symbol, "side": side, "type": type_,
            "quantity": quantity, "price": price, "status": "pending",
            "filled_price": None, "filled_qty": None, "commission": None,
            "ts": ts, "filled_at": None,
        }
        self._history_db.upsert_order(self._account_id, self._connector.exchange_name, order_dict)
        self._pending_order_ids.append(exchange_id)
        return Order(id=exchange_id, symbol=symbol, side=side, type=type_,
                     quantity=quantity, price=price, status="pending")

    def wait_fills(self, timeout: float = 30.0) -> None:
        if not self._pending_order_ids:
            return
        deadline = time.time() + timeout
        remaining = list(self._pending_order_ids)
        while remaining and time.time() < deadline:
            still_pending = []
            for oid in remaining:
                order_dict = self._connector.query_order(self._symbol, oid)
                if order_dict["status"] not in ("filled", "canceled", "expired", "rejected"):
                    still_pending.append(oid)
                else:
                    self._history_db.upsert_order(
                        self._account_id, self._connector.exchange_name, order_dict
                    )
            remaining = still_pending
            if remaining:
                time.sleep(0.5)
        if remaining:
            print(f"[WARN] orders not confirmed within {timeout}s: {remaining}")
        self._pending_order_ids.clear()
```

- [ ] **Step 4: Add missing pytest import to test file**

Add at top of `tests/test_live_exchange.py`:
```python
import pytest
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_live_exchange.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/live_exchange.py tests/test_live_exchange.py
git commit -m "refactor: LiveExchange takes BaseExchangeConnector + LiveHistoryDB; records order/position history"
```

---

## Task 4: Refactor LiveFeed

**Files:**
- Modify: `src/backtest/live_feed.py`
- Modify: `tests/test_live_feed.py`

- [ ] **Step 1: Update live_feed.py** (one-line type change, no logic change)

In `src/backtest/live_feed.py`, change the constructor signature and replace all `self._client` with `self._connector`:

```python
# src/backtest/live_feed.py
import time
from collections.abc import Iterator

from backtest.live_connector import BaseExchangeConnector
from backtest.models import Bar

_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400,
}


def _interval_to_seconds(interval: str) -> int:
    if interval not in _INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval: {interval!r}. Choose from {list(_INTERVAL_SECONDS)}")
    return _INTERVAL_SECONDS[interval]


def _bar_close_time(interval_sec: int, ref_time: float | None = None) -> float:
    now = ref_time if ref_time is not None else time.time()
    last_close = (int(now) // interval_sec) * interval_sec
    return float(last_close + interval_sec)


def _kline_to_bar(symbol: str, interval: str, k: list) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=int(k[0]),
        open=float(k[1]),
        high=float(k[2]),
        low=float(k[3]),
        close=float(k[4]),
        volume=float(k[5]),
        interval=interval,
    )


class LiveFeed:
    def __init__(self, connector: BaseExchangeConnector, symbol: str, interval: str,
                 close_buffer_sec: float = 5.0):
        self._connector = connector
        self._symbol = symbol
        self._interval = interval
        self._interval_sec = _interval_to_seconds(interval)
        self._close_buffer_sec = close_buffer_sec
        self._last_bar_ts: int | None = None

    def __iter__(self) -> Iterator[Bar]:
        while True:
            next_close = _bar_close_time(self._interval_sec)
            sleep_until = next_close + self._close_buffer_sec
            now = time.time()
            if sleep_until > now:
                time.sleep(sleep_until - now)

            klines = self._connector.klines(
                symbol=self._symbol,
                interval=self._interval,
                limit=2,
            )
            if len(klines) < 2:
                continue

            closed_bar = _kline_to_bar(self._symbol, self._interval, klines[-2])

            if self._last_bar_ts is not None:
                expected_ts = self._last_bar_ts + self._interval_sec * 1000
                if closed_bar.timestamp > expected_ts:
                    yield from self._backfill(expected_ts, closed_bar.timestamp)

            self._last_bar_ts = closed_bar.timestamp
            yield closed_bar

    def _backfill(self, from_ts: int, to_ts: int) -> Iterator[Bar]:
        klines = self._connector.klines(
            symbol=self._symbol,
            interval=self._interval,
            startTime=from_ts,
            endTime=to_ts - 1,
            limit=1000,
        )
        for k in klines:
            bar = _kline_to_bar(self._symbol, self._interval, k)
            if bar.timestamp < to_ts:
                self._last_bar_ts = bar.timestamp
                yield bar
```

- [ ] **Step 2: Update test_live_feed.py** — rename `client` to `connector` in the feed iteration and backfill tests

In `tests/test_live_feed.py`, replace every `client = MagicMock()` with:
```python
connector = MagicMock()
connector.exchange_name = "binance"
```
And replace `LiveFeed(client, ...)` with `LiveFeed(connector, ...)` and `client.klines` with `connector.klines`. The helper functions `_make_kline`, `_interval_to_seconds`, `_bar_close_time`, `_kline_to_bar` are unchanged.

Full updated iteration/backfill test classes:

```python
class TestLiveFeedIteration:
    def test_yields_second_to_last_kline_as_closed_bar(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            bar = next(iter(feed))
        assert bar.timestamp == ts1
        assert bar.symbol == "BTCUSDT"

    def test_updates_last_bar_ts_after_yield(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            next(iter(feed))
        assert feed._last_bar_ts == ts1


class TestLiveFeedBackfill:
    def test_backfill_yields_missing_bar(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000
        ts1, ts2, ts3 = ts_base, ts_base + interval_ms, ts_base + 2 * interval_ms
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts2, close=200.0)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts1
        bars = list(feed._backfill(ts2, ts3))
        assert len(bars) == 1
        assert bars[0].timestamp == ts2
        assert bars[0].close == 200.0

    def test_backfill_excludes_current_bar_timestamp(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000
        ts2, ts3 = ts_base + interval_ms, ts_base + 2 * interval_ms
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts2), _make_kline(ts3)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts_base
        bars = list(feed._backfill(ts2, ts3))
        assert len(bars) == 1
        assert bars[0].timestamp == ts2

    def test_backfill_updates_last_bar_ts(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000
        ts2, ts3 = ts_base + interval_ms, ts_base + 2 * interval_ms
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts_base
        list(feed._backfill(ts2, ts3))
        assert feed._last_bar_ts == ts2

    def test_no_gap_no_backfill_called(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts1
        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            bar = next(iter(feed))
        assert connector.klines.call_count == 1
        assert bar.timestamp == ts1
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_live_feed.py -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/backtest/live_feed.py tests/test_live_feed.py
git commit -m "refactor: LiveFeed takes BaseExchangeConnector instead of UMFutures client"
```

---

## Task 5: Refactor LiveEngine — history writes + startup sync

**Files:**
- Modify: `src/backtest/live_engine.py`
- Modify: `tests/test_live_engine.py`

- [ ] **Step 1: Rewrite live_engine.py**

```python
# src/backtest/live_engine.py
import hashlib
import json
import signal
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from backtest.live_connector import BaseExchangeConnector
from backtest.live_exchange import LiveExchange
from backtest.live_feed import LiveFeed
from backtest.live_history import LiveHistoryDB
from backtest.models import Bar
from backtest.strategy import BaseStrategy


def _account_id(api_key: str) -> str:
    if not api_key:
        return "dry_run"
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


class _StateDB:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_state (
                    account_id  TEXT NOT NULL,
                    strategy    TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    interval    TEXT NOT NULL,
                    state_json  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (account_id, strategy, symbol, interval)
                )
            """)

    def load(self, account_id: str, strategy: str, symbol: str, interval: str) -> dict | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT state_json FROM live_state "
                "WHERE account_id=? AND strategy=? AND symbol=? AND interval=?",
                (account_id, strategy, symbol, interval),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, account_id: str, strategy: str, symbol: str, interval: str, state: dict) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO live_state "
                "(account_id, strategy, symbol, interval, state_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (account_id, strategy, symbol, interval,
                 json.dumps(state), datetime.now(timezone.utc).isoformat()),
            )


class LiveEngine:
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        symbol: str,
        interval: str,
        leverage: int,
        connector: BaseExchangeConnector,
        history_db: LiveHistoryDB,
        account_id: str,
        commission_rate: float = 0.0004,
        dry_run: bool = False,
        state_db: str = "data/live_state.db",
        sync_interval: int = 300,
    ):
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.interval = interval
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.dry_run = dry_run
        self._connector = connector
        self._history_db = history_db
        self._account_id = account_id
        self._state_db = _StateDB(state_db)
        self._sync_interval = sync_interval
        self._stop_sync: threading.Event | None = None

    def run(self) -> None:
        if not self.dry_run:
            try:
                self._connector.change_leverage(self.symbol, self.leverage)
            except Exception as e:
                print(f"[WARN] change_leverage: {e}", file=sys.stderr)

        live_exchange = LiveExchange(
            connector=self._connector,
            history_db=self._history_db,
            account_id=self._account_id,
            symbol=self.symbol,
            leverage=self.leverage,
            commission_rate=self.commission_rate,
            dry_run=self.dry_run,
        )
        live_exchange.sync()

        strategy = self.strategy_class(exchange=live_exchange, symbol=self.symbol)
        strategy.on_init()

        strategy_name = self.strategy_class.__name__
        saved = self._state_db.load(self._account_id, strategy_name, self.symbol, self.interval)
        if saved:
            strategy.load_state(saved)
            print(f"[INFO] Restored state: account={self._account_id} "
                  f"{strategy_name}/{self.symbol}/{self.interval}")

        # Startup sync: pull history since last known timestamp
        self._do_sync()

        self._print_startup_summary(live_exchange)

        # Start background sync daemon
        self._stop_sync = threading.Event()
        sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        sync_thread.start()

        def _handle_sigterm(_sig, _frame):
            self._on_exit(live_exchange)
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle_sigterm)

        feed = LiveFeed(connector=self._connector, symbol=self.symbol, interval=self.interval)
        try:
            for bar in feed:
                self._process_bar(bar, strategy, live_exchange)
        except KeyboardInterrupt:
            self._on_exit(live_exchange)

    def _process_bar(self, bar: Bar, strategy: BaseStrategy, live_exchange: LiveExchange) -> None:
        ts = datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        try:
            live_exchange.sync()
            strategy._push_bar(bar)
            live_exchange.wait_fills(timeout=30.0)
            self._save_state(strategy)
            pos = live_exchange.get_position(self.symbol)
            pos_str = f"{pos.side} {pos.quantity:.2f}@{pos.entry_price:.2f}" if pos else "flat"
            print(f"[{ts}] balance={live_exchange.balance:.2f} equity={live_exchange.equity:.2f} pos={pos_str}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self._alert(f"[ERROR] bar {ts}: {e}")
            traceback.print_exc(file=sys.stderr)

    def _save_state(self, strategy: BaseStrategy) -> None:
        state = strategy.save_state()
        self._state_db.save(
            self._account_id, self.strategy_class.__name__, self.symbol, self.interval, state
        )
        self._history_db.record_state_snapshot(
            account_id=self._account_id,
            exchange=self._connector.exchange_name,
            strategy=self.strategy_class.__name__,
            symbol=self.symbol,
            interval=self.interval,
            state=state,
            ts=int(time.time() * 1000),
        )

    def _do_sync(self) -> None:
        exchange = self._connector.exchange_name
        since_orders = self._history_db.latest_ts(self._account_id, exchange, self.symbol, "orders")
        orders = self._connector.fetch_orders(self.symbol, since_ms=since_orders or None)
        for order in orders:
            self._history_db.upsert_order(self._account_id, exchange, order)

        since_trades = self._history_db.latest_ts(self._account_id, exchange, self.symbol, "trades")
        trades = self._connector.fetch_trades(self.symbol, since_ms=since_trades or None)
        self._history_db.upsert_trades(self._account_id, exchange, trades)

    def _sync_loop(self) -> None:
        while self._stop_sync is not None and not self._stop_sync.wait(self._sync_interval):
            try:
                self._do_sync()
            except Exception as e:
                print(f"[WARN] background sync error: {e}", file=sys.stderr)

    def _alert(self, msg: str) -> None:
        print(msg, file=sys.stderr)

    def _print_startup_summary(self, live_exchange: LiveExchange) -> None:
        mode = "TESTNET" if True else "MAINNET"  # connector knows, but engine doesn't need to
        dry = " [DRY-RUN]" if self.dry_run else ""
        print(f"\n=== LiveEngine{dry} ===")
        print(f"Strategy: {self.strategy_class.__name__}  Symbol: {self.symbol}  "
              f"Interval: {self.interval}  Leverage: {self.leverage}x")
        print(f"Account ID: {self._account_id}  Exchange: {self._connector.exchange_name}")
        print(f"Balance: {live_exchange.balance:.2f} USDT  Equity: {live_exchange.equity:.2f} USDT")
        pos = live_exchange.get_position(self.symbol)
        if pos:
            print(f"Position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}  "
                  f"PnL: {pos.unrealized_pnl:.2f}")
        else:
            print("Position: flat")
        print()

    def _on_exit(self, live_exchange: LiveExchange) -> None:
        if self._stop_sync is not None:
            self._stop_sync.set()
        pos = live_exchange.get_position(self.symbol)
        print("\n=== Stopping LiveEngine ===")
        if pos:
            print(f"Open position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}")
            try:
                answer = input("Close position before exit? [y/N] ")
            except EOFError:
                answer = "n"
            if answer.strip().lower() == "y":
                side = "sell" if pos.side == "long" else "buy"
                live_exchange.submit_order(self.symbol, side, "market", pos.quantity)
                live_exchange.wait_fills(timeout=30.0)
                print("Position closed.")
        else:
            print("No open position.")
```

- [ ] **Step 2: Rewrite test_live_engine.py**

```python
# tests/test_live_engine.py
import json
import tempfile
from unittest.mock import MagicMock, patch

from backtest.strategy import BaseStrategy
from backtest.models import Bar


def _make_connector(balance=1000.0, position=None, mark_price=50000.0):
    connector = MagicMock()
    connector.exchange_name = "binance"
    connector.exchange_info.return_value = {
        "symbol": "BTCUSDT",
        "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}],
    }
    connector.fetch_balance.return_value = balance
    connector.fetch_position.return_value = position
    connector.fetch_mark_price.return_value = mark_price
    connector.fetch_orders.return_value = []
    connector.fetch_trades.return_value = []
    return connector


def _make_history_db():
    return MagicMock()


def _make_bar(ts: int = 1_700_000_000_000) -> Bar:
    return Bar("BTCUSDT", ts, 50000.0, 51000.0, 49000.0, 50500.0, 100.0, "1h")


class _NoOpStrategy(BaseStrategy):
    def on_bar(self, bar): pass


class TestProcessBar:
    def _make_engine(self, connector=None, history_db=None, state_dir=None):
        from backtest.live_engine import LiveEngine
        from backtest.live_exchange import LiveExchange
        conn = connector or _make_connector()
        hdb = history_db or _make_history_db()
        with tempfile.TemporaryDirectory() as tmp:
            engine = LiveEngine(
                strategy_class=_NoOpStrategy,
                symbol="BTCUSDT",
                interval="1h",
                leverage=10,
                connector=conn,
                history_db=hdb,
                account_id="testacc",
                dry_run=True,
                state_db=f"{tmp}/state.db",
            )
            ex = LiveExchange(
                connector=conn, history_db=hdb,
                account_id="testacc", symbol="BTCUSDT",
                leverage=10, commission_rate=0.0004, dry_run=True,
            )
            ex.sync()
            strategy = _NoOpStrategy(exchange=ex, symbol="BTCUSDT")
            strategy.on_init()
            return engine, ex, strategy

    def test_process_bar_prints_status(self, capsys):
        engine, ex, strategy = self._make_engine()
        engine._process_bar(_make_bar(), strategy, ex)
        captured = capsys.readouterr()
        assert "balance=" in captured.out

    def test_process_bar_saves_state(self):
        engine, ex, strategy = self._make_engine()
        engine._process_bar(_make_bar(), strategy, ex)
        engine._history_db.record_state_snapshot.assert_called_once()

    def test_process_bar_exception_does_not_propagate(self):
        engine, ex, strategy = self._make_engine()
        strategy._push_bar = MagicMock(side_effect=RuntimeError("boom"))
        engine._process_bar(_make_bar(), strategy, ex)  # must not raise


class TestDoSync:
    def test_do_sync_fetches_orders_and_trades(self):
        from backtest.live_engine import LiveEngine
        connector = _make_connector()
        history_db = _make_history_db()
        history_db.latest_ts.return_value = 0
        with tempfile.TemporaryDirectory() as tmp:
            engine = LiveEngine(
                strategy_class=_NoOpStrategy, symbol="BTCUSDT", interval="1h",
                leverage=10, connector=connector, history_db=history_db,
                account_id="testacc", state_db=f"{tmp}/state.db",
            )
            engine._do_sync()
        connector.fetch_orders.assert_called_once_with("BTCUSDT", since_ms=None)
        connector.fetch_trades.assert_called_once_with("BTCUSDT", since_ms=None)

    def test_do_sync_uses_latest_ts_as_since(self):
        from backtest.live_engine import LiveEngine
        connector = _make_connector()
        history_db = _make_history_db()
        history_db.latest_ts.side_effect = lambda acc, ex, sym, table: (
            1_700_000_000_000 if table == "orders" else 1_699_000_000_000
        )
        with tempfile.TemporaryDirectory() as tmp:
            engine = LiveEngine(
                strategy_class=_NoOpStrategy, symbol="BTCUSDT", interval="1h",
                leverage=10, connector=connector, history_db=history_db,
                account_id="testacc", state_db=f"{tmp}/state.db",
            )
            engine._do_sync()
        connector.fetch_orders.assert_called_once_with("BTCUSDT", since_ms=1_700_000_000_000)
        connector.fetch_trades.assert_called_once_with("BTCUSDT", since_ms=1_699_000_000_000)


class TestStateDB:
    def test_save_and_load_roundtrip(self):
        from backtest.live_engine import _StateDB
        with tempfile.TemporaryDirectory() as tmp:
            db = _StateDB(f"{tmp}/state.db")
            db.save("acc1", "MyStrat", "BTCUSDT", "1h", {"count": 42})
            result = db.load("acc1", "MyStrat", "BTCUSDT", "1h")
            assert result == {"count": 42}

    def test_load_returns_none_when_missing(self):
        from backtest.live_engine import _StateDB
        with tempfile.TemporaryDirectory() as tmp:
            db = _StateDB(f"{tmp}/state.db")
            assert db.load("acc1", "MyStrat", "BTCUSDT", "1h") is None

    def test_save_overwrites_previous(self):
        from backtest.live_engine import _StateDB
        with tempfile.TemporaryDirectory() as tmp:
            db = _StateDB(f"{tmp}/state.db")
            db.save("acc1", "MyStrat", "BTCUSDT", "1h", {"v": 1})
            db.save("acc1", "MyStrat", "BTCUSDT", "1h", {"v": 2})
            assert db.load("acc1", "MyStrat", "BTCUSDT", "1h") == {"v": 2}
```

- [ ] **Step 3: Run all live tests**

```bash
pytest tests/test_live_engine.py tests/test_live_exchange.py tests/test_live_feed.py -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/backtest/live_engine.py tests/test_live_engine.py
git commit -m "refactor: LiveEngine takes connector+history_db; adds startup sync and per-bar history writes"
```

---

## Task 6: Update CLI

**Files:**
- Modify: `src/backtest/__main__.py`

- [ ] **Step 1: Update cmd_live and argument parser**

In `src/backtest/__main__.py`, replace `cmd_live` and the `p_live` parser section:

```python
def cmd_live(args: argparse.Namespace) -> None:
    import hashlib
    import os
    from backtest.live_connector import BinanceConnector
    from backtest.live_engine import LiveEngine
    from backtest.live_history import LiveHistoryDB

    if args.env_file:
        try:
            env = _load_env_file(args.env_file)
        except FileNotFoundError:
            print(f"Error: env file not found: {args.env_file}")
            sys.exit(1)
        api_key = env.get("BINANCE_API_KEY", "")
        secret = env.get("BINANCE_SECRET", "")
    else:
        api_key = os.environ.get("BINANCE_API_KEY", "")
        secret = os.environ.get("BINANCE_SECRET", "")

    if (not api_key or not secret) and not args.dry_run:
        print(
            "Error: BINANCE_API_KEY and BINANCE_SECRET must be set via environment variables "
            "or --env-file. (Not required for --dry-run)"
        )
        sys.exit(1)

    strategy_class = _load_strategy(args.strategy)
    if args.extra_params:
        _apply_extra_params(strategy_class, args.extra_params)

    if args.exchange == "binance":
        connector = BinanceConnector(api_key=api_key, secret=secret, testnet=not args.no_testnet)
    else:
        print(f"Error: unsupported exchange: {args.exchange}")
        sys.exit(1)

    account_id = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "dry_run"
    history_db = LiveHistoryDB(args.history_db)

    engine = LiveEngine(
        strategy_class=strategy_class,
        symbol=args.symbol,
        interval=args.interval,
        leverage=args.leverage,
        connector=connector,
        history_db=history_db,
        account_id=account_id,
        commission_rate=args.commission_rate,
        dry_run=args.dry_run,
        state_db=args.state_db,
        sync_interval=args.sync_interval,
    )
    engine.run()
```

Replace the `p_live` parser block with:

```python
p_live = sub.add_parser("live", help="Run strategy in live trading mode")
p_live.add_argument("--strategy", required=True, help="Path to strategy .py file")
p_live.add_argument("--symbol", required=True)
p_live.add_argument("--interval", required=True)
p_live.add_argument("--leverage", type=int, required=True)
p_live.add_argument("--exchange", default="binance", choices=["binance"],
                    help="Exchange connector (default: binance)")
p_live.add_argument("--commission-rate", type=float, default=0.0004, dest="commission_rate")
p_live.add_argument("--no-testnet", action="store_true", default=False,
                    help="Use mainnet (default is testnet)")
p_live.add_argument("--dry-run", action="store_true", default=False,
                    help="Log orders without sending them (no API key required)")
p_live.add_argument("--state-db", default="data/live_state.db", dest="state_db",
                    help="SQLite path for strategy state (default: data/live_state.db)")
p_live.add_argument("--history-db", default="data/live_history.db", dest="history_db",
                    help="SQLite path for trading history (default: data/live_history.db)")
p_live.add_argument("--sync-interval", type=int, default=300, dest="sync_interval",
                    help="Seconds between background exchange syncs (default: 300)")
p_live.add_argument("--env-file", default=None, dest="env_file",
                    help="Path to .env file containing BINANCE_API_KEY and BINANCE_SECRET")
p_live.add_argument("extra_params", nargs=argparse.REMAINDER,
                    help="Extra strategy params e.g. --CONSECUTIVE_THRESHOLD 5")
```

- [ ] **Step 2: Verify CLI help**

```bash
python -m backtest live --help
```
Expected output contains: `--exchange`, `--sync-interval`, `--history-db`, `--history-db`

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests pass, no regressions.

- [ ] **Step 4: Update README** — replace stale `--state-dir` references in multi-instance section, add `--sync-interval` and `--history-db` to parameter table, add history query commands

In `README.md`, update the live parameter table:

```markdown
| --exchange | 交易所 connector | binance |
| --sync-interval | 后台同步间隔（秒） | 300 |
| --history-db | 历史数据库路径 | data/live_history.db |
```

- [ ] **Step 5: Commit**

```bash
git add src/backtest/__main__.py README.md
git commit -m "feat: wire live CLI to connector/history_db; add --exchange, --sync-interval, --history-db args"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `live_history.db` with 4 tables (state_snapshots, orders, positions, trades) — Task 1
- ✅ Append-only state snapshots with Unix timestamps — Task 1 + Task 5 `_save_state`
- ✅ `BaseExchangeConnector` ABC — Task 2
- ✅ `BinanceConnector` wrapping UMFutures — Task 2
- ✅ Clock-skew correction moved into `BinanceConnector.__init__` — Task 2
- ✅ `LiveExchange` takes connector — Task 3
- ✅ `submit_order` records pending order; `wait_fills` records fill — Task 3
- ✅ `LiveFeed` takes connector — Task 4
- ✅ Startup sync (fetch_orders + fetch_trades since latest_ts) — Task 5 `_do_sync`
- ✅ Background sync thread with `threading.Event` — Task 5 `_sync_loop`
- ✅ Per-bar position record — Task 3 `sync()` + Task 5 `_process_bar`
- ✅ Per-bar state snapshot — Task 5 `_save_state`
- ✅ `live_state.db` unchanged (INSERT OR REPLACE) — Task 5
- ✅ `--exchange`, `--sync-interval`, `--history-db` CLI args — Task 6
- ✅ Thread safety via `LiveHistoryDB._lock` — Task 1
