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
        with self._lock, sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT MAX(ts) FROM {table} WHERE account_id=? AND exchange=? AND symbol=?",
                (account_id, exchange, symbol),
            ).fetchone()
        return row[0] if row and row[0] is not None else 0
