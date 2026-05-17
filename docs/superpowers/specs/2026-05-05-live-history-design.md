# Live Trading History & Replay Design

**Date:** 2026-05-05  
**Status:** Approved

## Goal

Record full live trading activity (state snapshots, orders, positions, trades) to SQLite for post-trade review. Support multiple accounts and multiple exchanges via an abstraction layer.

---

## Databases

### `data/live_state.db` (existing, unchanged)

Keeps INSERT OR REPLACE semantics for fast startup state recovery.

### `data/live_history.db` (new)

Append-only historical record. Four tables, all keyed on `(account_id, exchange)`:

```sql
CREATE TABLE state_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id  TEXT    NOT NULL,
    exchange    TEXT    NOT NULL,
    strategy    TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,
    state_json  TEXT    NOT NULL,
    ts          INTEGER NOT NULL   -- Unix ms
);

CREATE TABLE orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL,
    exchange     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    order_id     TEXT    NOT NULL,
    side         TEXT    NOT NULL,   -- buy | sell
    type         TEXT    NOT NULL,   -- market | limit
    quantity     REAL    NOT NULL,   -- USDT notional
    price        REAL,
    status       TEXT    NOT NULL,   -- pending | filled | canceled
    filled_price REAL,
    filled_qty   REAL,
    commission   REAL,
    ts           INTEGER NOT NULL,   -- submission time ms
    filled_at    INTEGER,
    UNIQUE (exchange, account_id, order_id)
);

CREATE TABLE positions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id     TEXT    NOT NULL,
    exchange       TEXT    NOT NULL,
    symbol         TEXT    NOT NULL,
    side           TEXT,             -- long | short | NULL (flat)
    quantity       REAL    NOT NULL,
    entry_price    REAL,
    leverage       INTEGER,
    unrealized_pnl REAL,
    margin         REAL,
    balance        REAL    NOT NULL,
    equity         REAL    NOT NULL,
    ts             INTEGER NOT NULL
);

CREATE TABLE trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL,
    exchange     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    trade_id     TEXT    NOT NULL,
    order_id     TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    price        REAL    NOT NULL,
    qty          REAL    NOT NULL,   -- contract quantity
    notional     REAL    NOT NULL,   -- USDT value
    commission   REAL    NOT NULL,
    realized_pnl REAL,
    ts           INTEGER NOT NULL,
    UNIQUE (exchange, account_id, trade_id)
);
```

---

## New Module: `live_connector.py`

Abstract base class + Binance implementation.

```
BaseExchangeConnector (ABC)
  exchange_name: str              # "binance" / "okx" / "bybit"
  server_time() → int             # Unix ms, for clock skew correction
  exchange_info(symbol) → dict
  klines(symbol, interval, **kw) → list
  fetch_balance() → float
  fetch_position(symbol) → dict | None
  fetch_mark_price(symbol) → float
  fetch_orders(symbol, since_ms) → list[dict]
  fetch_trades(symbol, since_ms) → list[dict]
  submit_order(symbol, side, type_, quantity, price) → dict
  query_order(symbol, order_id) → dict
  change_leverage(symbol, leverage) → None

BinanceConnector(BaseExchangeConnector)
  exchange_name = "binance"
  Wraps UMFutures; applies clock-skew correction in __init__
```

Adding a new exchange requires only implementing `BaseExchangeConnector` — no changes to engine, exchange, or feed.

---

## New Module: `live_history.py`

Owns all writes to `live_history.db`.

```
LiveHistoryDB(db_path)
  record_state_snapshot(account_id, exchange, strategy, symbol, interval, state, ts)
  record_position(account_id, exchange, symbol, position, balance, equity, ts)
  upsert_order(account_id, exchange, order_dict)       # INSERT OR IGNORE + UPDATE on status change
  upsert_trades(account_id, exchange, trades: list)    # bulk, UNIQUE deduplication
  latest_ts(account_id, exchange, symbol, table) → int # for incremental sync
```

---

## Updated: `live_exchange.py`

- Constructor takes `BaseExchangeConnector` instead of `UMFutures`
- All API calls routed through connector
- `submit_order()` calls `history_db.upsert_order()` on submission and again after fill

## Updated: `live_feed.py`

- Constructor takes `BaseExchangeConnector` instead of `UMFutures`
- Uses `connector.klines()` for bar fetching

---

## Updated: `live_engine.py`

### Startup sequence

```
1. BinanceConnector.__init__()  ← includes clock-skew correction
2. connector.change_leverage()
3. live_exchange.sync() → history_db.record_position()
4. startup sync: fetch_orders + fetch_trades (since latest_ts) → upsert
5. strategy.load_state() from live_state.db
6. start background sync thread (daemon)
```

### Per-bar (`_process_bar`)

```
1. live_exchange.sync() → history_db.record_position()
2. strategy._push_bar()
3. live_exchange.wait_fills() → upsert filled orders
4. strategy.save_state() → live_state.db (recovery)
                         + history_db.record_state_snapshot() (replay)
5. print status line
```

### Background sync thread

```
interval: --sync-interval seconds (default 300)
loop:
  acquire lock
  fetch_orders(since=latest_ts) → upsert_order
  fetch_trades(since=latest_ts) → upsert_trades
  release lock
```

A single `threading.Lock` is shared between the main loop and the background thread for all `LiveHistoryDB` writes.

---

## CLI Changes (`__main__.py`)

New `live` argument:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sync-interval` | 300 | Seconds between background exchange syncs |
| `--history-db` | `data/live_history.db` | Path to history database |

`--exchange` added to `live` subcommand (default `binance`) to select connector.

---

## File Structure

```
src/backtest/
├── live_connector.py   ← NEW: BaseExchangeConnector + BinanceConnector
├── live_history.py     ← NEW: LiveHistoryDB
├── live_engine.py      ← UPDATED: connector, bg thread, history writes
├── live_exchange.py    ← UPDATED: takes connector
├── live_feed.py        ← UPDATED: takes connector
└── __main__.py         ← UPDATED: --sync-interval, --history-db, --exchange
```

No changes to backtest engine, strategy, optimizer, or web layer.

---

## Querying History

```bash
# All accounts/strategies
sqlite3 data/live_history.db \
  "SELECT account_id, exchange, strategy, symbol, datetime(ts/1000,'unixepoch') FROM state_snapshots ORDER BY ts DESC LIMIT 20"

# Trade history for an account
sqlite3 data/live_history.db \
  "SELECT datetime(ts/1000,'unixepoch'), side, price, notional, realized_pnl FROM trades WHERE account_id='4f122308952fe175' ORDER BY ts"

# Position snapshots
sqlite3 data/live_history.db \
  "SELECT datetime(ts/1000,'unixepoch'), side, quantity, entry_price, unrealized_pnl, equity FROM positions WHERE symbol='BTCUSDT' ORDER BY ts"
```
