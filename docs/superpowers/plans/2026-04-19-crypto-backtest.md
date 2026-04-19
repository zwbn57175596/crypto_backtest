# Crypto Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an event-driven backtesting engine for USDT-margined perpetual futures with exchange data collection and web-based reporting.

**Architecture:** Event-driven engine where BacktestEngine drives a loop: DataFeed pushes bars one-by-one → Strategy responds via on_bar() → SimExchange handles order matching, position management, margin, fees, funding rate, and liquidation. Reporter collects trades and computes metrics. Web layer is a thin FastAPI server serving a single ECharts HTML page.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx, pyyaml, SQLite, ECharts (CDN), pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, CLI entry point |
| `config/default.yaml` | Default backtest configuration |
| `src/backtest/__init__.py` | Package init, version |
| `src/backtest/__main__.py` | CLI entry: collect, run, web subcommands |
| `src/backtest/models.py` | Dataclasses: Bar, Order, Position, Trade |
| `src/backtest/data_feed.py` | DataFeed: read klines from SQLite, iterate bars |
| `src/backtest/exchange.py` | SimExchange: order matching, position, margin, funding, liquidation |
| `src/backtest/strategy.py` | BaseStrategy: on_bar/on_init, buy/sell/close, property accessors |
| `src/backtest/engine.py` | BacktestEngine: event loop, wiring components |
| `src/backtest/reporter.py` | Compute metrics, generate JSON report, store to SQLite |
| `src/backtest/collector/__init__.py` | Collector package init |
| `src/backtest/collector/base.py` | BaseCollector + SQLite storage logic |
| `src/backtest/collector/binance.py` | BinanceCollector |
| `src/backtest/collector/okx.py` | OkxCollector |
| `src/backtest/collector/htx.py` | HtxCollector |
| `src/backtest/web/__init__.py` | Web package init |
| `src/backtest/web/app.py` | FastAPI app factory |
| `src/backtest/web/routes.py` | API routes: reports list, detail, run backtest |
| `src/backtest/web/static/index.html` | Single-page report UI with ECharts |
| `strategies/example_ma_cross.py` | Example: MA crossover strategy |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_models.py` | Model tests |
| `tests/test_exchange.py` | SimExchange tests |
| `tests/test_data_feed.py` | DataFeed tests |
| `tests/test_engine.py` | Engine integration tests |
| `tests/test_reporter.py` | Reporter tests |
| `tests/test_collector.py` | Collector tests |
| `tests/test_web.py` | Web API tests |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config/default.yaml`
- Create: `src/backtest/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "crypto-backtest"
version = "0.1.0"
description = "USDT-margined perpetual futures backtesting engine"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[project.scripts]
backtest = "backtest.__main__:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create config/default.yaml**

```yaml
backtest:
  initial_balance: 10000
  leverage: 10
  commission_rate: 0.0004
  funding_rate: 0.0001
  maintenance_margin: 0.005
```

- [ ] **Step 3: Create src/backtest/__init__.py**

```python
"""Crypto backtest engine for USDT-margined perpetual futures."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create virtual env and install**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" 2>&1 | tail -5`

Note: If `.[dev]` fails (no dev extras yet), use `pip install -e .` instead. We'll add pytest later.

- [ ] **Step 5: Install pytest**

Run: `pip install pytest pytest-asyncio httpx`

- [ ] **Step 6: Verify import works**

Run: `python -c "import backtest; print(backtest.__version__)"`
Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml config/default.yaml src/backtest/__init__.py
git commit -m "chore: scaffold project with pyproject.toml and config"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/backtest/models.py`
- Create: `tests/conftest.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test for models**

Create `tests/conftest.py`:

```python
import pytest
from backtest.models import Bar, Order, Position, Trade


@pytest.fixture
def sample_bar():
    return Bar(
        symbol="BTCUSDT",
        timestamp=1704067200000,
        open=42000.0,
        high=42500.0,
        low=41800.0,
        close=42300.0,
        volume=1500.0,
        interval="1h",
    )
```

Create `tests/test_models.py`:

```python
from backtest.models import Bar, Order, Position, Trade


def test_bar_creation(sample_bar):
    assert sample_bar.symbol == "BTCUSDT"
    assert sample_bar.close == 42300.0
    assert sample_bar.interval == "1h"


def test_order_defaults():
    order = Order(
        id="o1",
        symbol="BTCUSDT",
        side="buy",
        type="market",
        quantity=1000.0,
    )
    assert order.status == "pending"
    assert order.price is None
    assert order.filled_price == 0.0
    assert order.commission == 0.0


def test_position_unrealized_pnl():
    pos = Position(
        symbol="BTCUSDT",
        side="long",
        quantity=1000.0,
        entry_price=42000.0,
        leverage=10,
    )
    assert pos.unrealized_pnl == 0.0
    assert pos.margin == 0.0


def test_trade_creation():
    trade = Trade(
        id="t1",
        order_id="o1",
        symbol="BTCUSDT",
        side="buy",
        price=42000.0,
        quantity=1000.0,
        pnl=0.0,
        commission=0.4,
        timestamp=1704067200000,
    )
    assert trade.commission == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.models'`

- [ ] **Step 3: Implement models**

Create `src/backtest/models.py`:

```python
from dataclasses import dataclass, field


@dataclass
class Bar:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    type: str  # "market" | "limit"
    quantity: float
    price: float | None = None
    status: str = "pending"
    filled_price: float = 0.0
    filled_at: int = 0
    commission: float = 0.0


@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    quantity: float
    entry_price: float
    leverage: int
    unrealized_pnl: float = 0.0
    margin: float = 0.0


@dataclass
class Trade:
    id: str
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    pnl: float
    commission: float
    timestamp: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/backtest/models.py tests/conftest.py tests/test_models.py
git commit -m "feat: add core data models (Bar, Order, Position, Trade)"
```

---

### Task 3: SimExchange — Order Matching & Position Management

**Files:**
- Create: `src/backtest/exchange.py`
- Create: `tests/test_exchange.py`

- [ ] **Step 1: Write failing tests for SimExchange**

Create `tests/test_exchange.py`:

```python
import pytest
from backtest.exchange import SimExchange
from backtest.models import Bar


@pytest.fixture
def exchange():
    return SimExchange(
        balance=10000.0,
        leverage=10,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.005,
    )


@pytest.fixture
def bar1():
    return Bar(
        symbol="BTCUSDT", timestamp=1704067200000,
        open=42000.0, high=42500.0, low=41800.0, close=42300.0,
        volume=1500.0, interval="1h",
    )


@pytest.fixture
def bar2():
    return Bar(
        symbol="BTCUSDT", timestamp=1704070800000,
        open=42300.0, high=42800.0, low=42100.0, close=42600.0,
        volume=1200.0, interval="1h",
    )


class TestMarketOrder:
    def test_buy_market_opens_long(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)
        assert order.status == "filled"
        assert order.filled_price == 42000.0  # bar.open
        pos = exchange.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "long"
        assert pos.quantity == 1000.0

    def test_sell_market_opens_short(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "sell", "market", 1000.0)
        exchange.on_new_bar(bar1)
        assert order.status == "filled"
        pos = exchange.get_position("BTCUSDT")
        assert pos.side == "short"

    def test_close_long_position(self, exchange, bar1, bar2):
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)
        exchange.submit_order("BTCUSDT", "sell", "market", 1000.0)
        exchange.on_new_bar(bar2)
        assert exchange.get_position("BTCUSDT") is None


class TestLimitOrder:
    def test_limit_buy_fills_when_low_reaches(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "buy", "limit", 1000.0, price=41900.0)
        exchange.on_new_bar(bar1)  # low=41800 <= 41900
        assert order.status == "filled"
        assert order.filled_price == 41900.0

    def test_limit_buy_not_filled_when_low_above(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "buy", "limit", 1000.0, price=41700.0)
        exchange.on_new_bar(bar1)  # low=41800 > 41700
        assert order.status == "pending"

    def test_limit_sell_fills_when_high_reaches(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "sell", "limit", 1000.0, price=42400.0)
        exchange.on_new_bar(bar1)  # high=42500 >= 42400
        assert order.status == "filled"
        assert order.filled_price == 42400.0


class TestCommission:
    def test_commission_deducted(self, exchange, bar1):
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)
        # commission = 1000 * 0.0004 = 0.4
        assert exchange.balance < 10000.0
        trades = exchange.get_trades()
        assert trades[0].commission == pytest.approx(0.4)


class TestUnrealizedPnl:
    def test_long_unrealized_pnl(self, exchange, bar1, bar2):
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)  # entry at 42000
        exchange.on_new_bar(bar2)  # close at 42600
        pos = exchange.get_position("BTCUSDT")
        # pnl = qty * (close - entry) / entry = 1000 * (42600-42000)/42000
        expected = 1000.0 * (42600.0 - 42000.0) / 42000.0
        assert pos.unrealized_pnl == pytest.approx(expected, rel=1e-4)

    def test_short_unrealized_pnl(self, exchange, bar1, bar2):
        exchange.submit_order("BTCUSDT", "sell", "market", 1000.0)
        exchange.on_new_bar(bar1)  # entry at 42000
        exchange.on_new_bar(bar2)  # close at 42600, price went up = loss for short
        pos = exchange.get_position("BTCUSDT")
        expected = 1000.0 * (42000.0 - 42600.0) / 42000.0
        assert pos.unrealized_pnl == pytest.approx(expected, rel=1e-4)


class TestFundingRate:
    def test_funding_settled_at_8h_boundary(self, exchange):
        # 00:00 UTC = funding time
        bar_funding = Bar(
            symbol="BTCUSDT", timestamp=1704096000000,  # 2024-01-01 08:00 UTC
            open=42000.0, high=42100.0, low=41900.0, close=42050.0,
            volume=500.0, interval="1h",
        )
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        # Need a bar to fill the order first
        bar_pre = Bar(
            symbol="BTCUSDT", timestamp=1704092400000,  # 07:00 UTC
            open=42000.0, high=42100.0, low=41900.0, close=42000.0,
            volume=500.0, interval="1h",
        )
        exchange.on_new_bar(bar_pre)
        balance_before = exchange.balance
        exchange.on_new_bar(bar_funding)
        # funding = qty * rate = 1000 * 0.0001 = 0.1 (long pays)
        assert exchange.balance < balance_before


class TestLiquidation:
    def test_long_liquidated_on_large_drop(self, exchange):
        # Use high leverage and big position to trigger liquidation
        exch = SimExchange(
            balance=100.0, leverage=100,
            commission_rate=0.0004, funding_rate=0.0001,
            maintenance_margin=0.005,
        )
        exch.submit_order("BTCUSDT", "buy", "market", 5000.0)
        bar_open = Bar(
            symbol="BTCUSDT", timestamp=1704067200000,
            open=42000.0, high=42000.0, low=41999.0, close=42000.0,
            volume=100.0, interval="1h",
        )
        exch.on_new_bar(bar_open)
        # Price crashes
        bar_crash = Bar(
            symbol="BTCUSDT", timestamp=1704070800000,
            open=42000.0, high=42000.0, low=40000.0, close=40500.0,
            volume=100.0, interval="1h",
        )
        exch.on_new_bar(bar_crash)
        assert exch.get_position("BTCUSDT") is None  # liquidated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exchange.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.exchange'`

- [ ] **Step 3: Implement SimExchange**

Create `src/backtest/exchange.py`:

```python
import uuid
from backtest.models import Bar, Order, Position, Trade

_FUNDING_HOURS = {0, 8, 16}


class SimExchange:
    def __init__(
        self,
        balance: float,
        leverage: int,
        commission_rate: float,
        funding_rate: float,
        maintenance_margin: float,
    ):
        self.initial_balance = balance
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin

        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []
        self._trades: list[Trade] = []
        self._equity_curve: list[tuple[int, float]] = []

    def submit_order(
        self, symbol: str, side: str, type_: str, quantity: float, price: float | None = None
    ) -> Order:
        order = Order(
            id=uuid.uuid4().hex[:8],
            symbol=symbol,
            side=side,
            type=type_,
            quantity=quantity,
            price=price,
        )
        self._pending_orders.append(order)
        return order

    def on_new_bar(self, bar: Bar) -> None:
        self._settle_funding(bar)
        self._match_orders(bar)
        self._update_unrealized_pnl(bar)
        self._check_liquidation(bar)
        self._record_equity(bar)

    def _settle_funding(self, bar: Bar) -> None:
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc)
        if dt.hour not in _FUNDING_HOURS or dt.minute != 0:
            return
        for pos in self._positions.values():
            payment = pos.quantity * self.funding_rate
            if pos.side == "long":
                self.balance -= payment
            else:
                self.balance += payment

    def _match_orders(self, bar: Bar) -> None:
        still_pending = []
        for order in self._pending_orders:
            if order.symbol != bar.symbol:
                still_pending.append(order)
                continue
            filled_price = self._try_fill(order, bar)
            if filled_price is not None:
                self._execute_fill(order, filled_price, bar)
            else:
                still_pending.append(order)
        self._pending_orders = still_pending

    def _try_fill(self, order: Order, bar: Bar) -> float | None:
        if order.type == "market":
            return bar.open
        if order.type == "limit":
            if order.side == "buy" and bar.low <= order.price:
                return order.price
            if order.side == "sell" and bar.high >= order.price:
                return order.price
        return None

    def _execute_fill(self, order: Order, price: float, bar: Bar) -> None:
        order.status = "filled"
        order.filled_price = price
        order.filled_at = bar.timestamp

        commission = order.quantity * self.commission_rate
        order.commission = commission
        self.balance -= commission

        pos = self._positions.get(order.symbol)
        pnl = 0.0

        if pos is None:
            # Open new position
            side = "long" if order.side == "buy" else "short"
            margin = order.quantity / self.leverage
            self.balance -= margin
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                side=side,
                quantity=order.quantity,
                entry_price=price,
                leverage=self.leverage,
                margin=margin,
            )
        elif (pos.side == "long" and order.side == "sell") or (
            pos.side == "short" and order.side == "buy"
        ):
            # Close position (full or partial)
            close_qty = min(order.quantity, pos.quantity)
            if pos.side == "long":
                pnl = close_qty * (price - pos.entry_price) / pos.entry_price
            else:
                pnl = close_qty * (pos.entry_price - price) / pos.entry_price
            self.balance += pnl
            # Return margin proportionally
            margin_returned = pos.margin * (close_qty / pos.quantity)
            self.balance += margin_returned
            pos.margin -= margin_returned

            remaining = pos.quantity - close_qty
            if remaining <= 1e-8:
                del self._positions[order.symbol]
            else:
                pos.quantity = remaining

            # If order qty > position qty, open reverse
            leftover = order.quantity - close_qty
            if leftover > 1e-8:
                side = "long" if order.side == "buy" else "short"
                margin = leftover / self.leverage
                self.balance -= margin
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    side=side,
                    quantity=leftover,
                    entry_price=price,
                    leverage=self.leverage,
                    margin=margin,
                )
        else:
            # Same direction: add to position
            total_qty = pos.quantity + order.quantity
            pos.entry_price = (
                pos.entry_price * pos.quantity + price * order.quantity
            ) / total_qty
            pos.quantity = total_qty
            additional_margin = order.quantity / self.leverage
            self.balance -= additional_margin
            pos.margin += additional_margin

        trade = Trade(
            id=uuid.uuid4().hex[:8],
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=order.quantity,
            pnl=pnl,
            commission=commission,
            timestamp=bar.timestamp,
        )
        self._trades.append(trade)

    def _update_unrealized_pnl(self, bar: Bar) -> None:
        pos = self._positions.get(bar.symbol)
        if pos is None:
            return
        if pos.side == "long":
            pos.unrealized_pnl = pos.quantity * (bar.close - pos.entry_price) / pos.entry_price
        else:
            pos.unrealized_pnl = pos.quantity * (pos.entry_price - bar.close) / pos.entry_price

    def _check_liquidation(self, bar: Bar) -> None:
        pos = self._positions.get(bar.symbol)
        if pos is None:
            return
        equity_in_position = pos.margin + pos.unrealized_pnl
        if equity_in_position <= 0 or (pos.margin / equity_in_position) >= (1 / self.maintenance_margin):
            # Liquidate: position wiped, margin lost
            self._trades.append(Trade(
                id=uuid.uuid4().hex[:8],
                order_id="liquidation",
                symbol=bar.symbol,
                side="sell" if pos.side == "long" else "buy",
                price=bar.close,
                quantity=pos.quantity,
                pnl=-pos.margin,
                commission=0.0,
                timestamp=bar.timestamp,
            ))
            self.balance -= min(pos.margin, self.balance)
            del self._positions[bar.symbol]

    def _record_equity(self, bar: Bar) -> None:
        equity = self.balance
        for pos in self._positions.values():
            equity += pos.margin + pos.unrealized_pnl
        self._equity_curve.append((bar.timestamp, equity))

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def get_trades(self) -> list[Trade]:
        return list(self._trades)

    def get_equity_curve(self) -> list[tuple[int, float]]:
        return list(self._equity_curve)

    @property
    def equity(self) -> float:
        total = self.balance
        for pos in self._positions.values():
            total += pos.margin + pos.unrealized_pnl
        return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_exchange.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/backtest/exchange.py tests/test_exchange.py
git commit -m "feat: add SimExchange with order matching, position, funding, liquidation"
```

---

### Task 4: DataFeed — SQLite Reader

**Files:**
- Create: `src/backtest/data_feed.py`
- Create: `tests/test_data_feed.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_data_feed.py`:

```python
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
    conn.executemany(
        "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return str(path)


def test_iterate_all_bars(db_path):
    feed = DataFeed(
        db_path=db_path, symbol="BTCUSDT", interval="1h",
        exchange="binance",
    )
    bars = list(feed)
    assert len(bars) == 3
    assert all(isinstance(b, Bar) for b in bars)
    assert bars[0].timestamp < bars[1].timestamp < bars[2].timestamp


def test_filter_by_time_range(db_path):
    feed = DataFeed(
        db_path=db_path, symbol="BTCUSDT", interval="1h",
        exchange="binance",
        start_ts=1704070800000,
        end_ts=1704070800000,
    )
    bars = list(feed)
    assert len(bars) == 1
    assert bars[0].timestamp == 1704070800000


def test_empty_result(db_path):
    feed = DataFeed(
        db_path=db_path, symbol="ETHUSDT", interval="1h",
        exchange="binance",
    )
    bars = list(feed)
    assert len(bars) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_feed.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DataFeed**

Create `src/backtest/data_feed.py`:

```python
import sqlite3
from collections.abc import Iterator
from backtest.models import Bar


class DataFeed:
    def __init__(
        self,
        db_path: str,
        symbol: str,
        interval: str,
        exchange: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ):
        self.db_path = db_path
        self.symbol = symbol
        self.interval = interval
        self.exchange = exchange
        self.start_ts = start_ts
        self.end_ts = end_ts

    def __iter__(self) -> Iterator[Bar]:
        conn = sqlite3.connect(self.db_path)
        query = (
            "SELECT symbol, interval, timestamp, open, high, low, close, volume "
            "FROM klines WHERE symbol = ? AND interval = ? AND exchange = ?"
        )
        params: list = [self.symbol, self.interval, self.exchange]

        if self.start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(self.start_ts)
        if self.end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(self.end_ts)

        query += " ORDER BY timestamp ASC"

        cursor = conn.execute(query, params)
        for row in cursor:
            yield Bar(
                symbol=row[0],
                interval=row[1],
                timestamp=row[2],
                open=row[3],
                high=row[4],
                low=row[5],
                close=row[6],
                volume=row[7],
            )
        conn.close()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_data_feed.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/backtest/data_feed.py tests/test_data_feed.py
git commit -m "feat: add DataFeed to read klines from SQLite"
```

---

### Task 5: BaseStrategy

**Files:**
- Create: `src/backtest/strategy.py`

- [ ] **Step 1: Implement BaseStrategy**

Create `src/backtest/strategy.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
from backtest.models import Bar, Position

if TYPE_CHECKING:
    from backtest.exchange import SimExchange


class BaseStrategy:
    def __init__(self, exchange: SimExchange, symbol: str):
        self._exchange = exchange
        self._symbol = symbol
        self._bar_history: list[Bar] = []

    def _push_bar(self, bar: Bar) -> None:
        self._bar_history.append(bar)
        self.on_bar(bar)

    # --- lifecycle callbacks (user overrides) ---
    def on_init(self) -> None:
        pass

    def on_bar(self, bar: Bar) -> None:
        pass

    # --- trading operations ---
    def buy(self, quantity: float, price: float | None = None) -> None:
        type_ = "limit" if price is not None else "market"
        self._exchange.submit_order(self._symbol, "buy", type_, quantity, price)

    def sell(self, quantity: float, price: float | None = None) -> None:
        type_ = "limit" if price is not None else "market"
        self._exchange.submit_order(self._symbol, "sell", type_, quantity, price)

    def close(self) -> None:
        pos = self._exchange.get_position(self._symbol)
        if pos is None:
            return
        side = "sell" if pos.side == "long" else "buy"
        self._exchange.submit_order(self._symbol, side, "market", pos.quantity)

    # --- query ---
    @property
    def position(self) -> Position | None:
        return self._exchange.get_position(self._symbol)

    @property
    def balance(self) -> float:
        return self._exchange.balance

    @property
    def equity(self) -> float:
        return self._exchange.equity

    def history(self, n: int) -> list[Bar]:
        return self._bar_history[-n:]
```

- [ ] **Step 2: Verify import**

Run: `python -c "from backtest.strategy import BaseStrategy; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/backtest/strategy.py
git commit -m "feat: add BaseStrategy with trading API and lifecycle callbacks"
```

---

### Task 6: BacktestEngine — Event Loop

**Files:**
- Create: `src/backtest/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_engine.py`:

```python
import sqlite3
import pytest
from backtest.engine import BacktestEngine
from backtest.strategy import BaseStrategy
from backtest.models import Bar


class BuyAndHoldStrategy(BaseStrategy):
    """Buys on first bar, holds forever."""
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
        db_path=db_path,
        symbol="BTCUSDT",
        interval="1h",
        exchange="binance",
        strategy_class=BuyAndHoldStrategy,
        balance=10000.0,
        leverage=10,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.005,
    )
    result = engine.run()
    assert result["trades_count"] > 0
    assert len(result["equity_curve"]) == 10
    assert result["final_equity"] > 0


def test_engine_with_time_range(db_path):
    engine = BacktestEngine(
        db_path=db_path,
        symbol="BTCUSDT",
        interval="1h",
        exchange="binance",
        strategy_class=BuyAndHoldStrategy,
        balance=10000.0,
        leverage=10,
        start="2024-01-01 01:00:00",
        end="2024-01-01 05:00:00",
    )
    result = engine.run()
    assert len(result["equity_curve"]) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BacktestEngine**

Create `src/backtest/engine.py`:

```python
from datetime import datetime, timezone
from backtest.data_feed import DataFeed
from backtest.exchange import SimExchange
from backtest.strategy import BaseStrategy


class BacktestEngine:
    def __init__(
        self,
        db_path: str,
        symbol: str,
        interval: str,
        exchange: str,
        strategy_class: type[BaseStrategy],
        balance: float = 10000.0,
        leverage: int = 10,
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        start: str | None = None,
        end: str | None = None,
    ):
        self.db_path = db_path
        self.symbol = symbol
        self.interval = interval
        self.exchange_name = exchange
        self.strategy_class = strategy_class
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.start_ts = self._parse_time(start) if start else None
        self.end_ts = self._parse_time(end) if end else None

    @staticmethod
    def _parse_time(s: str) -> int:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def run(self) -> dict:
        sim_exchange = SimExchange(
            balance=self.balance,
            leverage=self.leverage,
            commission_rate=self.commission_rate,
            funding_rate=self.funding_rate,
            maintenance_margin=self.maintenance_margin,
        )
        strategy = self.strategy_class(exchange=sim_exchange, symbol=self.symbol)
        strategy.on_init()

        feed = DataFeed(
            db_path=self.db_path,
            symbol=self.symbol,
            interval=self.interval,
            exchange=self.exchange_name,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
        )

        for bar in feed:
            sim_exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        trades = sim_exchange.get_trades()
        return {
            "trades": trades,
            "trades_count": len(trades),
            "equity_curve": sim_exchange.get_equity_curve(),
            "final_equity": sim_exchange.equity,
            "initial_balance": self.balance,
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_engine.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/backtest/engine.py tests/test_engine.py
git commit -m "feat: add BacktestEngine event loop"
```

---

### Task 7: Reporter — Metrics Calculation

**Files:**
- Create: `src/backtest/reporter.py`
- Create: `tests/test_reporter.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_reporter.py`:

```python
import pytest
from backtest.reporter import Reporter
from backtest.models import Trade


@pytest.fixture
def sample_result():
    """Simulated engine result with trades and equity curve."""
    trades = [
        Trade(id="t1", order_id="o1", symbol="BTCUSDT", side="buy",
              price=42000.0, quantity=1000.0, pnl=0.0, commission=0.4,
              timestamp=1704067200000),
        Trade(id="t2", order_id="o2", symbol="BTCUSDT", side="sell",
              price=42600.0, quantity=1000.0, pnl=14.29, commission=0.4,
              timestamp=1704153600000),
        Trade(id="t3", order_id="o3", symbol="BTCUSDT", side="sell",
              price=42800.0, quantity=1000.0, pnl=0.0, commission=0.4,
              timestamp=1704240000000),
        Trade(id="t4", order_id="o4", symbol="BTCUSDT", side="buy",
              price=43200.0, quantity=1000.0, pnl=-9.35, commission=0.4,
              timestamp=1704326400000),
    ]
    # Equity curve: start 10000, up, down, up
    equity_curve = [
        (1704067200000, 10000.0),
        (1704153600000, 10013.49),
        (1704240000000, 10013.49),
        (1704326400000, 10002.54),
        (1704412800000, 10020.0),
    ]
    return {
        "trades": trades,
        "trades_count": 4,
        "equity_curve": equity_curve,
        "final_equity": 10020.0,
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
    # 2 trades with pnl: t2=+14.29, t4=-9.35 → 1 win / 2 = 0.5
    assert report["win_rate"] == pytest.approx(0.5)


def test_reporter_max_drawdown(sample_result):
    report = Reporter.generate(sample_result)
    # Peak at 10013.49, trough at 10002.54
    # dd = (10013.49 - 10002.54) / 10013.49
    expected_dd = (10013.49 - 10002.54) / 10013.49
    assert report["max_drawdown"] == pytest.approx(expected_dd, rel=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Reporter**

Create `src/backtest/reporter.py`:

```python
import math
from backtest.models import Trade


class Reporter:
    @staticmethod
    def generate(result: dict) -> dict:
        trades: list[Trade] = result["trades"]
        equity_curve: list[tuple[int, float]] = result["equity_curve"]
        initial = result["initial_balance"]
        final = result["final_equity"]

        # Basic
        net_return = (final - initial) / initial if initial else 0.0
        total_commission = sum(t.commission for t in trades)
        total_trades = len(trades)

        # Win rate & profit factor
        closing_trades = [t for t in trades if t.pnl != 0.0]
        wins = [t for t in closing_trades if t.pnl > 0]
        losses = [t for t in closing_trades if t.pnl < 0]
        win_rate = len(wins) / len(closing_trades) if closing_trades else 0.0
        total_profit = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Long/short counts
        long_trades = sum(1 for t in trades if t.side == "buy")
        short_trades = sum(1 for t in trades if t.side == "sell")

        # Max drawdown
        max_dd, max_dd_duration = Reporter._calc_drawdown(equity_curve)

        # Sharpe & Sortino
        returns = Reporter._calc_returns(equity_curve)
        sharpe = Reporter._calc_sharpe(returns)
        sortino = Reporter._calc_sortino(returns)

        # Annual return
        if len(equity_curve) >= 2:
            days = (equity_curve[-1][0] - equity_curve[0][0]) / (1000 * 86400)
            annual_return = (1 + net_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0.0
        else:
            annual_return = 0.0

        # Avg hold time
        avg_hold_time = Reporter._calc_avg_hold_time(trades)

        return {
            "net_return": net_return,
            "annual_return": annual_return,
            "max_drawdown": max_dd,
            "max_dd_duration": max_dd_duration,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "avg_hold_time": avg_hold_time,
            "total_commission": total_commission,
            "total_funding": 0.0,  # tracked separately if needed
            "equity_curve": equity_curve,
            "trades": [
                {
                    "id": t.id, "order_id": t.order_id, "symbol": t.symbol,
                    "side": t.side, "price": t.price, "quantity": t.quantity,
                    "pnl": t.pnl, "commission": t.commission, "timestamp": t.timestamp,
                }
                for t in trades
            ],
        }

    @staticmethod
    def _calc_drawdown(curve: list[tuple[int, float]]) -> tuple[float, int]:
        if not curve:
            return 0.0, 0
        peak = curve[0][1]
        max_dd = 0.0
        dd_start = curve[0][0]
        max_dd_dur = 0
        for ts, eq in curve:
            if eq >= peak:
                peak = eq
                dd_start = ts
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                max_dd_dur = ts - dd_start
        return max_dd, max_dd_dur

    @staticmethod
    def _calc_returns(curve: list[tuple[int, float]]) -> list[float]:
        if len(curve) < 2:
            return []
        return [
            (curve[i][1] - curve[i - 1][1]) / curve[i - 1][1]
            for i in range(1, len(curve))
            if curve[i - 1][1] > 0
        ]

    @staticmethod
    def _calc_sharpe(returns: list[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return (mean - risk_free) * math.sqrt(365 * 24) / std  # hourly → annualized

    @staticmethod
    def _calc_sortino(returns: list[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if mean > 0 else 0.0
        down_var = sum(r ** 2 for r in downside) / len(downside)
        down_std = math.sqrt(down_var)
        if down_std == 0:
            return 0.0
        return (mean - risk_free) * math.sqrt(365 * 24) / down_std

    @staticmethod
    def _calc_avg_hold_time(trades: list[Trade]) -> int:
        """Average hold time in milliseconds between open and close trades."""
        if len(trades) < 2:
            return 0
        # Pair trades: every 2 trades = one round trip
        pairs = []
        for i in range(0, len(trades) - 1, 2):
            pairs.append(trades[i + 1].timestamp - trades[i].timestamp)
        return int(sum(pairs) / len(pairs)) if pairs else 0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_reporter.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/backtest/reporter.py tests/test_reporter.py
git commit -m "feat: add Reporter with metrics calculation"
```

---

### Task 8: Data Collectors — Binance, OKX, HTX

**Files:**
- Create: `src/backtest/collector/__init__.py`
- Create: `src/backtest/collector/base.py`
- Create: `src/backtest/collector/binance.py`
- Create: `src/backtest/collector/okx.py`
- Create: `src/backtest/collector/htx.py`
- Create: `tests/test_collector.py`

- [ ] **Step 1: Write failing test for BaseCollector storage**

Create `tests/test_collector.py`:

```python
import sqlite3
import pytest
from unittest.mock import AsyncMock, patch
from backtest.collector.base import BaseCollector
from backtest.collector.binance import BinanceCollector
from backtest.collector.okx import OkxCollector
from backtest.collector.htx import HtxCollector
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BaseCollector and all exchange collectors**

Create `src/backtest/collector/__init__.py`:

```python
from backtest.collector.binance import BinanceCollector
from backtest.collector.okx import OkxCollector
from backtest.collector.htx import HtxCollector

COLLECTORS = {
    "binance": BinanceCollector,
    "okx": OkxCollector,
    "htx": HtxCollector,
}
```

Create `src/backtest/collector/base.py`:

```python
import sqlite3
from abc import ABC, abstractmethod
from backtest.models import Bar


class BaseCollector(ABC):
    exchange_name: str = ""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                symbol    TEXT,
                interval  TEXT,
                timestamp INTEGER,
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL,
                volume    REAL,
                exchange  TEXT,
                PRIMARY KEY (exchange, symbol, interval, timestamp)
            )
        """)
        conn.commit()
        conn.close()

    def _save_bars(self, bars: list[Bar]) -> None:
        if not bars:
            return
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT OR IGNORE INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (b.symbol, b.interval, b.timestamp, b.open, b.high,
                 b.low, b.close, b.volume, self.exchange_name)
                for b in bars
            ],
        )
        conn.commit()
        conn.close()

    def _get_latest_timestamp(self, symbol: str, interval: str) -> int | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT MAX(timestamp) FROM klines WHERE symbol=? AND interval=? AND exchange=?",
            (symbol, interval, self.exchange_name),
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else None

    @abstractmethod
    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        """Fetch klines from exchange API and save to DB."""
```

Create `src/backtest/collector/binance.py`:

```python
import httpx
from backtest.collector.base import BaseCollector
from backtest.models import Bar

_BASE_URL = "https://fapi.binance.com"
_LIMIT = 1500


class BinanceCollector(BaseCollector):
    exchange_name = "binance"

    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            current = start_ms
            while current < end_ms:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": _LIMIT,
                }
                resp = await client.get(f"{_BASE_URL}/fapi/v1/klines", params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                bars = [
                    Bar(
                        symbol=symbol,
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        interval=interval,
                    )
                    for row in data
                ]
                self._save_bars(bars)
                current = int(data[-1][0]) + 1
                if len(data) < _LIMIT:
                    break
```

Create `src/backtest/collector/okx.py`:

```python
import httpx
from backtest.collector.base import BaseCollector
from backtest.models import Bar

_BASE_URL = "https://www.okx.com"
_LIMIT = 100


def _symbol_to_inst_id(symbol: str) -> str:
    """Convert BTCUSDT → BTC-USDT-SWAP."""
    base = symbol.replace("USDT", "")
    return f"{base}-USDT-SWAP"


class OkxCollector(BaseCollector):
    exchange_name = "okx"

    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        inst_id = _symbol_to_inst_id(symbol)
        okx_bar = self._convert_interval(interval)
        async with httpx.AsyncClient(timeout=30) as client:
            current = end_ms
            while current > start_ms:
                params = {
                    "instId": inst_id,
                    "bar": okx_bar,
                    "before": str(start_ms - 1) if current == end_ms else "",
                    "after": str(current),
                    "limit": str(_LIMIT),
                }
                if current == end_ms:
                    params = {
                        "instId": inst_id,
                        "bar": okx_bar,
                        "after": str(start_ms - 1),
                        "before": str(end_ms + 1),
                        "limit": str(_LIMIT),
                    }
                resp = await client.get(
                    f"{_BASE_URL}/api/v5/market/history-candles", params=params
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                if not data:
                    break
                bars = [
                    Bar(
                        symbol=symbol,
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        interval=interval,
                    )
                    for row in data
                ]
                self._save_bars(bars)
                current = min(int(row[0]) for row in data) - 1
                if len(data) < _LIMIT:
                    break

    @staticmethod
    def _convert_interval(interval: str) -> str:
        mapping = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
        return mapping.get(interval, interval)
```

Create `src/backtest/collector/htx.py`:

```python
import httpx
from backtest.collector.base import BaseCollector
from backtest.models import Bar

_BASE_URL = "https://api.hbdm.com"
_LIMIT = 2000


def _symbol_to_contract(symbol: str) -> str:
    """Convert BTCUSDT → BTC-USDT."""
    base = symbol.replace("USDT", "")
    return f"{base}-USDT"


class HtxCollector(BaseCollector):
    exchange_name = "htx"

    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        contract = _symbol_to_contract(symbol)
        htx_period = self._convert_interval(interval)
        async with httpx.AsyncClient(timeout=30) as client:
            current = start_ms // 1000
            end_sec = end_ms // 1000
            while current < end_sec:
                params = {
                    "contract_code": contract,
                    "period": htx_period,
                    "from": current,
                    "to": end_sec,
                    "size": _LIMIT,
                }
                resp = await client.get(
                    f"{_BASE_URL}/linear-swap-ex/market/history/kline", params=params
                )
                resp.raise_for_status()
                result = resp.json()
                data = result.get("data", [])
                if not data:
                    break
                bars = [
                    Bar(
                        symbol=symbol,
                        timestamp=int(row["id"]) * 1000,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["vol"]),
                        interval=interval,
                    )
                    for row in data
                ]
                self._save_bars(bars)
                current = max(int(row["id"]) for row in data) + 1
                if len(data) < _LIMIT:
                    break

    @staticmethod
    def _convert_interval(interval: str) -> str:
        mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "60min", "4h": "4hour", "1d": "1day"}
        return mapping.get(interval, interval)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_collector.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/backtest/collector/ tests/test_collector.py
git commit -m "feat: add data collectors for Binance, OKX, and HTX"
```

---

### Task 9: CLI Entry Point

**Files:**
- Create: `src/backtest/__main__.py`

- [ ] **Step 1: Implement CLI**

Create `src/backtest/__main__.py`:

```python
import argparse
import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from backtest.engine import BacktestEngine
from backtest.reporter import Reporter
from backtest.strategy import BaseStrategy


def _load_strategy(path: str) -> type[BaseStrategy]:
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in dir(module):
        obj = getattr(module, attr)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseStrategy)
            and obj is not BaseStrategy
        ):
            return obj
    raise ValueError(f"No BaseStrategy subclass found in {path}")


def _parse_date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def cmd_collect(args: argparse.Namespace) -> None:
    from backtest.collector import COLLECTORS

    collector_cls = COLLECTORS.get(args.exchange)
    if collector_cls is None:
        print(f"Unknown exchange: {args.exchange}. Choose from: {list(COLLECTORS.keys())}")
        sys.exit(1)

    db_path = args.db or str(Path("data") / "klines.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    collector = collector_cls(db_path)
    start_ms = _parse_date_to_ms(args.start)
    end_ms = _parse_date_to_ms(args.end)

    print(f"Collecting {args.symbol} {args.interval} from {args.exchange} ...")
    asyncio.run(collector.fetch(args.symbol, args.interval, start_ms, end_ms))
    print("Done.")


def cmd_run(args: argparse.Namespace) -> None:
    import yaml

    config_path = Path("config/default.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f).get("backtest", {})

    strategy_class = _load_strategy(args.strategy)
    db_path = args.db or str(Path("data") / "klines.db")

    engine = BacktestEngine(
        db_path=db_path,
        symbol=args.symbol,
        interval=args.interval,
        exchange=args.exchange or "binance",
        strategy_class=strategy_class,
        balance=args.balance or config.get("initial_balance", 10000),
        leverage=args.leverage or config.get("leverage", 10),
        commission_rate=config.get("commission_rate", 0.0004),
        funding_rate=config.get("funding_rate", 0.0001),
        maintenance_margin=config.get("maintenance_margin", 0.005),
        start=f"{args.start} 00:00:00" if args.start else None,
        end=f"{args.end} 23:59:59" if args.end else None,
    )

    print(f"Running backtest: {strategy_class.__name__} on {args.symbol} {args.interval} ...")
    result = engine.run()
    report = Reporter.generate(result)

    # Save report to SQLite
    import json
    import sqlite3

    report_db = str(Path(db_path).parent / "reports.db")
    conn = sqlite3.connect(report_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    conn.execute(
        "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) VALUES (?,?,?,?,?)",
        (strategy_class.__name__, args.symbol, args.interval,
         datetime.now(timezone.utc).isoformat(), json.dumps(report)),
    )
    conn.commit()
    conn.close()

    print(f"\nBacktest Complete: {strategy_class.__name__}")
    print(f"  Net Return:     {report['net_return']:.2%}")
    print(f"  Max Drawdown:   {report['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:   {report['sharpe_ratio']:.2f}")
    print(f"  Win Rate:       {report['win_rate']:.2%}")
    print(f"  Total Trades:   {report['total_trades']}")
    print(f"  Total Commission: {report['total_commission']:.2f} USDT")
    print(f"\nReport saved. View with: python -m backtest web")


def cmd_web(args: argparse.Namespace) -> None:
    import uvicorn
    from backtest.web.app import create_app

    db_path = args.db or str(Path("data") / "reports.db")
    app = create_app(db_path)
    print(f"Starting web server at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="backtest", description="Crypto futures backtester")
    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Fetch historical klines")
    p_collect.add_argument("--exchange", required=True, choices=["binance", "okx", "htx"])
    p_collect.add_argument("--symbol", required=True)
    p_collect.add_argument("--interval", required=True)
    p_collect.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_collect.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_collect.add_argument("--db", default=None)

    # run
    p_run = sub.add_parser("run", help="Run backtest")
    p_run.add_argument("--strategy", required=True, help="Path to strategy .py file")
    p_run.add_argument("--symbol", required=True)
    p_run.add_argument("--interval", required=True)
    p_run.add_argument("--exchange", default="binance")
    p_run.add_argument("--start", default=None)
    p_run.add_argument("--end", default=None)
    p_run.add_argument("--balance", type=float, default=None)
    p_run.add_argument("--leverage", type=int, default=None)
    p_run.add_argument("--db", default=None)

    # web
    p_web = sub.add_parser("web", help="Start web report viewer")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--db", default=None)

    args = parser.parse_args()
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python -m backtest --help`
Expected: Shows usage with collect, run, web subcommands

- [ ] **Step 3: Commit**

```bash
git add src/backtest/__main__.py
git commit -m "feat: add CLI entry point with collect, run, web subcommands"
```

---

### Task 10: Web Layer — FastAPI + ECharts

**Files:**
- Create: `src/backtest/web/__init__.py`
- Create: `src/backtest/web/app.py`
- Create: `src/backtest/web/routes.py`
- Create: `src/backtest/web/static/index.html`
- Create: `tests/test_web.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_web.py`:

```python
import json
import sqlite3
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backtest.web.app import create_app


@pytest.fixture
def report_db(tmp_path):
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    report = {
        "net_return": 0.05,
        "max_drawdown": 0.02,
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
        "total_trades": 10,
        "equity_curve": [[1704067200000, 10000], [1704070800000, 10500]],
        "trades": [],
    }
    conn.execute(
        "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) VALUES (?,?,?,?,?)",
        ("TestStrategy", "BTCUSDT", "1h", "2024-01-01T00:00:00", json.dumps(report)),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def client(report_db):
    app = create_app(report_db)
    return TestClient(app)


def test_get_reports_list(client):
    resp = client.get("/api/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["strategy"] == "TestStrategy"


def test_get_report_detail(client):
    resp = client.get("/api/reports/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["net_return"] == 0.05


def test_get_report_not_found(client):
    resp = client.get("/api/reports/999")
    assert resp.status_code == 404


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ECharts" in resp.text or "echarts" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web.py -v`
Expected: FAIL

- [ ] **Step 3: Implement web layer**

Create `src/backtest/web/__init__.py`:

```python
```

Create `src/backtest/web/app.py`:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="Crypto Backtest")
    app.state.db_path = db_path

    from backtest.web.routes import router
    app.include_router(router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
```

Create `src/backtest/web/routes.py`:

```python
import json
import sqlite3
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()


def _get_db(request: Request) -> str:
    return request.app.state.db_path


@router.get("/api/reports")
def list_reports(request: Request):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, strategy, symbol, interval, created_at FROM reports ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/api/reports/{report_id}")
def get_report(report_id: int, request: Request):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    report = json.loads(row["report_json"])
    report["id"] = row["id"]
    report["strategy"] = row["strategy"]
    report["symbol"] = row["symbol"]
    report["interval"] = row["interval"]
    report["created_at"] = row["created_at"]
    return report


@router.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Create index.html**

Create `src/backtest/web/static/index.html` — this is a large file, content below:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crypto Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; }
  .header { background: #1a1a2e; padding: 16px 24px; display: flex;
            justify-content: space-between; align-items: center;
            border-bottom: 1px solid #333; }
  .header h1 { font-size: 18px; color: #00d4aa; }
  .header .meta { font-size: 13px; color: #888; }
  .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr);
             gap: 12px; padding: 16px 24px; }
  .kpi { background: #1e293b; border-radius: 8px; padding: 16px; text-align: center; }
  .kpi .label { font-size: 11px; color: #888; text-transform: uppercase; }
  .kpi .value { font-size: 28px; font-weight: bold; margin-top: 4px; }
  .kpi .positive { color: #22c55e; }
  .kpi .negative { color: #ef4444; }
  .kpi .neutral { color: #00d4aa; }
  .charts { padding: 0 24px 24px; }
  .chart-box { background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  .chart-box h3 { font-size: 13px; color: #aaa; margin-bottom: 8px; }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .trade-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .trade-table th { padding: 10px; text-align: left; color: #888;
                    border-bottom: 1px solid #333; }
  .trade-table td { padding: 10px; border-bottom: 1px solid #1a1a2e; }
  .long { color: #22c55e; }
  .short { color: #ef4444; }
  .report-select { padding: 16px 24px; }
  .report-select select { background: #1e293b; color: #e0e0e0; border: 1px solid #333;
                          padding: 8px 12px; border-radius: 6px; font-size: 14px; }
  #loading { text-align: center; padding: 60px; color: #888; }
</style>
</head>
<body>

<div class="header">
  <h1>Crypto Backtest</h1>
  <div class="meta" id="meta-info"></div>
</div>

<div class="report-select">
  <select id="report-selector" onchange="loadReport(this.value)">
    <option value="">Select a report...</option>
  </select>
</div>

<div id="loading">Loading reports...</div>
<div id="content" style="display:none;">
  <div class="kpi-row">
    <div class="kpi"><div class="label">Net Return</div><div class="value" id="kpi-return"></div></div>
    <div class="kpi"><div class="label">Max Drawdown</div><div class="value" id="kpi-dd"></div></div>
    <div class="kpi"><div class="label">Sharpe Ratio</div><div class="value" id="kpi-sharpe"></div></div>
    <div class="kpi"><div class="label">Win Rate</div><div class="value" id="kpi-winrate"></div></div>
  </div>
  <div class="charts">
    <div class="chart-box"><h3>Equity Curve</h3><div id="chart-equity" style="height:300px;"></div></div>
    <div class="chart-row">
      <div class="chart-box"><h3>Drawdown</h3><div id="chart-dd" style="height:200px;"></div></div>
      <div class="chart-box"><h3>Trade PnL</h3><div id="chart-pnl" style="height:200px;"></div></div>
    </div>
    <div class="chart-box">
      <h3>Trade Log</h3>
      <table class="trade-table">
        <thead><tr><th>#</th><th>Time</th><th>Side</th><th>Price</th><th>Quantity</th><th>PnL</th><th>Fee</th></tr></thead>
        <tbody id="trade-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const eqChart = echarts.init(document.getElementById('chart-equity'));
const ddChart = echarts.init(document.getElementById('chart-dd'));
const pnlChart = echarts.init(document.getElementById('chart-pnl'));
window.addEventListener('resize', () => { eqChart.resize(); ddChart.resize(); pnlChart.resize(); });

fetch('/api/reports').then(r => r.json()).then(reports => {
  const sel = document.getElementById('report-selector');
  reports.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.id;
    opt.textContent = `#${r.id} ${r.strategy} | ${r.symbol} ${r.interval} | ${r.created_at.slice(0,19)}`;
    sel.appendChild(opt);
  });
  document.getElementById('loading').textContent = reports.length ? 'Select a report above' : 'No reports yet. Run a backtest first.';
  if (reports.length) { sel.value = reports[0].id; loadReport(reports[0].id); }
});

function loadReport(id) {
  if (!id) return;
  fetch(`/api/reports/${id}`).then(r => r.json()).then(d => {
    document.getElementById('content').style.display = '';
    document.getElementById('loading').style.display = 'none';
    document.getElementById('meta-info').textContent =
      `${d.strategy} | ${d.symbol} ${d.interval} | ${d.created_at.slice(0,19)}`;

    const retEl = document.getElementById('kpi-return');
    retEl.textContent = (d.net_return * 100).toFixed(2) + '%';
    retEl.className = 'value ' + (d.net_return >= 0 ? 'positive' : 'negative');

    const ddEl = document.getElementById('kpi-dd');
    ddEl.textContent = (d.max_drawdown * 100).toFixed(2) + '%';
    ddEl.className = 'value negative';

    const shEl = document.getElementById('kpi-sharpe');
    shEl.textContent = d.sharpe_ratio.toFixed(2);
    shEl.className = 'value neutral';

    const wrEl = document.getElementById('kpi-winrate');
    wrEl.textContent = (d.win_rate * 100).toFixed(1) + '%';
    wrEl.className = 'value ' + (d.win_rate >= 0.5 ? 'positive' : 'negative');

    // Equity curve
    const dates = d.equity_curve.map(p => new Date(p[0]).toLocaleDateString());
    const vals = d.equity_curve.map(p => p[1]);
    eqChart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: dates, axisLabel: { color: '#888' }, axisLine: { lineStyle: { color: '#333' }}},
      yAxis: { type: 'value', axisLabel: { color: '#888' }, splitLine: { lineStyle: { color: '#222' }}},
      series: [{ data: vals, type: 'line', smooth: true, lineStyle: { color: '#22c55e' },
                 areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,
                   [{offset:0,color:'rgba(34,197,94,0.3)'},{offset:1,color:'rgba(34,197,94,0.02)'}])}}],
      grid: { left: 60, right: 20, top: 10, bottom: 30 },
    });

    // Drawdown
    let peak = vals[0]; const ddVals = vals.map(v => { if(v>peak) peak=v; return -((peak-v)/peak)*100; });
    ddChart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: dates, show: false },
      yAxis: { type: 'value', axisLabel: { color: '#888', formatter: '{value}%' }, splitLine: { lineStyle: { color: '#222' }}},
      series: [{ data: ddVals, type: 'line', areaStyle: { color: 'rgba(239,68,68,0.15)' },
                 lineStyle: { color: '#ef4444' }}],
      grid: { left: 50, right: 10, top: 10, bottom: 10 },
    });

    // Trade PnL bars
    const closingTrades = (d.trades || []).filter(t => t.pnl !== 0);
    pnlChart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: closingTrades.map((_,i) => i+1), axisLabel: { color: '#888' }},
      yAxis: { type: 'value', axisLabel: { color: '#888' }, splitLine: { lineStyle: { color: '#222' }}},
      series: [{ data: closingTrades.map(t => ({
        value: t.pnl, itemStyle: { color: t.pnl >= 0 ? '#22c55e' : '#ef4444' }
      })), type: 'bar' }],
      grid: { left: 50, right: 10, top: 10, bottom: 30 },
    });

    // Trade table
    const tbody = document.getElementById('trade-body');
    tbody.innerHTML = '';
    (d.trades || []).forEach((t, i) => {
      const tr = document.createElement('tr');
      const sideClass = t.side === 'buy' ? 'long' : 'short';
      const pnlClass = t.pnl > 0 ? 'long' : t.pnl < 0 ? 'short' : '';
      tr.innerHTML = `<td>${i+1}</td><td>${new Date(t.timestamp).toLocaleString()}</td>
        <td class="${sideClass}">${t.side.toUpperCase()}</td>
        <td>${t.price.toFixed(2)}</td><td>${t.quantity.toFixed(2)}</td>
        <td class="${pnlClass}">${t.pnl.toFixed(2)}</td><td>${t.commission.toFixed(2)}</td>`;
      tbody.appendChild(tr);
    });
  });
}
</script>
</body>
</html>
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_web.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/backtest/web/ tests/test_web.py
git commit -m "feat: add FastAPI web layer with ECharts report page"
```

---

### Task 11: Example Strategy — MA Crossover

**Files:**
- Create: `strategies/example_ma_cross.py`

- [ ] **Step 1: Create example strategy**

Create `strategies/example_ma_cross.py`:

```python
"""
Example strategy: Moving Average Crossover

Buy when short MA crosses above long MA.
Sell when short MA crosses below long MA.
"""

from backtest.strategy import BaseStrategy
from backtest.models import Bar


class MaCrossStrategy(BaseStrategy):
    short_period = 7
    long_period = 25
    trade_quantity = 1000.0  # USDT per trade

    def on_bar(self, bar: Bar) -> None:
        bars = self.history(self.long_period)
        if len(bars) < self.long_period:
            return

        short_ma = sum(b.close for b in bars[-self.short_period:]) / self.short_period
        long_ma = sum(b.close for b in bars) / self.long_period

        prev_bars = self.history(self.long_period + 1)
        if len(prev_bars) < self.long_period + 1:
            return
        prev_short = sum(b.close for b in prev_bars[-self.short_period - 1:-1]) / self.short_period
        prev_long = sum(b.close for b in prev_bars[:-1]) / self.long_period

        pos = self.position

        # Golden cross: short MA crosses above long MA → buy
        if prev_short <= prev_long and short_ma > long_ma:
            if pos is None:
                self.buy(self.trade_quantity)
            elif pos.side == "short":
                self.close()
                self.buy(self.trade_quantity)

        # Death cross: short MA crosses below long MA → sell
        elif prev_short >= prev_long and short_ma < long_ma:
            if pos is None:
                self.sell(self.trade_quantity)
            elif pos.side == "long":
                self.close()
                self.sell(self.trade_quantity)
```

- [ ] **Step 2: Verify import**

Run: `python -c "from strategies.example_ma_cross import MaCrossStrategy; print(MaCrossStrategy.__name__)"`

Note: This may fail due to path issues. That's OK — the CLI uses `importlib.util.spec_from_file_location` which handles this.

- [ ] **Step 3: Commit**

```bash
git add strategies/example_ma_cross.py
git commit -m "feat: add MA crossover example strategy"
```

---

### Task 12: Create data/ directory and final integration verification

**Files:**
- Create: `data/.gitkeep`

- [ ] **Step 1: Create data directory**

```bash
mkdir -p data
touch data/.gitkeep
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Verify CLI end-to-end**

Run: `python -m backtest --help`
Expected: Shows help with collect, run, web subcommands

- [ ] **Step 4: Commit**

```bash
git add data/.gitkeep
git commit -m "chore: add data directory and verify full test suite"
```
