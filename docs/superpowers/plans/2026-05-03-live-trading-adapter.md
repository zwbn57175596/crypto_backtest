# Live Trading Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic live trading adapter so any `BaseStrategy` subclass can run against the real Binance U本位合约 API with B-level reliability.

**Architecture:** Three new files (`live_exchange.py`, `live_feed.py`, `live_engine.py`) mirror their backtest counterparts (`SimExchange`, `DataFeed`, `BacktestEngine`). `BaseStrategy` receives two optional state hooks (`save_state`/`load_state`). All existing backtest code is untouched.

**Tech Stack:** Python 3.11+, `binance-futures-connector` (`UMFutures`), `unittest.mock` for tests.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/backtest/strategy.py` | Add `save_state` / `load_state` hooks |
| Modify | `strategies/consecutive_reverse.py` | Implement state hooks |
| Create | `src/backtest/live_exchange.py` | Binance API adapter matching SimExchange interface |
| Create | `src/backtest/live_feed.py` | Real-time kline polling + gap backfill |
| Create | `src/backtest/live_engine.py` | Live event loop with reliability logic |
| Modify | `src/backtest/__main__.py` | Add `live` subcommand |
| Modify | `pyproject.toml` | Add `binance-futures-connector` dependency |
| Create | `tests/test_live_exchange.py` | Tests for LiveExchange |
| Create | `tests/test_live_feed.py` | Tests for LiveFeed helpers |
| Create | `tests/test_live_engine.py` | Tests for LiveEngine state logic |

---

## Task 1: BaseStrategy state hooks + ConsecutiveReverse implementation

**Files:**
- Modify: `src/backtest/strategy.py`
- Modify: `strategies/consecutive_reverse.py`
- Create: `tests/test_live_engine.py` (first test only)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_engine.py
from strategies.consecutive_reverse import ConsecutiveReverseStrategy
from unittest.mock import MagicMock
from backtest.models import Position


def _make_exchange():
    ex = MagicMock()
    ex.balance = 1000.0
    ex.equity = 1000.0
    ex.get_position.return_value = None
    return ex


class TestStrategyStateHooks:
    def test_base_strategy_save_state_returns_empty(self):
        from backtest.strategy import BaseStrategy
        strategy = BaseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        assert strategy.save_state() == {}

    def test_base_strategy_load_state_does_nothing(self):
        from backtest.strategy import BaseStrategy
        strategy = BaseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.load_state({"anything": 42})  # must not raise

    def test_consecutive_reverse_save_state(self):
        strategy = ConsecutiveReverseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.on_init()
        strategy._consecutive_count = 5
        strategy._streak_direction = -1
        strategy._profit_candle_count = 2
        state = strategy.save_state()
        assert state == {
            "consecutive_count": 5,
            "streak_direction": -1,
            "profit_candle_count": 2,
        }

    def test_consecutive_reverse_load_state(self):
        strategy = ConsecutiveReverseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.on_init()
        strategy.load_state({"consecutive_count": 7, "streak_direction": 1, "profit_candle_count": 3})
        assert strategy._consecutive_count == 7
        assert strategy._streak_direction == 1
        assert strategy._profit_candle_count == 3

    def test_consecutive_reverse_load_state_missing_keys(self):
        strategy = ConsecutiveReverseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.on_init()
        strategy.load_state({})  # empty state → all defaults
        assert strategy._consecutive_count == 0
        assert strategy._streak_direction == 0
        assert strategy._profit_candle_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_live_engine.py::TestStrategyStateHooks -v
```

Expected: FAIL — `BaseStrategy` has no `save_state` / `load_state`, `ConsecutiveReverseStrategy` has none either.

- [ ] **Step 3: Add hooks to BaseStrategy**

In `src/backtest/strategy.py`, add after the `history` method:

```python
    def save_state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass
```

- [ ] **Step 4: Add implementation to ConsecutiveReverseStrategy**

In `strategies/consecutive_reverse.py`, add after the `_is_profit_candle` method (before the `# ===== 预留方法` section):

```python
    def save_state(self) -> dict:
        return {
            "consecutive_count": self._consecutive_count,
            "streak_direction": self._streak_direction,
            "profit_candle_count": self._profit_candle_count,
        }

    def load_state(self, state: dict) -> None:
        self._consecutive_count = state.get("consecutive_count", 0)
        self._streak_direction = state.get("streak_direction", 0)
        self._profit_candle_count = state.get("profit_candle_count", 0)
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_live_engine.py::TestStrategyStateHooks -v
```

Expected: 5 PASSED.

- [ ] **Step 6: Run full test suite to confirm no regressions**

```
pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/backtest/strategy.py strategies/consecutive_reverse.py tests/test_live_engine.py
git commit -m "feat: add save_state/load_state hooks to BaseStrategy and ConsecutiveReverse"
```

---

## Task 2: LiveExchange — sync, balance, equity, get_position

**Files:**
- Create: `src/backtest/live_exchange.py`
- Create: `tests/test_live_exchange.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_exchange.py
from unittest.mock import MagicMock


def _make_client(
    balance_usdt="1000.00",
    position_amt="0",
    entry_price="0",
    unrealized="0",
    mark_price="50000.0",
    step_size="0.001",
):
    client = MagicMock()
    client.exchange_info.return_value = {
        "symbols": [{"symbol": "BTCUSDT", "filters": [
            {"filterType": "LOT_SIZE", "stepSize": step_size}
        ]}]
    }
    client.balance.return_value = [
        {"asset": "USDT", "availableBalance": balance_usdt, "balance": balance_usdt}
    ]
    client.get_position_risk.return_value = [{
        "positionAmt": position_amt,
        "entryPrice": entry_price,
        "unrealizedProfit": unrealized,
    }]
    client.mark_price.return_value = {"markPrice": mark_price}
    return client


class TestLiveExchangeSync:
    def test_balance_after_sync(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(balance_usdt="2500.00"), "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        assert ex.balance == 2500.0

    def test_no_position_when_flat(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(), "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        assert ex.get_position("BTCUSDT") is None

    def test_long_position_parsed(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(
            _make_client(position_amt="0.1", entry_price="50000", unrealized="200"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.unrealized_pnl == 200.0
        assert abs(pos.quantity - 5000.0) < 0.01   # 0.1 BTC * 50000 USDT/BTC
        assert abs(pos.margin - 500.0) < 0.01       # 5000 / leverage=10

    def test_short_position_parsed(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(
            _make_client(position_amt="-0.05", entry_price="48000", unrealized="-100"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

    def test_equity_no_position(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(balance_usdt="1000"), "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        assert ex.equity == 1000.0

    def test_equity_with_position(self):
        from backtest.live_exchange import LiveExchange
        # 0.1 BTC long @ 50000, unrealized=200, margin=500
        ex = LiveExchange(
            _make_client(balance_usdt="500", position_amt="0.1",
                         entry_price="50000", unrealized="200"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        # equity = balance(500) + margin(500) + unrealized(200) = 1200
        assert abs(ex.equity - 1200.0) < 0.01

    def test_get_position_wrong_symbol_returns_none(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(
            _make_client(position_amt="0.1", entry_price="50000"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        assert ex.get_position("ETHUSDT") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_live_exchange.py::TestLiveExchangeSync -v
```

Expected: FAIL — `live_exchange` module does not exist.

- [ ] **Step 3: Implement LiveExchange core**

Create `src/backtest/live_exchange.py`:

```python
import math
import time
import uuid

from backtest.models import Order, Position

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None


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


class LiveExchange:
    def __init__(self, client, symbol: str, leverage: int,
                 commission_rate: float, dry_run: bool = False):
        self._client = client
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
        info = _retry(lambda: self._client.exchange_info())
        for s in info["symbols"]:
            if s["symbol"] == self._symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        self._lot_step = float(f["stepSize"])
                        return

    def _round_qty(self, qty: float) -> float:
        if self._lot_step <= 0:
            return qty
        precision = max(0, -int(round(math.log10(self._lot_step))))
        return round(qty, precision)

    def sync(self) -> None:
        balances = _retry(lambda: self._client.balance())
        for b in balances:
            if b["asset"] == "USDT":
                self._balance = float(b["availableBalance"])
                break

        positions = _retry(lambda: self._client.get_position_risk(symbol=self._symbol))
        self._position = None
        for p in positions:
            qty = float(p["positionAmt"])
            if abs(qty) < 1e-8:
                continue
            entry_price = float(p["entryPrice"])
            unrealized_pnl = float(p["unrealizedProfit"])
            notional = abs(qty) * entry_price
            margin = notional / self._leverage
            self._position = Position(
                symbol=self._symbol,
                side="long" if qty > 0 else "short",
                quantity=notional,
                entry_price=entry_price,
                leverage=self._leverage,
                unrealized_pnl=unrealized_pnl,
                margin=margin,
            )
            break

        premium = _retry(lambda: self._client.mark_price(symbol=self._symbol))
        self._current_price = float(premium["markPrice"])

    def get_position(self, symbol: str) -> Position | None:
        if self._position and self._position.symbol == symbol:
            return self._position
        return None

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
        raise NotImplementedError("Implemented in Task 3")

    def wait_fills(self, timeout: float = 30.0) -> None:
        raise NotImplementedError("Implemented in Task 3")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_live_exchange.py::TestLiveExchangeSync -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/live_exchange.py tests/test_live_exchange.py
git commit -m "feat: add LiveExchange with sync, balance, equity, get_position"
```

---

## Task 3: LiveExchange — submit_order + wait_fills

**Files:**
- Modify: `src/backtest/live_exchange.py`
- Modify: `tests/test_live_exchange.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_exchange.py`:

```python
class TestLiveExchangeSubmitOrder:
    def test_dry_run_does_not_call_api(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client(mark_price="50000")
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004, dry_run=True)
        ex.sync()
        order = ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        client.new_order.assert_not_called()
        assert order.status == "filled"

    def test_dry_run_prints_log(self, capsys):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(), "BTCUSDT", leverage=10, commission_rate=0.0004, dry_run=True)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        captured = capsys.readouterr()
        assert "dry-run" in captured.out.lower()

    def test_market_order_converts_usdt_to_contracts(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client(mark_price="50000")
        client.new_order.return_value = {"orderId": 1, "status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        kwargs = client.new_order.call_args[1]
        assert kwargs["symbol"] == "BTCUSDT"
        assert kwargs["side"] == "BUY"
        assert kwargs["type"] == "MARKET"
        # 1000 USDT / 50000 price = 0.020 BTC (lot_step=0.001 → 3 decimal places)
        assert kwargs["quantity"] == 0.020

    def test_order_id_tracked_in_pending(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.new_order.return_value = {"orderId": 42, "status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "42" in ex._pending_order_ids

    def test_sell_side_uppercased(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.new_order.return_value = {"orderId": 7, "status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        assert client.new_order.call_args[1]["side"] == "SELL"


class TestLiveExchangeWaitFills:
    def test_clears_pending_when_filled(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.new_order.return_value = {"orderId": 99, "status": "NEW"}
        client.query_order.return_value = {"status": "FILLED"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "99" in ex._pending_order_ids
        ex.wait_fills(timeout=5.0)
        assert ex._pending_order_ids == []

    def test_warns_when_order_never_fills(self, capsys):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.query_order.return_value = {"status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex._pending_order_ids = ["77"]
        ex.wait_fills(timeout=0.01)   # expires immediately
        captured = capsys.readouterr()
        assert "77" in captured.out or "WARN" in captured.out

    def test_no_api_call_when_no_pending(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.wait_fills()
        client.query_order.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_live_exchange.py::TestLiveExchangeSubmitOrder tests/test_live_exchange.py::TestLiveExchangeWaitFills -v
```

Expected: FAIL — `submit_order` and `wait_fills` raise `NotImplementedError`.

- [ ] **Step 3: Implement submit_order and wait_fills**

Replace the two stub methods at the bottom of `LiveExchange` in `src/backtest/live_exchange.py`:

```python
    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None) -> Order:
        order_id = uuid.uuid4().hex[:8]

        if self._dry_run:
            print(f"[dry-run] {side.upper()} {type_} {quantity:.2f} USDT @ {price or 'market'}")
            return Order(
                id=order_id, symbol=symbol, side=side, type=type_,
                quantity=quantity, price=price, status="filled",
                filled_price=self._current_price, filled_at=int(time.time() * 1000),
                commission=quantity * self._commission_rate,
            )

        contract_qty = self._round_qty(quantity / self._current_price)
        if contract_qty <= 0:
            return Order(id=order_id, symbol=symbol, side=side, type=type_,
                         quantity=quantity, price=price, status="canceled")

        params: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET" if type_ == "market" else "LIMIT",
            "quantity": contract_qty,
        }
        if type_ == "limit" and price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"

        resp = _retry(lambda: self._client.new_order(**params))
        binance_id = str(resp["orderId"])
        self._pending_order_ids.append(binance_id)
        return Order(
            id=binance_id, symbol=symbol, side=side, type=type_,
            quantity=quantity, price=price, status="pending",
        )

    def wait_fills(self, timeout: float = 30.0) -> None:
        if not self._pending_order_ids:
            return
        deadline = time.time() + timeout
        remaining = list(self._pending_order_ids)
        while remaining and time.time() < deadline:
            still_pending = []
            for oid in remaining:
                resp = _retry(lambda oid=oid: self._client.query_order(
                    symbol=self._symbol, orderId=int(oid)
                ))
                if resp["status"] not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                    still_pending.append(oid)
            remaining = still_pending
            if remaining:
                time.sleep(0.5)
        if remaining:
            print(f"[WARN] orders not confirmed within {timeout}s: {remaining}")
        self._pending_order_ids.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_live_exchange.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/live_exchange.py tests/test_live_exchange.py
git commit -m "feat: implement LiveExchange submit_order and wait_fills"
```

---

## Task 4: LiveFeed — timing helpers + bar iteration

**Files:**
- Create: `src/backtest/live_feed.py`
- Create: `tests/test_live_feed.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_feed.py
from unittest.mock import MagicMock, patch
from backtest.live_feed import _interval_to_seconds, _bar_close_time, _kline_to_bar


def _make_kline(ts_ms: int, open_=100.0, high=110.0, low=90.0, close=105.0, vol=10.0) -> list:
    close_ts = ts_ms + 3_599_999
    return [ts_ms, str(open_), str(high), str(low), str(close), str(vol),
            close_ts, "0", 0, "0", "0", "0"]


class TestHelpers:
    def test_interval_to_seconds_1h(self):
        assert _interval_to_seconds("1h") == 3600

    def test_interval_to_seconds_4h(self):
        assert _interval_to_seconds("4h") == 14400

    def test_interval_to_seconds_1m(self):
        assert _interval_to_seconds("1m") == 60

    def test_interval_unsupported_raises(self):
        import pytest
        with pytest.raises(ValueError):
            _interval_to_seconds("99x")

    def test_bar_close_time_aligns_to_interval(self):
        # ref_time at 02:00:00 UTC (7200s), interval=4h (14400s)
        close = _bar_close_time(14400, ref_time=7200.0)
        assert close == 14400  # 04:00:00 UTC

    def test_bar_close_time_when_already_past_close(self):
        # ref_time at 05:00:00 UTC (18000s), interval=4h (14400s)
        # last_close = floor(18000/14400)*14400 = 1*14400 = 14400
        # next_close = 14400 + 14400 = 28800 (08:00 UTC)
        close = _bar_close_time(14400, ref_time=18000.0)
        assert close == 28800

    def test_kline_to_bar_fields(self):
        k = _make_kline(1_700_000_000_000)
        bar = _kline_to_bar("BTCUSDT", "1h", k)
        assert bar.symbol == "BTCUSDT"
        assert bar.interval == "1h"
        assert bar.timestamp == 1_700_000_000_000
        assert bar.open == 100.0
        assert bar.high == 110.0
        assert bar.low == 90.0
        assert bar.close == 105.0
        assert bar.volume == 10.0


class TestLiveFeedIteration:
    def test_yields_second_to_last_kline_as_closed_bar(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        client = MagicMock()
        client.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]

        feed = LiveFeed(client, "BTCUSDT", "1h", close_buffer_sec=0)

        # time is well past the bar close, no sleep needed
        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            bar = next(iter(feed))

        assert bar.timestamp == ts1
        assert bar.symbol == "BTCUSDT"

    def test_updates_last_bar_ts_after_yield(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        client = MagicMock()
        client.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]

        feed = LiveFeed(client, "BTCUSDT", "1h", close_buffer_sec=0)

        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            next(iter(feed))

        assert feed._last_bar_ts == ts1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_live_feed.py::TestHelpers tests/test_live_feed.py::TestLiveFeedIteration -v
```

Expected: FAIL — `live_feed` module does not exist.

- [ ] **Step 3: Implement LiveFeed**

Create `src/backtest/live_feed.py`:

```python
import time
from typing import Iterator

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
    def __init__(self, client, symbol: str, interval: str,
                 close_buffer_sec: float = 5.0):
        self._client = client
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

            klines = self._client.klines(
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
        klines = self._client.klines(
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

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_live_feed.py::TestHelpers tests/test_live_feed.py::TestLiveFeedIteration -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/live_feed.py tests/test_live_feed.py
git commit -m "feat: add LiveFeed with timing helpers and bar iteration"
```

---

## Task 5: LiveFeed — gap detection + backfill

**Files:**
- Modify: `tests/test_live_feed.py`

(The `_backfill` method was already implemented in Task 4; this task adds tests to verify gap detection.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_feed.py`:

```python
class TestLiveFeedBackfill:
    def test_backfill_yields_missing_bar(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000  # 1h

        ts1 = ts_base
        ts2 = ts_base + interval_ms   # missing bar
        ts3 = ts_base + 2 * interval_ms  # current bar

        client = MagicMock()
        # _backfill fetches ts2 from history
        client.klines.return_value = [
            _make_kline(ts2, close=200.0),
        ]

        feed = LiveFeed(client, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts1  # simulate: last bar was ts1

        bars = list(feed._backfill(ts2, ts3))
        assert len(bars) == 1
        assert bars[0].timestamp == ts2
        assert bars[0].close == 200.0

    def test_backfill_excludes_current_bar_timestamp(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000

        ts2 = ts_base + interval_ms
        ts3 = ts_base + 2 * interval_ms

        client = MagicMock()
        # Binance returns ts2 AND ts3 — ts3 must be excluded (it's the "current" bar)
        client.klines.return_value = [
            _make_kline(ts2),
            _make_kline(ts3),
        ]

        feed = LiveFeed(client, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts_base

        bars = list(feed._backfill(ts2, ts3))
        assert len(bars) == 1
        assert bars[0].timestamp == ts2

    def test_backfill_updates_last_bar_ts(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000

        ts2 = ts_base + interval_ms
        ts3 = ts_base + 2 * interval_ms

        client = MagicMock()
        client.klines.return_value = [_make_kline(ts2)]

        feed = LiveFeed(client, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts_base

        list(feed._backfill(ts2, ts3))
        assert feed._last_bar_ts == ts2

    def test_no_gap_no_backfill_called(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000

        client = MagicMock()
        client.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]

        feed = LiveFeed(client, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts1  # last bar was ts1, next expected ts2 — no gap

        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            bar = next(iter(feed))

        # klines called once (for the regular poll), not for backfill
        assert client.klines.call_count == 1
        assert bar.timestamp == ts1
```

- [ ] **Step 2: Run tests to verify they pass (already implemented)**

```
pytest tests/test_live_feed.py -v
```

Expected: all PASSED (implementation was in Task 4).

- [ ] **Step 3: Commit**

```bash
git add tests/test_live_feed.py
git commit -m "test: add LiveFeed gap detection and backfill tests"
```

---

## Task 6: LiveEngine — startup + state persistence

**Files:**
- Create: `src/backtest/live_engine.py`
- Modify: `tests/test_live_engine.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_engine.py`:

```python
import json
import os


def _make_live_exchange(balance="1000", position_amt="0"):
    from unittest.mock import MagicMock
    client = MagicMock()
    client.exchange_info.return_value = {"symbols": [{"symbol": "BTCUSDT", "filters": [
        {"filterType": "LOT_SIZE", "stepSize": "0.001"}
    ]}]}
    client.balance.return_value = [{"asset": "USDT", "availableBalance": balance, "balance": balance}]
    client.get_position_risk.return_value = [{
        "positionAmt": position_amt, "entryPrice": "0", "unrealizedProfit": "0"
    }]
    client.mark_price.return_value = {"markPrice": "50000"}
    return client


class TestLiveEngineStatePersistenceIntegration:
    def test_save_state_writes_json_file(self, tmp_path):
        from backtest.live_exchange import LiveExchange
        from backtest.live_engine import LiveEngine
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        client = _make_live_exchange()
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()

        strategy = ConsecutiveReverseStrategy(exchange=ex, symbol="BTCUSDT")
        strategy.on_init()
        strategy._consecutive_count = 4
        strategy._streak_direction = 1
        strategy._profit_candle_count = 1

        engine = LiveEngine(
            ConsecutiveReverseStrategy, "BTCUSDT", "1h",
            leverage=10, state_dir=str(tmp_path)
        )
        engine._save_state(strategy)

        state_file = tmp_path / "BTCUSDT_1h.json"
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert saved["consecutive_count"] == 4
        assert saved["streak_direction"] == 1
        assert saved["profit_candle_count"] == 1

    def test_load_state_restores_strategy_on_restart(self, tmp_path):
        from backtest.live_exchange import LiveExchange
        from backtest.live_engine import LiveEngine
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        state_file = tmp_path / "BTCUSDT_4h.json"
        state_file.write_text(json.dumps({
            "consecutive_count": 6,
            "streak_direction": -1,
            "profit_candle_count": 0,
        }))

        client = _make_live_exchange()
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()

        strategy = ConsecutiveReverseStrategy(exchange=ex, symbol="BTCUSDT")
        strategy.on_init()

        engine = LiveEngine(
            ConsecutiveReverseStrategy, "BTCUSDT", "4h",
            leverage=10, state_dir=str(tmp_path)
        )
        if os.path.exists(engine._state_file):
            with open(engine._state_file) as f:
                strategy.load_state(json.load(f))

        assert strategy._consecutive_count == 6
        assert strategy._streak_direction == -1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_live_engine.py::TestLiveEngineStatePersistenceIntegration -v
```

Expected: FAIL — `live_engine` module does not exist.

- [ ] **Step 3: Create LiveEngine with startup and persistence**

Create `src/backtest/live_engine.py`:

```python
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

from backtest.live_exchange import LiveExchange
from backtest.live_feed import LiveFeed
from backtest.strategy import BaseStrategy

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

_TESTNET_URL = "https://testnet.binancefuture.com"
_MAINNET_URL = "https://fapi.binance.com"


class LiveEngine:
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        symbol: str,
        interval: str,
        leverage: int,
        commission_rate: float = 0.0004,
        api_key: str = "",
        secret: str = "",
        testnet: bool = True,
        dry_run: bool = False,
        state_dir: str = "live_state",
    ):
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.interval = interval
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet
        self.dry_run = dry_run
        self.state_dir = state_dir
        self._state_file = os.path.join(state_dir, f"{symbol}_{interval}.json")

    def run(self) -> None:
        os.makedirs(self.state_dir, exist_ok=True)

        base_url = _TESTNET_URL if self.testnet else _MAINNET_URL
        client = UMFutures(key=self.api_key, secret=self.secret, base_url=base_url)

        try:
            client.change_leverage(symbol=self.symbol, leverage=self.leverage)
        except Exception as e:
            print(f"[WARN] change_leverage: {e}", file=sys.stderr)

        live_exchange = LiveExchange(
            client=client, symbol=self.symbol, leverage=self.leverage,
            commission_rate=self.commission_rate, dry_run=self.dry_run,
        )
        live_exchange.sync()

        strategy = self.strategy_class(exchange=live_exchange, symbol=self.symbol)
        strategy.on_init()

        if os.path.exists(self._state_file):
            with open(self._state_file) as f:
                strategy.load_state(json.load(f))
            print(f"[INFO] Restored state from {self._state_file}")

        self._print_startup_summary(live_exchange)

        def _handle_sigterm(sig, frame):
            self._on_exit(live_exchange)
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle_sigterm)

        feed = LiveFeed(client=client, symbol=self.symbol, interval=self.interval)
        try:
            for bar in feed:
                try:
                    self._process_bar(bar, strategy, live_exchange)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self._alert(f"[ERROR] bar {bar.timestamp}: {e}")
                    traceback.print_exc(file=sys.stderr)
        except KeyboardInterrupt:
            self._on_exit(live_exchange)

    def _process_bar(self, bar, strategy: BaseStrategy, live_exchange: LiveExchange) -> None:
        ts = datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        live_exchange.sync()
        strategy._push_bar(bar)
        live_exchange.wait_fills(timeout=30.0)
        self._save_state(strategy)
        pos = live_exchange.get_position(self.symbol)
        pos_str = f"{pos.side} {pos.quantity:.2f}@{pos.entry_price:.2f}" if pos else "flat"
        print(f"[{ts}] balance={live_exchange.balance:.2f} equity={live_exchange.equity:.2f} pos={pos_str}")

    def _save_state(self, strategy: BaseStrategy) -> None:
        state = strategy.save_state()
        with open(self._state_file, "w") as f:
            json.dump(state, f)

    def _alert(self, msg: str) -> None:
        print(msg, file=sys.stderr)

    def _print_startup_summary(self, live_exchange: LiveExchange) -> None:
        mode = "TESTNET" if self.testnet else "MAINNET"
        dry = " [DRY-RUN]" if self.dry_run else ""
        print(f"\n=== LiveEngine {mode}{dry} ===")
        print(f"Strategy: {self.strategy_class.__name__}  Symbol: {self.symbol}  "
              f"Interval: {self.interval}  Leverage: {self.leverage}x")
        print(f"Balance: {live_exchange.balance:.2f} USDT  Equity: {live_exchange.equity:.2f} USDT")
        pos = live_exchange.get_position(self.symbol)
        if pos:
            print(f"Position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}  "
                  f"PnL: {pos.unrealized_pnl:.2f}")
        else:
            print("Position: flat")
        print()

    def _on_exit(self, live_exchange: LiveExchange) -> None:
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

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_live_engine.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/live_engine.py tests/test_live_engine.py
git commit -m "feat: add LiveEngine with startup sequence and state persistence"
```

---

## Task 7: LiveEngine — process_bar exception handling

**Files:**
- Modify: `tests/test_live_engine.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_engine.py`:

```python
class TestLiveEngineProcessBar:
    def test_process_bar_skips_on_sync_failure(self, capsys):
        from backtest.live_exchange import LiveExchange
        from backtest.live_engine import LiveEngine
        from backtest.models import Bar

        # sync() will fail because balance raises
        client = _make_live_exchange()
        client.balance.side_effect = Exception("Network timeout")

        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        strategy = FixedStrategy(exchange=ex, symbol="BTCUSDT")
        strategy.on_init()

        engine = LiveEngine(FixedStrategy, "BTCUSDT", "1h", leverage=10)
        bar = Bar("BTCUSDT", 1_700_000_000_000, 50000, 51000, 49000, 50500, 100.0, "1h")

        # Must not raise — exception is caught and logged
        engine._process_bar(bar, strategy, ex)

        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_process_bar_saves_state_on_success(self, tmp_path):
        from backtest.live_exchange import LiveExchange
        from backtest.live_engine import LiveEngine
        from backtest.models import Bar

        client = _make_live_exchange()
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()

        strategy = FixedStrategy(exchange=ex, symbol="BTCUSDT")
        strategy.on_init()

        engine = LiveEngine(FixedStrategy, "BTCUSDT", "1h", leverage=10, state_dir=str(tmp_path))
        bar = Bar("BTCUSDT", 1_700_000_000_000, 50000, 51000, 49000, 50500, 100.0, "1h")

        engine._process_bar(bar, strategy, ex)

        state_file = tmp_path / "BTCUSDT_1h.json"
        assert state_file.exists()
        assert json.loads(state_file.read_text()) == {"count": 42}
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_live_engine.py::TestLiveEngineProcessBar -v
```

Expected: FAIL — `_process_bar` does not catch exceptions (they propagate instead of being logged to stderr).

- [ ] **Step 3: Wrap _process_bar in LiveEngine.run() exception handler**

The `_process_bar` method itself does NOT catch exceptions — that's intentional. The outer `run()` loop already wraps calls to `_process_bar`. The tests call `_process_bar` directly, so the exception will propagate.

Update `_process_bar` to catch non-`KeyboardInterrupt` exceptions internally (so it can be tested in isolation):

```python
    def _process_bar(self, bar, strategy: BaseStrategy, live_exchange: LiveExchange) -> None:
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
```

Also remove the inner try/except from `run()` since `_process_bar` now handles it:

```python
        feed = LiveFeed(client=client, symbol=self.symbol, interval=self.interval)
        try:
            for bar in feed:
                self._process_bar(bar, strategy, live_exchange)
        except KeyboardInterrupt:
            self._on_exit(live_exchange)
```

- [ ] **Step 4: Run all engine tests to verify they pass**

```
pytest tests/test_live_engine.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/live_engine.py tests/test_live_engine.py
git commit -m "feat: LiveEngine process_bar catches exceptions, logs to stderr"
```

---

## Task 8: CLI integration + dependency

**Files:**
- Modify: `src/backtest/__main__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add binance-futures-connector to pyproject.toml**

In `pyproject.toml`, update the `dependencies` list:

```toml
dependencies = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "httpx>=0.27",
    "pyyaml>=6.0",
    "binance-futures-connector>=3.3",
]
```

- [ ] **Step 2: Install the new dependency**

```
pip install -e .
```

Expected: `binance-futures-connector` installed successfully.

- [ ] **Step 3: Add cmd_live function to __main__.py**

After the `cmd_optimize` function in `src/backtest/__main__.py`, add:

```python
def cmd_live(args: argparse.Namespace) -> None:
    import os
    from backtest.live_engine import LiveEngine

    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_SECRET", "")
    if not api_key or not secret:
        print("Error: BINANCE_API_KEY and BINANCE_SECRET environment variables must be set.")
        sys.exit(1)

    strategy_class = _load_strategy(args.strategy)
    if args.extra_params:
        _apply_extra_params(strategy_class, args.extra_params)

    engine = LiveEngine(
        strategy_class=strategy_class,
        symbol=args.symbol,
        interval=args.interval,
        leverage=args.leverage,
        commission_rate=args.commission_rate,
        api_key=api_key,
        secret=secret,
        testnet=not args.no_testnet,
        dry_run=args.dry_run,
        state_dir=args.state_dir,
    )
    engine.run()
```

- [ ] **Step 4: Add the live subparser to main()**

Find the `main()` function in `__main__.py` and add the `live` subparser alongside the existing ones. Locate the section where subparsers are added and insert:

```python
    # --- live ---
    live_parser = subparsers.add_parser("live", help="Run strategy in live trading mode")
    live_parser.add_argument("--strategy", required=True, help="Path to strategy .py file")
    live_parser.add_argument("--symbol", required=True)
    live_parser.add_argument("--interval", required=True)
    live_parser.add_argument("--leverage", type=int, required=True)
    live_parser.add_argument("--commission-rate", type=float, default=0.0004, dest="commission_rate")
    live_parser.add_argument("--no-testnet", action="store_true", default=False,
                             help="Use mainnet (default is testnet)")
    live_parser.add_argument("--dry-run", action="store_true", default=False,
                             help="Log orders without sending them")
    live_parser.add_argument("--state-dir", default="live_state", dest="state_dir")
    live_parser.add_argument("extra_params", nargs=argparse.REMAINDER,
                             help="Extra strategy params e.g. --CONSECUTIVE_THRESHOLD 5")
    live_parser.set_defaults(func=cmd_live)
```

- [ ] **Step 5: Verify CLI help works**

```
python -m backtest live --help
```

Expected output includes `--strategy`, `--symbol`, `--interval`, `--leverage`, `--no-testnet`, `--dry-run`.

- [ ] **Step 6: Smoke test dry-run with testnet (requires BINANCE_API_KEY + BINANCE_SECRET)**

If API keys are available:

```bash
export BINANCE_API_KEY=<testnet_key>
export BINANCE_SECRET=<testnet_secret>
python -m backtest live \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --leverage 17 \
    --dry-run
```

Expected: prints startup summary with balance/equity, then waits for next 1h bar.

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/backtest/__main__.py pyproject.toml
git commit -m "feat: add live subcommand to CLI; add binance-futures-connector dependency"
```

---

## Self-Review

**Spec coverage:**
- ✅ `LiveExchange` with `submit_order`, `get_position`, `balance`, `equity`, `sync`, `wait_fills`
- ✅ `LiveFeed` with interval timing, buffer, gap detection, backfill
- ✅ `LiveEngine` startup sequence (sync, leverage, state load, summary)
- ✅ `LiveEngine` main loop with exception handling and state persistence
- ✅ `BaseStrategy.save_state` / `load_state` hooks
- ✅ `ConsecutiveReverseStrategy` state implementation
- ✅ CLI `live` subcommand with `--testnet` default, `--dry-run`, env-var API keys
- ✅ B-level reliability: retry, crash recovery, order confirmation, gap backfill, exception skipping
- ✅ USDT notional ↔ contract quantity conversion
- ✅ `dry_run` mode
- ✅ `binance-futures-connector` dependency added
- ✅ Clean exit on Ctrl+C / SIGTERM with optional position close

**Type consistency:** All method signatures are consistent across tasks. `submit_order` signature matches `SimExchange` and `BaseStrategy` call sites.
