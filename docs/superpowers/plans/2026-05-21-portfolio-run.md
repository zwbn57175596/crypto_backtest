# Portfolio Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `python -m backtest portfolio-run` CLI command that runs the consecutive-reverse strategy simultaneously across multiple USDT perpetual symbols with a shared balance pool and account-level risk controls.

**Architecture:** New `PortfolioBacktestEngine` (independent from `BacktestEngine`) + `MultiSymbolDataFeed` that aligns bars by timestamp union. `SimExchange` reused with zero changes — its `_positions` dict is already keyed by symbol. Reporter reused for the 14 combined metrics; per-symbol stats computed separately in the engine.

**Tech Stack:** Python 3.11+, sqlite3, existing `SimExchange` / `Reporter` / `models.py`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/backtest/multi_data_feed.py` | Yield `(timestamp, {sym: Bar})` aligned across N symbols |
| Create | `src/backtest/portfolio_engine.py` | Main run loop: per-symbol strategy state, risk controls, order dispatch |
| Modify | `src/backtest/__main__.py` | Add `portfolio-run` subcommand + `cmd_portfolio_run()` |
| Create | `tests/test_multi_data_feed.py` | Unit tests for feed alignment |
| Create | `tests/test_portfolio_engine.py` | Unit + integration tests for engine |

---

## Task 1: MultiSymbolDataFeed

**Files:**
- Create: `src/backtest/multi_data_feed.py`
- Create: `tests/test_multi_data_feed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_multi_data_feed.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_multi_data_feed.py -v
```

Expected: `ImportError: cannot import name 'MultiSymbolDataFeed'`

- [ ] **Step 3: Implement MultiSymbolDataFeed**

```python
# src/backtest/multi_data_feed.py
import sqlite3
from collections.abc import Iterator
from backtest.models import Bar


class MultiSymbolDataFeed:
    """
    Yields (timestamp, {symbol: Bar}) groups aligned across N symbols.
    Uses the union of all timestamps — symbols missing at a given timestamp
    are simply absent from the dict for that step.
    """

    def __init__(
        self,
        db_path: str,
        symbols: list[str],
        interval: str,
        exchange: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ):
        self.db_path = db_path
        self.symbols = symbols
        self.interval = interval
        self.exchange = exchange
        self.start_ts = start_ts
        self.end_ts = end_ts

    def __iter__(self) -> Iterator[tuple[int, dict[str, Bar]]]:
        conn = sqlite3.connect(self.db_path)
        placeholders = ",".join("?" * len(self.symbols))
        query = (
            f"SELECT symbol, interval, timestamp, open, high, low, close, volume "
            f"FROM klines "
            f"WHERE symbol IN ({placeholders}) AND interval = ? AND exchange = ?"
        )
        params: list = list(self.symbols) + [self.interval, self.exchange]
        if self.start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(self.start_ts)
        if self.end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(self.end_ts)
        query += " ORDER BY timestamp ASC, symbol ASC"

        current_ts: int | None = None
        current_bars: dict[str, Bar] = {}

        for row in conn.execute(query, params):
            bar = Bar(
                symbol=row[0], interval=row[1], timestamp=row[2],
                open=row[3], high=row[4], low=row[5], close=row[6], volume=row[7],
            )
            if bar.timestamp != current_ts:
                if current_ts is not None:
                    yield current_ts, current_bars
                current_ts = bar.timestamp
                current_bars = {}
            current_bars[bar.symbol] = bar

        if current_ts is not None:
            yield current_ts, current_bars

        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_multi_data_feed.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/multi_data_feed.py tests/test_multi_data_feed.py
git commit -m "feat: add MultiSymbolDataFeed for aligned multi-symbol bar iteration"
```

---

## Task 2: PortfolioBacktestEngine

**Files:**
- Create: `src/backtest/portfolio_engine.py`
- Create: `tests/test_portfolio_engine.py`

### Key design notes

**Wallet balance for sizing** (matches BET live):
```
wallet = sim_exchange.balance + sum(pos.margin for pos in positions.values())
```
This includes locked margins but excludes unrealized PnL, so sizing stays stable as positions open.

**Account risk controls** apply only when opening a NEW symbol slot (pos is None). Close+reopen on the same symbol is always allowed.

**Equity curve deduplication**: `on_new_bar` appends one equity snapshot per symbol call. After run, deduplicate by keeping the last value per timestamp before passing to Reporter.

**Order timing**: Orders submitted in `_process_symbol` at timestamp T are filled at timestamp T+1 by `on_new_bar` (same as single-symbol engine). Close + same-bar reopen are both pending — filled together at T+1.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portfolio_engine.py
import sqlite3
import pytest
from backtest.portfolio_engine import PortfolioBacktestEngine


INTERVAL_MS = 3_600_000  # 1 hour in ms


def _make_db(tmp_path, rows):
    db = str(tmp_path / "klines.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


def _bullish_bars(symbol, n, start_price=100.0):
    """n consecutive bullish 1h bars, each +1 USDT close > open."""
    return [
        ("binance", symbol, "1h", (i + 1) * INTERVAL_MS,
         start_price + i, start_price + i + 1.5, start_price + i - 0.5,
         start_price + i + 1.0, 1000.0)
        for i in range(n)
    ]


@pytest.fixture
def single_sym_7bars(tmp_path):
    """8 bullish bars for AAAUSDT — threshold=7 triggers on bar 7, filled at bar 8."""
    return _make_db(tmp_path, _bullish_bars("AAAUSDT", 8))


@pytest.fixture
def two_sym_db(tmp_path):
    """AAAUSDT: 8 bullish bars. BBBUSDT: 8 bearish bars."""
    bullish = _bullish_bars("AAAUSDT", 8)
    bearish = [
        ("binance", "BBBUSDT", "1h", (i + 1) * INTERVAL_MS,
         200.0 + (7 - i), 200.0 + (7 - i) + 0.5, 200.0 + (7 - i) - 1.5,
         200.0 + (7 - i) - 1.0, 2000.0)
        for i in range(8)
    ]
    return _make_db(tmp_path, bullish + bearish)


def _make_engine(db, symbols, **kwargs):
    defaults = dict(
        interval="1h", exchange="binance",
        balance=1000.0, leverage=10,
        commission_rate=0.0004, funding_rate=0.0001,
        maintenance_margin=0.005,
        consecutive_threshold=7,
        position_multiplier=1.0,
        initial_position_pct=0.1,
        profit_candle_threshold=3,
        max_position_symbols=10,
        margin_ratio_limit=0.9,
    )
    defaults.update(kwargs)
    return PortfolioBacktestEngine(db_path=db, symbols=symbols, **defaults)


# ── Basic result shape ────────────────────────────────────────────

def test_result_has_required_keys(single_sym_7bars):
    engine = _make_engine(single_sym_7bars, ["AAAUSDT"])
    result = engine.run()
    for key in ("trades", "trades_count", "equity_curve", "final_equity",
                "initial_balance", "portfolio_symbols", "per_symbol_stats"):
        assert key in result, f"missing key: {key}"


def test_equity_curve_deduplicated_one_per_ts(single_sym_7bars):
    engine = _make_engine(single_sym_7bars, ["AAAUSDT"])
    result = engine.run()
    tss = [ts for ts, _ in result["equity_curve"]]
    assert len(tss) == len(set(tss)), "duplicate timestamps in equity_curve"


def test_initial_balance_preserved(single_sym_7bars):
    engine = _make_engine(single_sym_7bars, ["AAAUSDT"])
    result = engine.run()
    assert result["initial_balance"] == 1000.0


# ── Strategy logic: consecutive threshold ────────────────────────

def test_no_trades_below_threshold(tmp_path):
    """6 bullish bars with threshold=7 → no orders submitted."""
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 6))
    engine = _make_engine(db, ["AAAUSDT"])
    result = engine.run()
    assert result["trades_count"] == 0


def test_opens_short_after_7_bullish_bars(single_sym_7bars):
    """7 consecutive bullish bars → SHORT opened (filled at bar 8 open)."""
    engine = _make_engine(single_sym_7bars, ["AAAUSDT"])
    result = engine.run()
    # One trade: the short open (pnl=0 for opening trades)
    open_trades = [t for t in result["trades"] if t.pnl == 0.0]
    assert len(open_trades) >= 1
    assert open_trades[0].side == "sell"  # short = sell


# ── Per-symbol stats ─────────────────────────────────────────────

def test_per_symbol_stats_contains_all_symbols(two_sym_db):
    engine = _make_engine(two_sym_db, ["AAAUSDT", "BBBUSDT"])
    result = engine.run()
    assert "AAAUSDT" in result["per_symbol_stats"]
    assert "BBBUSDT" in result["per_symbol_stats"]


def test_per_symbol_stats_structure(two_sym_db):
    engine = _make_engine(two_sym_db, ["AAAUSDT", "BBBUSDT"])
    result = engine.run()
    stats = result["per_symbol_stats"]["AAAUSDT"]
    assert "total_trades" in stats
    assert "win_rate" in stats
    assert "net_pnl" in stats


# ── Account risk control: max_position_symbols ───────────────────

def test_max_position_symbols_limits_new_opens(tmp_path):
    """2 symbols both trigger but max_position_symbols=1 → only 1 position opened."""
    rows = _bullish_bars("AAAUSDT", 8) + _bullish_bars("BBBUSDT", 8, start_price=200.0)
    db = _make_db(tmp_path, rows)
    engine = _make_engine(db, ["AAAUSDT", "BBBUSDT"], max_position_symbols=1)
    result = engine.run()
    open_trades = [t for t in result["trades"] if t.pnl == 0.0]
    assert len(open_trades) <= 1


# ── Wallet balance for sizing ─────────────────────────────────────

def test_final_equity_positive_after_run(single_sym_7bars):
    engine = _make_engine(single_sym_7bars, ["AAAUSDT"])
    result = engine.run()
    assert result["final_equity"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_portfolio_engine.py -v
```

Expected: `ImportError: cannot import name 'PortfolioBacktestEngine'`

- [ ] **Step 3: Implement PortfolioBacktestEngine**

```python
# src/backtest/portfolio_engine.py
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from backtest.exchange import SimExchange
from backtest.models import Bar, Position, Trade
from backtest.multi_data_feed import MultiSymbolDataFeed


class PortfolioBacktestEngine:
    """
    Runs the consecutive-reverse strategy simultaneously across N symbols
    with a shared SimExchange (cross-margin style).

    Position sizing uses wallet balance = cash + sum(locked margins),
    matching BET live behaviour where sizing is decoupled from unrealised PnL.
    """

    def __init__(
        self,
        db_path: str,
        symbols: list[str],
        interval: str,
        exchange: str,
        balance: float = 10000.0,
        leverage: int = 10,
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        consecutive_threshold: int = 7,
        position_multiplier: float = 2.9,
        initial_position_pct: float = 0.05,
        profit_candle_threshold: int = 3,
        max_position_symbols: int = 10,
        margin_ratio_limit: float = 0.8,
        start: str | None = None,
        end: str | None = None,
    ):
        self.db_path = db_path
        self.symbols = symbols
        self.interval = interval
        self.exchange_name = exchange
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.consecutive_threshold = consecutive_threshold
        self.position_multiplier = position_multiplier
        self.initial_position_pct = initial_position_pct
        self.profit_candle_threshold = profit_candle_threshold
        self.max_position_symbols = max_position_symbols
        self.margin_ratio_limit = margin_ratio_limit
        self.start_ts = self._parse_time(start) if start else None
        self.end_ts = self._parse_time(end) if end else None

    @staticmethod
    def _parse_time(s: str) -> int:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def run(self) -> dict:
        sim = SimExchange(
            balance=self.balance,
            leverage=self.leverage,
            commission_rate=self.commission_rate,
            funding_rate=self.funding_rate,
            maintenance_margin=self.maintenance_margin,
            margin_mode="cross",
        )

        states: dict[str, dict] = {
            sym: {"consecutive_count": 0, "streak_direction": 0, "profit_candle_count": 0}
            for sym in self.symbols
        }

        feed = MultiSymbolDataFeed(
            db_path=self.db_path,
            symbols=self.symbols,
            interval=self.interval,
            exchange=self.exchange_name,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
        )

        for _ts, bars in feed:
            for bar in bars.values():
                sim.on_new_bar(bar)
            for sym, bar in bars.items():
                self._process_symbol(sym, bar, states[sym], sim)

        trades = sim.get_trades()

        # Deduplicate equity curve: on_new_bar records one entry per symbol call;
        # keep the last value per timestamp so Reporter sees one point per bar.
        seen: dict[int, float] = {}
        for ts, eq in sim.get_equity_curve():
            seen[ts] = eq
        equity_curve = sorted(seen.items())

        return {
            "trades": trades,
            "trades_count": len(trades),
            "equity_curve": equity_curve,
            "final_equity": sim.equity,
            "initial_balance": self.balance,
            "portfolio_symbols": self.symbols,
            "per_symbol_stats": self._per_symbol_stats(trades),
        }

    # ── Strategy logic ───────────────────────────────────────────

    def _process_symbol(
        self,
        sym: str,
        bar: Bar,
        state: dict,
        sim: SimExchange,
    ) -> None:
        direction = self._bar_direction(bar)
        if direction == 0:
            return

        self._update_streak(state, direction)

        pos: Position | None = sim._positions.get(sym)
        can_trigger = state["consecutive_count"] >= self.consecutive_threshold

        if pos is None:
            state["profit_candle_count"] = 0
            if can_trigger and self._account_allows_new_slot(sim):
                self._submit_open(sym, state, direction, sim)

        elif self._is_profit_candle(pos, direction):
            state["profit_candle_count"] += 1
            if state["profit_candle_count"] >= self.profit_candle_threshold:
                self._submit_close(sym, pos, sim)
                state["profit_candle_count"] = 0
                if can_trigger:
                    self._submit_open(sym, state, direction, sim)

        else:
            self._submit_close(sym, pos, sim)
            state["profit_candle_count"] = 0
            if can_trigger:
                self._submit_open(sym, state, direction, sim)

    def _update_streak(self, state: dict, direction: int) -> None:
        if direction == state["streak_direction"]:
            state["consecutive_count"] += 1
        else:
            state["consecutive_count"] = 1
            state["streak_direction"] = direction

    @staticmethod
    def _bar_direction(bar: Bar) -> int:
        if bar.close > bar.open:
            return 1
        elif bar.close < bar.open:
            return -1
        return 0

    @staticmethod
    def _is_profit_candle(pos: Position, direction: int) -> bool:
        return (pos.side == "long" and direction == 1) or \
               (pos.side == "short" and direction == -1)

    # ── Account risk controls ────────────────────────────────────

    def _account_allows_new_slot(self, sim: SimExchange) -> bool:
        if len(sim._positions) >= self.max_position_symbols:
            return False
        equity = sim.equity
        if equity > 0:
            total_margin = sum(p.margin for p in sim._positions.values())
            if total_margin / equity > self.margin_ratio_limit:
                return False
        return True

    # ── Order helpers ────────────────────────────────────────────

    def _calc_quantity(self, state: dict, sim: SimExchange) -> float:
        wallet = sim.balance + sum(p.margin for p in sim._positions.values())
        n = state["consecutive_count"] - self.consecutive_threshold + 1
        multiplier = self.position_multiplier ** (n - 1)
        return wallet * self.initial_position_pct * multiplier

    def _submit_open(
        self, sym: str, state: dict, direction: int, sim: SimExchange
    ) -> None:
        qty = self._calc_quantity(state, sim)
        if qty <= 0:
            return
        side = "sell" if direction == 1 else "buy"  # reverse: up-streak → short
        sim.submit_order(sym, side, "market", qty)

    @staticmethod
    def _submit_close(sym: str, pos: Position, sim: SimExchange) -> None:
        side = "sell" if pos.side == "long" else "buy"
        sim.submit_order(sym, side, "market", pos.quantity)

    # ── Per-symbol statistics ────────────────────────────────────

    def _per_symbol_stats(self, trades: list[Trade]) -> dict:
        gross_pnl: dict[str, float] = defaultdict(float)
        commissions: dict[str, float] = defaultdict(float)
        closes: dict[str, list[float]] = defaultdict(list)

        for t in trades:
            commissions[t.symbol] += t.commission
            if t.pnl != 0.0:
                gross_pnl[t.symbol] += t.pnl
                closes[t.symbol].append(t.pnl)

        stats = {}
        for sym in self.symbols:
            pnls = closes[sym]
            total = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            stats[sym] = {
                "total_trades": total,
                "win_rate": wins / total if total > 0 else 0.0,
                "net_pnl": gross_pnl[sym] - commissions[sym],
            }
        return stats
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_portfolio_engine.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/portfolio_engine.py tests/test_portfolio_engine.py
git commit -m "feat: add PortfolioBacktestEngine for multi-symbol consecutive-reverse backtest"
```

---

## Task 3: CLI `portfolio-run` subcommand

**Files:**
- Modify: `src/backtest/__main__.py`

Two changes: (1) new `cmd_portfolio_run()` function, (2) new `p_portfolio` subparser + elif handler in `main()`.

- [ ] **Step 1: Add `cmd_portfolio_run` function**

Insert after `cmd_run` (around line 227) and before `cmd_web`:

```python
def cmd_portfolio_run(args: argparse.Namespace) -> None:
    import json
    import sqlite3
    from backtest.portfolio_engine import PortfolioBacktestEngine
    from backtest.reporter import Reporter

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    db_path = args.db or str(Path("data") / "klines.db")

    engine = PortfolioBacktestEngine(
        db_path=db_path,
        symbols=symbols,
        interval=args.interval,
        exchange=args.exchange or "binance",
        balance=args.balance,
        leverage=args.leverage,
        consecutive_threshold=args.consecutive_threshold,
        position_multiplier=args.position_multiplier,
        initial_position_pct=args.initial_position_pct,
        profit_candle_threshold=args.profit_candle_threshold,
        max_position_symbols=args.max_position_symbols,
        margin_ratio_limit=args.margin_ratio_limit,
        start=f"{args.start} 00:00:00" if args.start else None,
        end=f"{args.end} 23:59:59" if args.end else None,
    )

    print(
        f"[组合回测] 标的={len(symbols)} 区间={args.start}~{args.end} "
        f"余额={args.balance} 杠杆={args.leverage}x "
        f"连续阈值={args.consecutive_threshold} 乘数={args.position_multiplier}"
    )

    result = engine.run()
    report = Reporter.generate(result)
    report["portfolio_symbols"] = symbols
    report["per_symbol_stats"] = result["per_symbol_stats"]
    report["run_leverage"] = args.leverage
    report["run_balance"] = args.balance
    report["run_start"] = args.start
    report["run_end"] = args.end

    print("\n=== 组合指标 ===")
    print(f"  net_return:       {report['net_return']:+.2%}")
    print(f"  annual_return:    {report['annual_return']:+.2%}")
    print(f"  max_drawdown:     {report['max_drawdown']:.2%}")
    print(f"  sharpe_ratio:     {report['sharpe_ratio']:.2f}")
    print(f"  sortino_ratio:    {report['sortino_ratio']:.2f}")
    print(f"  win_rate:         {report['win_rate']:.2%}")
    print(f"  profit_factor:    {report['profit_factor']:.2f}")
    print(f"  total_trades:     {report['total_trades']}")
    print(f"  total_commission: {report['total_commission']:.2f} USDT")

    print("\n=== 各标的统计 ===")
    for sym, stats in result["per_symbol_stats"].items():
        print(
            f"  {sym:12s}  交易={stats['total_trades']:>4}  "
            f"胜率={stats['win_rate']:.1%}  "
            f"净PnL={stats['net_pnl']:+.2f} USDT"
        )

    report_db = str(Path(db_path).parent / "reports.db")
    conn = sqlite3.connect(report_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    cur = conn.execute(
        "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) "
        "VALUES (?,?,?,?,?)",
        (
            "PortfolioConsecutiveReverse",
            ",".join(symbols),
            args.interval,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(report),
        ),
    )
    conn.commit()
    report_id = cur.lastrowid
    conn.close()

    print(f"\n报告已存入 reports.db (id={report_id})")
    print("查看: python -m backtest web")
```

- [ ] **Step 2: Add subparser — insert in `main()` after the `p_run` block (around line 530)**

```python
    p_portfolio = sub.add_parser("portfolio-run", help="Run portfolio backtest across multiple symbols")
    p_portfolio.add_argument("--symbols", required=True,
                             help="Comma-separated symbols e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    p_portfolio.add_argument("--interval", required=True)
    p_portfolio.add_argument("--exchange", default="binance")
    p_portfolio.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_portfolio.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_portfolio.add_argument("--balance", type=float, default=10000.0)
    p_portfolio.add_argument("--leverage", type=int, default=10)
    p_portfolio.add_argument("--consecutive-threshold", type=int, default=7,
                             dest="consecutive_threshold")
    p_portfolio.add_argument("--position-multiplier", type=float, default=2.9,
                             dest="position_multiplier")
    p_portfolio.add_argument("--initial-position-pct", type=float, default=0.05,
                             dest="initial_position_pct",
                             help="Size per position as fraction of wallet (default 0.05 = 5%%)")
    p_portfolio.add_argument("--profit-candle-threshold", type=int, default=3,
                             dest="profit_candle_threshold")
    p_portfolio.add_argument("--max-position-symbols", type=int, default=10,
                             dest="max_position_symbols")
    p_portfolio.add_argument("--margin-ratio-limit", type=float, default=0.8,
                             dest="margin_ratio_limit",
                             help="Stop opening new symbols when total_margin/equity exceeds this")
    p_portfolio.add_argument("--db", default=None)
```

- [ ] **Step 3: Add elif handler — insert before `else: parser.print_help()` at the end of `main()`**

```python
    elif args.command == "portfolio-run":
        cmd_portfolio_run(args)
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Smoke-test the CLI help**

```bash
python -m backtest portfolio-run --help
```

Expected output includes `--symbols`, `--interval`, `--consecutive-threshold`, etc.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/__main__.py
git commit -m "feat: add portfolio-run CLI subcommand for multi-symbol backtest"
```

---

## Example Usage

After collecting data for the symbols:

```bash
python -m backtest collect --exchange binance --symbol BTCUSDT --interval 1h \
    --start 2023-01-01 --end 2024-12-31

python -m backtest collect --exchange binance --symbol ETHUSDT --interval 1h \
    --start 2023-01-01 --end 2024-12-31

python -m backtest portfolio-run \
    --symbols BTCUSDT,ETHUSDT \
    --interval 1h \
    --start 2023-01-01 --end 2024-12-31 \
    --balance 10000 --leverage 10 \
    --consecutive-threshold 7 \
    --position-multiplier 2.9 \
    --initial-position-pct 0.05 \
    --profit-candle-threshold 3 \
    --max-position-symbols 10 \
    --margin-ratio-limit 0.8
```
