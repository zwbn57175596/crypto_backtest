# Funding Carry — Phase 1 (Research) + Phase 2 (Viability) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify OKX account/market facts and decide with historical data whether spot-long + perp-short funding carry (optionally 1.5x levered) is worth building, before any execution code exists.

**Architecture:** Small pure modules in `src/carry/` (funding math, carry simulator, funding-history fetch/store, read-only OKX probe) plus three thin scripts in `scripts/`. Network access is injected (fake-able in tests). No order placement anywhere in this plan.

**Tech Stack:** Python 3.11+, stdlib `sqlite3`, `httpx` (Binance funding history), `python-okx` (read-only probes), pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-funding-carry-design.md`

**Out of scope (later plans):** mgnRatio shock-simulation/threshold calibration (needs the mgnRatio scale found in Task 4/6), signal/risk/watchdog/runner, FakeOKX, demo integration, live.

## Global Constraints

- Python >= 3.11, type hints on all signatures, `from __future__ import annotations`.
- Synchronous `sqlite3` only; HTTP via `httpx` only.
- Scope: BTC and ETH only (`BTC-USDT-SWAP`, `ETH-USDT-SWAP`, `BTC-USDT`, `ETH-USDT`; Binance `BTCUSDT`, `ETHUSDT`).
- Funding interval assumed 8h (`PERIODS_PER_YEAR = 365 * 3`); loader must verify the median interval is 8h.
- Default fee assumption: taker, 0.15% per side (spot 0.10% + swap 0.05%) → 0.30% round trip; maker scenario 0.10% per side. Replace with the real tier found by the probe.
- Research code is READ-ONLY: no order/trade/transfer/borrow API calls. API key must be created read-only. Credentials only from env: `OKX_API_KEY`, `OKX_API_SECRET`, `OKX_API_PASSPHRASE`; never write them to files or logs.
- Do not commit `data/carry.db` or any secrets.
- Tests: worktree has no venv. Use the main repo's interpreter:
  `export PY=/home/airwolf/Github/crypto_backtest/.venv/bin/python` and run `$PY -m pytest tests/<file> -v` from the worktree root.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`

## File Structure

| File | Responsibility |
|---|---|
| `src/carry/__init__.py` | package marker |
| `src/carry/funding_math.py` | APR conversions, net-carry formula, leverage break-even |
| `src/carry/simulate.py` | `CarryParams`, `CarryResult`, `simulate_carry` (event-driven over funding settlements) |
| `src/carry/funding_history.py` | paginated OKX/Binance fetchers (injectable), SQLite save/load, interval check |
| `src/carry/okx_probe.py` | `collect_probe` (read-only facts from injected API objects), `render_markdown` |
| `scripts/carry_fetch_funding.py` | download BTC/ETH funding history into `data/carry.db` |
| `scripts/carry_research_probe.py` | run the probe against real OKX (needs read-only key) → markdown |
| `scripts/carry_viability.py` | scenario grid + per-year breakdown → markdown tables |
| `tests/test_carry_funding_math.py`, `test_carry_simulate.py`, `test_carry_funding_history.py`, `test_carry_okx_probe.py` | unit tests |
| `docs/superpowers/specs/2026-09-19-funding-carry-research.md` | research findings + go/no-go (Task 6) |

---

### Task 1: Funding math

**Files:**
- Create: `src/carry/__init__.py` (empty)
- Create: `src/carry/funding_math.py`
- Test: `tests/test_carry_funding_math.py`

**Interfaces:**
- Produces: `PERIODS_PER_YEAR: int`, `funding_apr(rate_per_period: float, periods_per_year: int = PERIODS_PER_YEAR) -> float`, `borrow_cost_per_period(borrow_apr: float, periods_per_year: int = PERIODS_PER_YEAR) -> float`, `fee_apr(roundtrip_fee: float, hold_days: float) -> float`, `net_carry_apr(funding_apr_: float, borrow_apr: float, leverage: float, roundtrip_fee: float, hold_days: float) -> float`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from carry.funding_math import (
    PERIODS_PER_YEAR, funding_apr, borrow_cost_per_period, fee_apr, net_carry_apr,
)


def test_funding_apr_base_rate():
    assert PERIODS_PER_YEAR == 1095
    assert funding_apr(0.0001) == pytest.approx(0.1095)


def test_borrow_cost_per_period():
    assert borrow_cost_per_period(0.0876) == pytest.approx(0.0876 / 1095)


def test_fee_apr_amortised():
    assert fee_apr(0.003, 30) == pytest.approx(0.0365)


def test_net_carry_levered_vs_unlevered():
    lev = net_carry_apr(0.10, 0.06, 1.5, 0.003, 30)
    flat = net_carry_apr(0.10, 0.06, 1.0, 0.003, 30)
    assert flat == pytest.approx(0.0635)
    assert lev == pytest.approx(0.06525)
    # extra leverage adds (leverage-1) * (funding - borrow - fee_apr)
    assert lev - flat == pytest.approx(0.5 * (0.10 - 0.06 - 0.0365))


def test_leverage_hurts_when_spread_below_fees():
    lev = net_carry_apr(0.08, 0.06, 1.5, 0.003, 30)
    flat = net_carry_apr(0.08, 0.06, 1.0, 0.003, 30)
    assert lev < flat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_funding_math.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry'`

- [ ] **Step 3: Write minimal implementation**

`src/carry/__init__.py` is an empty file. `src/carry/funding_math.py`:

```python
"""Pure funding-carry arithmetic. All rates are fractions (0.10 = 10%)."""
from __future__ import annotations

PERIODS_PER_YEAR = 365 * 3  # 8h funding settlements


def funding_apr(rate_per_period: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    return rate_per_period * periods_per_year


def borrow_cost_per_period(borrow_apr: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    return borrow_apr / periods_per_year


def fee_apr(roundtrip_fee: float, hold_days: float) -> float:
    """Round-trip fee (all four legs) amortised over the expected holding time."""
    return roundtrip_fee * 365 / hold_days


def net_carry_apr(funding_apr_: float, borrow_apr: float, leverage: float,
                  roundtrip_fee: float, hold_days: float) -> float:
    """Net APR on capital. Notional per leg = leverage * capital; borrowed = (leverage-1) * capital."""
    return (leverage * funding_apr_
            - (leverage - 1.0) * borrow_apr
            - leverage * fee_apr(roundtrip_fee, hold_days))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_funding_math.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/__init__.py src/carry/funding_math.py tests/test_carry_funding_math.py
git commit -m "feat(carry): funding-carry arithmetic helpers

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Carry simulator

**Files:**
- Create: `src/carry/simulate.py`
- Test: `tests/test_carry_simulate.py`

**Interfaces:**
- Consumes: `funding_apr`, `borrow_cost_per_period`, `PERIODS_PER_YEAR` from `carry.funding_math`
- Produces: `CarryParams` (frozen dataclass: `window=9, enter_apr=0.06, exit_apr=0.02, max_leverage=1.5, lev_min_spread=0.03, borrow_apr=0.08, fee_side=0.0015, basis_slip=0.0002, min_hold=9`), `CarryResult` (dataclass: `net_apr, total_pnl, max_drawdown, time_in_market, round_trips, avg_leverage, years`), `simulate_carry(rates: Sequence[float], p: CarryParams) -> CarryResult`

Semantics (one loop step = one funding settlement `i`): the decision uses only `rates[i-window:i]` (no lookahead), position exists before settlement `i` and earns `lev*rates[i] - (lev-1)*borrow_per_period`. Cost per unit of notional traded = `fee_side + basis_slip` (covers both legs, one side). Equity is additive in fractions of initial capital.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from carry.simulate import CarryParams, simulate_carry

FLAT = dict(window=9, enter_apr=0.08, exit_apr=0.03, max_leverage=1.0,
            fee_side=0.0, basis_slip=0.0, min_hold=0, borrow_apr=0.08)


def test_constant_positive_funding_earns_funding():
    rates = [0.0001] * (9 + 1095)          # one year after warm-up
    r = simulate_carry(rates, CarryParams(**FLAT))
    assert r.years == pytest.approx(1.0)
    assert r.net_apr == pytest.approx(0.1095)
    assert r.time_in_market == pytest.approx(1.0)
    assert r.round_trips == 1
    assert r.max_drawdown == pytest.approx(0.0)


def test_entry_cost_is_charged_once():
    rates = [0.0001] * (9 + 1095)
    r = simulate_carry(rates, CarryParams(**{**FLAT, "fee_side": 0.0015, "basis_slip": 0.0002}))
    assert r.total_pnl == pytest.approx(0.1095 - 0.0017)


def test_leverage_earns_spread_on_borrowed_part():
    rates = [0.0001] * (9 + 1095)
    p = CarryParams(**{**FLAT, "max_leverage": 1.5, "borrow_apr": 0.05, "lev_min_spread": 0.04})
    r = simulate_carry(rates, p)
    assert r.avg_leverage == pytest.approx(1.5)
    assert r.net_apr == pytest.approx(1.5 * 0.1095 - 0.5 * 0.05)


def test_leverage_not_used_when_spread_too_thin():
    rates = [0.0001] * (9 + 1095)
    p = CarryParams(**{**FLAT, "max_leverage": 1.5, "borrow_apr": 0.09, "lev_min_spread": 0.04})
    assert simulate_carry(rates, p).avg_leverage == pytest.approx(1.0)


def test_negative_funding_never_enters():
    r = simulate_carry([-0.0001] * 100, CarryParams(**FLAT))
    assert r.round_trips == 0 and r.total_pnl == 0.0 and r.time_in_market == 0.0


def test_exits_when_trailing_funding_fades():
    rates = [0.0001] * 10 + [0.0] * 10
    p = CarryParams(**{**FLAT, "window": 3})
    r = simulate_carry(rates, p)
    assert r.round_trips == 1
    assert r.total_pnl == pytest.approx(0.0007)
    assert r.time_in_market == pytest.approx(10 / 17)


def test_min_hold_delays_exit():
    rates = [0.0001] * 10 + [0.0] * 10
    p = CarryParams(**{**FLAT, "window": 3, "min_hold": 20})
    assert simulate_carry(rates, p).time_in_market == pytest.approx(1.0)


def test_max_drawdown_measured_from_equity_peak():
    p = CarryParams(**{**FLAT, "window": 1, "enter_apr": 0.05, "exit_apr": -10.0})
    r = simulate_carry([0.0001, 0.0001, -0.001, 0.0001], p)
    assert r.max_drawdown == pytest.approx(0.001)


def test_too_short_series_returns_zeros():
    r = simulate_carry([0.0001] * 5, CarryParams(**FLAT))
    assert r.net_apr == 0.0 and r.years == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.simulate'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Event-driven carry simulator over funding settlements (spot long + perp short)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from carry.funding_math import PERIODS_PER_YEAR, borrow_cost_per_period, funding_apr


@dataclass(frozen=True)
class CarryParams:
    window: int = 9               # trailing settlements used for the signal (9 = 3 days)
    enter_apr: float = 0.06       # enter when trailing funding APR >= this
    exit_apr: float = 0.02        # exit when it falls below this (hysteresis)
    max_leverage: float = 1.5
    lev_min_spread: float = 0.03  # use max_leverage only if funding_apr - borrow_apr >= this
    borrow_apr: float = 0.08
    fee_side: float = 0.0015      # per unit notional traded, both legs, one side
    basis_slip: float = 0.0002    # per unit notional traded
    min_hold: int = 9             # settlements before a fade-exit is allowed


@dataclass
class CarryResult:
    net_apr: float
    total_pnl: float
    max_drawdown: float
    time_in_market: float
    round_trips: int
    avg_leverage: float
    years: float


def _pick_leverage(avg_apr: float, p: CarryParams) -> float:
    return p.max_leverage if avg_apr - p.borrow_apr >= p.lev_min_spread else 1.0


def simulate_carry(rates: Sequence[float], p: CarryParams) -> CarryResult:
    n = len(rates)
    if n <= p.window:
        return CarryResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    borrow_pp = borrow_cost_per_period(p.borrow_apr)
    cost_per_unit = p.fee_side + p.basis_slip
    equity = peak = max_dd = 0.0
    lev = 0.0
    held = in_mkt = trips = 0
    lev_sum = 0.0

    for i in range(p.window, n):
        avg_apr = funding_apr(sum(rates[i - p.window:i]) / p.window)
        target = lev
        if lev == 0.0:
            if avg_apr >= p.enter_apr:
                target = _pick_leverage(avg_apr, p)
        elif avg_apr >= p.exit_apr:
            target = _pick_leverage(avg_apr, p)
        elif held >= p.min_hold:
            target = 0.0

        equity -= abs(target - lev) * cost_per_unit
        if lev == 0.0 and target > 0.0:
            trips += 1
            held = 0
        lev = target

        if lev > 0.0:
            equity += lev * rates[i] - (lev - 1.0) * borrow_pp
            held += 1
            in_mkt += 1
            lev_sum += lev
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    events = n - p.window
    years = events / PERIODS_PER_YEAR
    return CarryResult(
        net_apr=equity / years,
        total_pnl=equity,
        max_drawdown=max_dd,
        time_in_market=in_mkt / events,
        round_trips=trips,
        avg_leverage=lev_sum / in_mkt if in_mkt else 0.0,
        years=years,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_simulate.py -v`
Expected: 9 passed. If `test_exits_when_trailing_funding_fades` fails, hand-trace the loop against the comment in the test (entry at i=3, last earning settlement i=12, exit at i=13) before touching the code.

- [ ] **Step 5: Commit**

```bash
git add src/carry/simulate.py tests/test_carry_simulate.py
git commit -m "feat(carry): event-driven funding carry simulator

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Funding history fetch + store

**Files:**
- Create: `src/carry/funding_history.py`
- Create: `scripts/carry_fetch_funding.py`
- Test: `tests/test_carry_funding_history.py`

**Interfaces:**
- Produces: `Rows = list[tuple[int, float]]` (ascending `(funding_time_ms, rate)`); `fetch_okx_history(fetch_page: Callable[[str, int | None], list[dict]], inst_id: str, max_pages: int = 500) -> Rows`; `fetch_binance_history(fetch_page: Callable[[str, int], list[dict]], symbol: str, start_ms: int, end_ms: int, max_pages: int = 500) -> Rows`; `make_okx_page_fetcher(public_api, pause: float = 0.25)`; `make_binance_page_fetcher(pause: float = 0.2)`; `save_rates(db_path, exchange: str, symbol: str, rows: Rows) -> None`; `load_rates(db_path, exchange: str, symbol: str) -> Rows`; `median_interval_hours(rows: Rows) -> float`

- [ ] **Step 1: Write the failing test**

```python
from carry.funding_history import (
    fetch_okx_history, fetch_binance_history, save_rates, load_rates, median_interval_hours,
)

H8 = 8 * 3600 * 1000


def _okx_fake(total: int):
    data = [{"fundingTime": str(1_700_000_000_000 + i * H8), "fundingRate": "0.0001"}
            for i in range(total)]
    data.sort(key=lambda r: -int(r["fundingTime"]))         # newest first, like OKX

    def fetch(inst_id, after):
        rows = [r for r in data if after is None or int(r["fundingTime"]) < after]
        return rows[:100]
    return fetch


def test_okx_pagination_collects_everything_ascending_unique():
    rows = fetch_okx_history(_okx_fake(250), "BTC-USDT-SWAP")
    assert len(rows) == 250
    assert [t for t, _ in rows] == sorted({t for t, _ in rows})
    assert rows[0][0] == 1_700_000_000_000


def test_binance_pagination_respects_end():
    start = 1_700_000_000_000
    data = [{"fundingTime": start + i * H8, "fundingRate": "0.0002"} for i in range(2500)]

    def fetch(symbol, cursor):
        return [r for r in data if r["fundingTime"] >= cursor][:1000]

    end = start + 1999 * H8
    rows = fetch_binance_history(fetch, "BTCUSDT", start, end)
    assert len(rows) == 2000 and rows[-1][0] == end


def test_sqlite_roundtrip_and_upsert(tmp_path):
    db = tmp_path / "sub" / "carry.db"
    save_rates(db, "okx", "BTC-USDT-SWAP", [(1, 0.1), (2, 0.2)])
    save_rates(db, "okx", "BTC-USDT-SWAP", [(2, 0.5), (3, 0.3)])
    assert load_rates(db, "okx", "BTC-USDT-SWAP") == [(1, 0.1), (2, 0.5), (3, 0.3)]
    assert load_rates(db, "binance", "BTCUSDT") == []


def test_median_interval_hours():
    rows = [(i * H8, 0.0) for i in range(10)]
    assert median_interval_hours(rows) == 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_funding_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.funding_history'`

- [ ] **Step 3: Write minimal implementation**

`src/carry/funding_history.py`:

```python
"""Funding-rate history: paginated fetchers (network injected) and a SQLite store."""
from __future__ import annotations

import sqlite3
import statistics
import time
from pathlib import Path
from typing import Callable

Rows = list[tuple[int, float]]  # ascending (funding_time_ms, rate)


def fetch_okx_history(fetch_page: Callable[[str, int | None], list[dict]],
                      inst_id: str, max_pages: int = 500) -> Rows:
    """Walk backwards: each page holds records earlier than `after` (newest first)."""
    seen: dict[int, float] = {}
    after: int | None = None
    for _ in range(max_pages):
        page = fetch_page(inst_id, after)
        if not page:
            break
        for r in page:
            seen[int(r["fundingTime"])] = float(r["fundingRate"])
        oldest = min(int(r["fundingTime"]) for r in page)
        if after is not None and oldest >= after:
            break
        after = oldest
    return sorted(seen.items())


def fetch_binance_history(fetch_page: Callable[[str, int], list[dict]], symbol: str,
                          start_ms: int, end_ms: int, max_pages: int = 500) -> Rows:
    """Walk forwards from start_ms; each page is ascending."""
    seen: dict[int, float] = {}
    cursor = start_ms
    for _ in range(max_pages):
        page = fetch_page(symbol, cursor)
        if not page:
            break
        for r in page:
            t = int(r["fundingTime"])
            if t <= end_ms:
                seen[t] = float(r["fundingRate"])
        last = max(int(r["fundingTime"]) for r in page)
        if last >= end_ms:
            break
        cursor = last + 1
    return sorted(seen.items())


def make_okx_page_fetcher(public_api, pause: float = 0.25) -> Callable[[str, int | None], list[dict]]:
    def fetch(inst_id: str, after: int | None) -> list[dict]:
        time.sleep(pause)
        resp = public_api.get_funding_rate_history(
            instId=inst_id, after=str(after) if after else "", limit="100")
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX funding history error: {resp}")
        return resp["data"]
    return fetch


def make_binance_page_fetcher(pause: float = 0.2) -> Callable[[str, int], list[dict]]:
    import httpx

    def fetch(symbol: str, start_ms: int) -> list[dict]:
        time.sleep(pause)
        r = httpx.get("https://fapi.binance.com/fapi/v1/fundingRate",
                      params={"symbol": symbol, "startTime": start_ms, "limit": 1000}, timeout=20)
        r.raise_for_status()
        return r.json()
    return fetch


def save_rates(db_path: str | Path, exchange: str, symbol: str, rows: Rows) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    with con:
        con.execute("CREATE TABLE IF NOT EXISTS funding_rates ("
                    "exchange TEXT, symbol TEXT, ts INTEGER, rate REAL, "
                    "PRIMARY KEY (exchange, symbol, ts))")
        con.executemany("INSERT OR REPLACE INTO funding_rates VALUES (?,?,?,?)",
                        [(exchange, symbol, t, r) for t, r in rows])
    con.close()


def load_rates(db_path: str | Path, exchange: str, symbol: str) -> Rows:
    if not Path(db_path).exists():
        return []
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT ts, rate FROM funding_rates WHERE exchange=? AND symbol=? ORDER BY ts",
                          (exchange, symbol))
        return [(int(t), float(r)) for t, r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def median_interval_hours(rows: Rows) -> float:
    diffs = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    return statistics.median(diffs) / 3_600_000 if diffs else 0.0
```

`scripts/carry_fetch_funding.py`:

```python
"""Download BTC/ETH funding history (OKX recent + Binance long history) into data/carry.db."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import okx.PublicData as PublicData

from carry.funding_history import (
    fetch_binance_history, fetch_okx_history, make_binance_page_fetcher,
    make_okx_page_fetcher, median_interval_hours, save_rates,
)

PAIRS = [("BTC-USDT-SWAP", "BTCUSDT"), ("ETH-USDT-SWAP", "ETHUSDT")]


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/carry.db")
    ap.add_argument("--binance-start", default="2019-09-01")
    args = ap.parse_args()

    start_ms = int(datetime.strptime(args.binance_start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    okx_fetch = make_okx_page_fetcher(PublicData.PublicAPI(flag="0", debug=False))
    bn_fetch = make_binance_page_fetcher()

    for okx_id, bn_sym in PAIRS:
        for exch, sym, rows in (
            ("okx", okx_id, fetch_okx_history(okx_fetch, okx_id)),
            ("binance", bn_sym, fetch_binance_history(bn_fetch, bn_sym, start_ms, end_ms)),
        ):
            save_rates(args.db, exch, sym, rows)
            if rows:
                print(f"{exch:8s} {sym:14s} n={len(rows):6d} {_fmt(rows[0][0])} → {_fmt(rows[-1][0])} "
                      f"median_interval={median_interval_hours(rows):.2f}h")
            else:
                print(f"{exch:8s} {sym:14s} NO DATA")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_funding_history.py -v`
Expected: 4 passed

- [ ] **Step 5: Smoke-check the real method names, then fetch**

Run: `$PY -c "import okx.PublicData as P, inspect; print(inspect.signature(P.PublicAPI.get_funding_rate_history))"`
Expected: signature contains `instId`, `before`, `after`, `limit`. If names differ, fix `make_okx_page_fetcher` accordingly.

Run: `PYTHONPATH=src $PY scripts/carry_fetch_funding.py`
Expected: four lines (okx/binance × BTC/ETH) with row counts, date ranges, `median_interval=8.00h`. Record OKX's earliest date (this answers "how deep is OKX history"). If Binance returns HTTP 451/403 (region block), note it in the research doc and continue with OKX-only data.

- [ ] **Step 6: Commit**

```bash
git check-ignore data/carry.db || echo "WARNING: data/carry.db not ignored — do not add it"
git add src/carry/funding_history.py scripts/carry_fetch_funding.py tests/test_carry_funding_history.py
git commit -m "feat(carry): funding-rate history fetch and SQLite store

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Read-only OKX probe

**Files:**
- Create: `src/carry/okx_probe.py`
- Create: `scripts/carry_research_probe.py`
- Test: `tests/test_carry_okx_probe.py`

**Interfaces:**
- Produces: `collect_probe(public, account, swap_ids=("BTC-USDT-SWAP", "ETH-USDT-SWAP"), spot_ids=("BTC-USDT", "ETH-USDT")) -> dict` with keys `swaps`, `spots`, `funding`, `loan_quota`, and (only when `account` is not None) `account`, `borrow`, `fees`, `balance`; `render_markdown(findings: dict) -> str`. `public`/`account` are python-okx `PublicAPI`/`AccountAPI`-shaped objects (only methods listed in the test fakes are used).

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from carry.okx_probe import collect_probe, render_markdown


def ok(data):
    return {"code": "0", "msg": "", "data": data}


class FakePublic:
    def get_instruments(self, instType, instId):
        if instType == "SWAP":
            return ok([{"ctVal": "0.01", "ctMult": "1", "lotSz": "1", "minSz": "1", "tickSz": "0.1"}])
        return ok([{"lotSz": "0.00000001", "minSz": "0.00001", "tickSz": "0.1"}])

    def get_funding_rate(self, instId):
        return ok([{"fundingRate": "0.0001", "fundingTime": "1700000000000",
                    "nextFundingTime": "1700028800000"}])

    def get_interest_rate_loan_quota(self):
        return ok([{"basic": [{"ccy": "USDT", "rate": "0.0000015", "quota": "1000000"}]}])


class FakeAccount:
    def get_account_config(self):
        return ok([{"acctLv": "3", "autoLoan": True, "posMode": "net_mode"}])

    def get_interest_rate(self, ccy):
        return ok([{"ccy": ccy, "interestRate": "0.00000833"}])

    def get_fee_rates(self, instType):
        return ok([{"maker": "-0.0008", "taker": "-0.001"}])

    def get_account_balance(self):
        return ok([{"totalEq": "1000", "adjEq": "990", "imr": "0", "mmr": "0",
                    "mgnRatio": "", "notionalUsd": "0"}])


def test_collect_public_only():
    out = collect_probe(FakePublic(), None)
    assert out["swaps"]["BTC-USDT-SWAP"]["ctVal"] == "0.01"
    assert out["funding"]["ETH-USDT-SWAP"]["interval_h"] == 8.0
    assert out["spots"]["BTC-USDT"]["minSz"] == "0.00001"
    assert "account" not in out


def test_collect_with_account():
    out = collect_probe(FakePublic(), FakeAccount())
    assert out["account"]["acctLv"] == "3"
    assert out["borrow"]["apr_if_hourly"] == 0.00000833 * 24 * 365
    assert out["fees"]["SPOT"]["taker"] == "-0.001"
    assert out["balance"]["mgnRatio"] == ""


def test_render_markdown_mentions_key_facts():
    md = render_markdown(collect_probe(FakePublic(), FakeAccount()))
    for needle in ("acctLv", "BTC-USDT-SWAP", "apr_if_hourly", "mgnRatio"):
        assert needle in md


def test_probe_code_is_read_only():
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/carry/okx_probe.py", "scripts/carry_research_probe.py"):
        text = (root / rel).read_text()
        for banned in ("place_order", "TradeAPI", "FundingAPI", "cancel_order", "set_leverage",
                       "set_account_level", "borrow_repay"):
            assert banned not in text, f"{banned} found in {rel}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_okx_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.okx_probe'`

- [ ] **Step 3: Write minimal implementation**

`src/carry/okx_probe.py`:

```python
"""Read-only OKX facts needed to design the carry strategy (no trading calls)."""
from __future__ import annotations


def _data(resp: dict, what: str) -> list[dict]:
    if resp.get("code") != "0":
        raise RuntimeError(f"OKX {what}: code={resp.get('code')} msg={resp.get('msg')}")
    return resp["data"]


def collect_probe(public, account,
                  swap_ids: tuple[str, ...] = ("BTC-USDT-SWAP", "ETH-USDT-SWAP"),
                  spot_ids: tuple[str, ...] = ("BTC-USDT", "ETH-USDT")) -> dict:
    out: dict = {"swaps": {}, "spots": {}, "funding": {}, "loan_quota": None}
    for i in swap_ids:
        d = _data(public.get_instruments(instType="SWAP", instId=i), f"instruments {i}")[0]
        out["swaps"][i] = {k: d[k] for k in ("ctVal", "ctMult", "lotSz", "minSz", "tickSz")}
        f = _data(public.get_funding_rate(instId=i), f"funding-rate {i}")[0]
        out["funding"][i] = {
            "rate": float(f["fundingRate"]),
            "interval_h": (int(f["nextFundingTime"]) - int(f["fundingTime"])) / 3_600_000,
        }
    for i in spot_ids:
        d = _data(public.get_instruments(instType="SPOT", instId=i), f"instruments {i}")[0]
        out["spots"][i] = {k: d[k] for k in ("lotSz", "minSz", "tickSz")}
    try:
        out["loan_quota"] = _data(public.get_interest_rate_loan_quota(), "loan-quota")
    except AttributeError:
        out["loan_quota"] = "unavailable: method missing in installed python-okx"

    if account is not None:
        cfg = _data(account.get_account_config(), "account-config")[0]
        out["account"] = {k: cfg.get(k) for k in ("acctLv", "autoLoan", "posMode")}
        ir = _data(account.get_interest_rate(ccy="USDT"), "interest-rate")[0]
        out["borrow"] = {"raw_interestRate": ir["interestRate"],
                         # ASSUMPTION: OKX quotes this per hour — confirm against the web UI
                         "apr_if_hourly": float(ir["interestRate"]) * 24 * 365}
        out["fees"] = {}
        for t in ("SPOT", "SWAP"):
            d = _data(account.get_fee_rates(instType=t), f"fee-rates {t}")[0]
            out["fees"][t] = {"maker": d["maker"], "taker": d["taker"]}  # OKX sign: negative = fee paid
        bal = _data(account.get_account_balance(), "balance")[0]
        out["balance"] = {k: bal.get(k) for k in
                          ("totalEq", "adjEq", "imr", "mmr", "mgnRatio", "notionalUsd")}
    return out


def render_markdown(findings: dict) -> str:
    lines = ["# OKX probe (read-only)", ""]
    for section in ("account", "borrow", "fees", "balance", "loan_quota"):
        if section in findings:
            lines += [f"## {section}", "```", str(findings[section]), "```", ""]
    lines += ["## swaps", "```", *(f"{k}: {v}" for k, v in findings["swaps"].items()), "```", ""]
    lines += ["## spots", "```", *(f"{k}: {v}" for k, v in findings["spots"].items()), "```", ""]
    lines += ["## funding", "```", *(f"{k}: {v}" for k, v in findings["funding"].items()), "```", ""]
    return "\n".join(lines)
```

`scripts/carry_research_probe.py`:

```python
"""Run the read-only OKX probe. Needs a READ-ONLY key in env for account facts."""
from __future__ import annotations

import argparse
import os
import sys

import okx.Account as Account
import okx.PublicData as PublicData

from carry.okx_probe import collect_probe, render_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-only", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    flag = "1" if args.demo else "0"
    public = PublicData.PublicAPI(flag=flag, debug=False)
    account = None
    if not args.public_only:
        key, secret, phrase = (os.environ.get(k, "") for k in
                               ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"))
        if not (key and secret and phrase):
            sys.exit("Set OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE (read-only key) or use --public-only")
        account = Account.AccountAPI(key, secret, phrase, False, flag=flag, debug=False)

    md = render_markdown(collect_probe(public, account))
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_okx_probe.py -v`
Expected: 4 passed

- [ ] **Step 5: Verify python-okx method names against the installed version**

Run: `$PY -c "import okx.Account as A, okx.PublicData as P; print([m for m in dir(A.AccountAPI) if m.startswith('get_') and any(s in m for s in ('config','interest','fee','balance'))]); print([m for m in dir(P.PublicAPI) if any(s in m for s in ('instruments','funding','loan'))])"`
Expected: names include `get_account_config`, `get_interest_rate`, `get_fee_rates`, `get_account_balance`, `get_instruments`, `get_funding_rate`, `get_interest_rate_loan_quota`. If any differ, adjust `collect_probe` and the test fakes together.

- [ ] **Step 6: Public-only smoke run**

Run: `PYTHONPATH=src $PY scripts/carry_research_probe.py --public-only`
Expected: markdown with `swaps` (ctVal 0.01 for BTC, 0.1 for ETH — verify), `spots`, `funding` (interval_h 8.0 unless OKX changed it), `loan_quota`.

- [ ] **Step 7: Commit**

```bash
git add src/carry/okx_probe.py scripts/carry_research_probe.py tests/test_carry_okx_probe.py
git commit -m "feat(carry): read-only OKX research probe

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Viability report script

**Files:**
- Create: `scripts/carry_viability.py`

**Interfaces:**
- Consumes: `load_rates`, `median_interval_hours` (Task 3); `CarryParams`, `simulate_carry` (Task 2)
- Produces: CLI `python scripts/carry_viability.py [--db data/carry.db] [--exchange binance|okx] [--enter 0.06 --exit 0.02 --lev-spread 0.03]` printing (a) full-period grid, (b) per-calendar-year table for the base scenario, (c) an entry-threshold sweep. No new library API, so no new unit test; the simulator is already tested. Verification is a run against real data.

- [ ] **Step 1: Write the script**

```python
"""Viability report: does spot-long + perp-short funding carry pay after fees and borrow interest?"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone

from carry.funding_history import load_rates, median_interval_hours
from carry.simulate import CarryParams, simulate_carry

SYMBOLS = {"binance": ("BTCUSDT", "ETHUSDT"), "okx": ("BTC-USDT-SWAP", "ETH-USDT-SWAP")}
FEES = {"taker": 0.0015, "maker": 0.0010}       # per side, both legs
BORROW = (0.05, 0.08, 0.12, 0.20)
HEADER = "| {} | net APR | maxDD | in-mkt | trips | avg lev |\n|---|---|---|---|---|---|"


def _row(label: str, r) -> str:
    return (f"| {label} | {r.net_apr:+.2%} | {r.max_drawdown:.2%} | "
            f"{r.time_in_market:.0%} | {r.round_trips} | {r.avg_leverage:.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/carry.db")
    ap.add_argument("--exchange", default="binance", choices=list(SYMBOLS))
    ap.add_argument("--enter", type=float, default=0.06)
    ap.add_argument("--exit", type=float, default=0.02)
    ap.add_argument("--lev-spread", type=float, default=0.03)
    args = ap.parse_args()
    base = CarryParams(enter_apr=args.enter, exit_apr=args.exit, lev_min_spread=args.lev_spread)

    for sym in SYMBOLS[args.exchange]:
        rows = load_rates(args.db, args.exchange, sym)
        if not rows:
            print(f"\n## {sym}: no data (run scripts/carry_fetch_funding.py)")
            continue
        assert abs(median_interval_hours(rows) - 8.0) < 0.01, "funding interval is not 8h"
        rates = [r for _, r in rows]
        first = datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc).date()
        last = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).date()
        print(f"\n## {args.exchange} {sym}  {first} → {last}  "
              f"(mean funding APR {sum(rates) / len(rates) * 1095:.2%})")

        print("\n### Full period: fee × borrow APR × max leverage\n")
        print(HEADER.format("fee / borrow / maxL"))
        for fee_name, fee in FEES.items():
            for b in BORROW:
                for lev in (1.0, 1.5):
                    p = replace(base, fee_side=fee, borrow_apr=b, max_leverage=lev)
                    print(_row(f"{fee_name} / {b:.0%} / {lev}", simulate_carry(rates, p)))

        print("\n### Per calendar year (taker fee, borrow 8%, maxL 1.5 vs 1.0)\n")
        print(HEADER.format("year / maxL"))
        by_year: dict[int, list[float]] = {}
        for t, r in rows:
            by_year.setdefault(datetime.fromtimestamp(t / 1000, tz=timezone.utc).year, []).append(r)
        for year, rs in sorted(by_year.items()):
            for lev in (1.0, 1.5):
                p = replace(base, max_leverage=lev)
                print(_row(f"{year} / {lev}", simulate_carry(rs, p)))

        print("\n### Entry-threshold sweep (taker, borrow 8%, maxL 1.5; exit = enter/3)\n")
        print(HEADER.format("enter APR"))
        for e in (0.04, 0.06, 0.08, 0.10):
            p = replace(base, enter_apr=e, exit_apr=e / 3)
            print(_row(f"{e:.0%}", simulate_carry(rates, p)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against the data from Task 3**

Run: `PYTHONPATH=src $PY scripts/carry_viability.py --exchange binance > /tmp/viability_binance.md; head -40 /tmp/viability_binance.md`
Expected: three markdown tables per symbol, no traceback. Sanity check: `mean funding APR` for BTC over 2019–2026 should be roughly 8–12% (the 0.01%/8h base rate dominates); if it is wildly off, suspect a units bug in the loader before trusting any result.

- [ ] **Step 3: Commit**

```bash
git add scripts/carry_viability.py
git commit -m "feat(carry): viability report script

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Run research, write findings, go/no-go

**Files:**
- Create: `docs/superpowers/specs/2026-09-19-funding-carry-research.md`

This task needs the user: a read-only OKX API key (Trade permission OFF, Withdraw OFF, IP-whitelist recommended) exported in the shell for one run. The agent must not ask the user to paste secrets into chat; use `! export OKX_API_KEY=... ` style input in the user's own shell or a local `.env` that is git-ignored.

- [ ] **Step 1: Run the authenticated probe**

Run: `PYTHONPATH=src $PY scripts/carry_research_probe.py --out /tmp/okx_probe.md`
Expected: `account.acctLv` (confirm 3), `autoLoan`, `borrow` (raw and `apr_if_hourly`), `fees` (real tier), `balance` (`mgnRatio` empty when no positions).

- [ ] **Step 2: Confirm borrow-rate units and mgnRatio against the OKX web UI**

In the OKX web UI (Trade → Margin / Borrow page) compare the displayed USDT borrow APR with `apr_if_hourly`. If they disagree, the "per hour" assumption is wrong: correct the computation in `okx_probe.py` and the finding in the doc. For `mgnRatio`, note whether the UI shows a percentage (e.g. 350%) and whether the API returns `3.5` or `350`; the doc must record the unit. (No position exists yet, so this may only be observable in Phase 4 on demo; record "unverified" explicitly if so.)

- [ ] **Step 3: Run the viability report on both data sources and save**

Run:
```
PYTHONPATH=src $PY scripts/carry_viability.py --exchange binance > /tmp/via_bn.md
PYTHONPATH=src $PY scripts/carry_viability.py --exchange okx > /tmp/via_okx.md
```
Expected: binance covers 2019/2020→now; okx covers only the recent window (record its earliest date).

- [ ] **Step 4: Write the research doc**

Create `docs/superpowers/specs/2026-09-19-funding-carry-research.md` with exactly these sections, filled from the outputs above (paste the actual tables; no placeholders):

1. **Account** — `acctLv`, `autoLoan`, whether user's "unified account" wording maps to acctLv=3 (state the finding).
2. **Instruments** — ctVal/lotSz/minSz for BTC/ETH swap and spot; min notional implied at 1000U.
3. **Fees** — real spot/swap maker/taker; round-trip cost; break-even hold days at mean funding APR.
4. **Borrow** — USDT rate (units confirmed or flagged unverified), quota/limits at 1.5x (500U).
5. **Funding history** — OKX depth, Binance depth, median interval, any gaps.
6. **Viability results** — grid, per-year table, threshold sweep (paste); a 3-line read: which years pay, which don't, whether 1.5x beat 1.0x after borrow cost.
7. **Open risks** — mgnRatio scale/threshold still to calibrate (Phase 3 plan), basis PnL not modelled (only a 0.02% slip per trade), borrow-rate history unavailable so borrow is scenario-based.
8. **Gate proposal** — evaluate against these PROPOSED criteria and state pass/fail for each (the user may change them): (a) 1.0x, taker fee, borrow 8%: net APR >= 4% on the most recent 2 full calendar years for both BTC and ETH; (b) 1.5x beats 1.0x by >= 1 percentage point in the same window, otherwise drop leverage from scope; (c) max drawdown <= 3% of capital; (d) beats the user's current risk-free alternative (OKX Simple Earn USDT APR — ask the user for the number).

- [ ] **Step 5: Commit and hand the decision to the user**

```bash
git add docs/superpowers/specs/2026-09-19-funding-carry-research.md
git commit -m "docs(carry): research findings and go/no-go evaluation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

Stop here. Present the gate result to the user. Only on an explicit "go" write the Phase 3 plan (signal/sizing/risk/state/watchdog/FakeOKX, including mgnRatio shock-simulation calibrated from Task 6 findings). On "no-go" the branch ends with the research doc as its deliverable.
