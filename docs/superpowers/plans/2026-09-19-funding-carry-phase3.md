# Funding Carry — Phase 3 (Offline Core: sizing, decision, risk, state, watchdog, executor, engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and fully unit-test everything that decides and executes a 1.0x spot-long + perp-short carry position, against a fault-injecting `FakeExchange`, with zero network access and no credentials.

**Architecture:** Pure modules (`types`, `config`, `sizing`, `hedge`, `decision`, `risk`, `state`) + an `Exchange` Protocol with a `FakeExchange` + `Executor` (open/close/rollback) + `Watchdog` (hedge-integrity/ADL) + `Engine.step()` (one loop iteration). The real OKX client, CLI, systemd unit, demo/live are Phase 4-5 (separate plan, written after this phase and after demo keys exist).

**Tech Stack:** Python 3.11+, stdlib only (`decimal`, `json`, `dataclasses`, `enum`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-funding-carry-design.md` (with the deviations below). Evidence: `2026-09-19-funding-carry-research.md`.

## Deviations from the spec (rulings, decided with the user's confirmation of scope)

- **1.0x only, no borrowing** (`autoLoan` stays off): research showed 1.5x loses in weak years. No leverage tiers, no repay logic.
- **ADL detection = REST polling of position drift** (authoritative: an ADL/liquidation shrinks the perp) plus an ADL-rank *alert*; the WebSocket `adl-warning` channel is deferred to Phase 4. ADL rank alone does NOT close the hedge (a high rank is normal for a profitable short); a shrunken perp does, immediately.
- **Market orders only** (backtest assumed taker cost).
- **After any watchdog or risk close the engine is `HALTED`** (no new entries) until a human calls `rearm()`.
- **Never sell coins the bot did not buy**: if the exchange shows BTC/ETH holdings while state is flat, the engine halts and alerts; it does not trade.
- `mgnRatio` thresholds are configurable with a unit divisor; defaults assume ratio form (1.0 = liquidation) and MUST be calibrated on demo (Phase 4).
- Module named `decision.py` (not `signal.py`) to avoid shadowing the stdlib `signal` module.

## Global Constraints

- Python >= 3.11, type hints on all signatures, `from __future__ import annotations`; stdlib only in `src/carry/` for this phase (no python-okx, no httpx, no network).
- Scope: BTC and ETH only. Real instrument facts (from the OKX probe): BTC swap ctVal 0.01, lot/min 0.01 contracts, spot lot 1e-8 / min 1e-5; ETH swap ctVal 0.1, lot/min 0.01 contracts, spot lot 1e-6 / min 1e-4. Funding interval 8h. Spot taker 0.10%, swap taker 0.05%.
- Signal defaults (from the research doc): window 9 settlements, enter 6% APR, exit 2% APR, min hold 9 settlements.
- No credentials, no files outside `live_state/` (state file path is configurable; tests use `tmp_path`).
- Tests: worktree has no venv: `export PY=/home/airwolf/Github/crypto_backtest/.venv/bin/python`, run `$PY -m pytest tests/test_carry_*.py -q` from the worktree root.
- Commit trailer exactly: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (whichever model you are).

## File Structure

| File | Responsibility |
|---|---|
| `src/carry/types.py` | `Instrument`, `OrderResult`, BTC/ETH instrument constants |
| `src/carry/config.py` | `CarryConfig` (validated) |
| `src/carry/sizing.py` | lot rounding, `plan_position`, `drift` |
| `src/carry/hedge.py` | `HedgeStatus`, `check_hedge` |
| `src/carry/decision.py` | `decide` (open/close/hold) |
| `src/carry/risk.py` | `Tier`, `classify` |
| `src/carry/state.py` | `CoinLeg`, `CarryState`, `save_state`, `load_state` |
| `src/carry/exchange.py` | `Exchange` Protocol |
| `src/carry/fake_exchange.py` | `FakeExchange` with fault injection |
| `src/carry/executor.py` | `Executor`, `HedgeError` |
| `src/carry/watchdog.py` | `Watchdog` |
| `src/carry/engine.py` | `Engine` (`step`, `force_open`, `force_close`, `rearm`) |
| `tests/test_carry_<module>.py` | one test file per module |

---

### Task 1: Types, config, sizing

**Files:**
- Create: `src/carry/types.py`, `src/carry/config.py`, `src/carry/sizing.py`
- Test: `tests/test_carry_sizing.py`, `tests/test_carry_config.py`

**Interfaces:**
- Produces: `Instrument(coin, spot_id, swap_id, ct_val, swap_lot, swap_min, spot_lot, spot_min)`, `OrderResult(ok, filled=0.0, avg_price=0.0, error="")`, `BTC`, `ETH`, `DEFAULT_INSTRUMENTS: dict[str, Instrument]`; `CarryConfig` (fields below, property `per_coin_notional`); `Sizing(contracts, coin_qty, spot_qty, notional)`, `floor_to(x, step)`, `ceil_to(x, step)`, `plan_position(notional_usdt, price, inst, spot_taker_fee) -> Sizing`, `drift(spot_qty, contracts, inst) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_carry_sizing.py`:

```python
import math

import pytest

from carry.sizing import ceil_to, drift, floor_to, plan_position
from carry.types import BTC, ETH


def test_floor_and_ceil_handle_float_noise():
    assert floor_to(0.20833333, 0.01) == pytest.approx(0.20)
    assert floor_to(1.4999999999999998, 0.01) == pytest.approx(1.5)
    assert ceil_to(0.015015015015, 1e-8) == pytest.approx(0.01501502)


def test_plan_btc():
    s = plan_position(900.0, 60000.0, BTC, 0.001)
    assert s.contracts == pytest.approx(1.5)
    assert s.coin_qty == pytest.approx(0.015)
    assert s.spot_qty == pytest.approx(0.01501502)
    assert s.notional == pytest.approx(900.0)


def test_plan_eth():
    s = plan_position(300.0, 3000.0, ETH, 0.001)
    assert s.contracts == pytest.approx(1.0)
    assert s.coin_qty == pytest.approx(0.1)
    assert s.spot_qty == pytest.approx(0.100101)


def test_plan_rounds_contracts_down():
    assert plan_position(905.0, 60000.0, BTC, 0.001).contracts == pytest.approx(1.5)


def test_plan_rejects_too_small():
    with pytest.raises(ValueError):
        plan_position(5.0, 60000.0, BTC, 0.001)


def test_drift():
    assert drift(0.01501502, 1.5, BTC) == pytest.approx(0.001, rel=0.02)
    assert drift(0.015, 0.0, BTC) == math.inf
    assert drift(0.0, 0.0, BTC) == 0.0
```

`tests/test_carry_config.py`:

```python
import pytest

from carry.config import CarryConfig


def test_defaults_are_valid():
    c = CarryConfig()
    assert c.per_coin_notional == pytest.approx(125.0)
    assert c.exit_apr < c.enter_apr


def test_rejects_inverted_apr_thresholds():
    with pytest.raises(ValueError):
        CarryConfig(enter_apr=0.02, exit_apr=0.06)


def test_rejects_bad_mgn_thresholds():
    with pytest.raises(ValueError):
        CarryConfig(mgn_warn=2.0, mgn_close=2.5, mgn_emergency=1.5)


def test_rejects_notional_above_cap():
    with pytest.raises(ValueError):
        CarryConfig(deploy_usdt=1000.0, max_order_notional=300.0)


def test_rejects_nonpositive_deploy():
    with pytest.raises(ValueError):
        CarryConfig(deploy_usdt=0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_carry_sizing.py tests/test_carry_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.sizing'` (and `carry.config`)

- [ ] **Step 3: Write minimal implementation**

`src/carry/types.py`:

```python
"""Shared value types for the carry engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    coin: str
    spot_id: str
    swap_id: str
    ct_val: float      # coins per swap contract
    swap_lot: float    # contract lot size
    swap_min: float    # minimum contracts
    spot_lot: float    # base-coin lot size
    spot_min: float    # minimum base-coin quantity


BTC = Instrument("BTC", "BTC-USDT", "BTC-USDT-SWAP", 0.01, 0.01, 0.01, 1e-8, 1e-5)
ETH = Instrument("ETH", "ETH-USDT", "ETH-USDT-SWAP", 0.1, 0.01, 0.01, 1e-6, 1e-4)
DEFAULT_INSTRUMENTS: dict[str, Instrument] = {"BTC": BTC, "ETH": ETH}


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    filled: float = 0.0       # base coins (spot) or contracts (swap), gross
    avg_price: float = 0.0
    error: str = ""
```

`src/carry/config.py`:

```python
"""Validated configuration for the carry engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarryConfig:
    coins: tuple[str, ...] = ("BTC", "ETH")
    deploy_usdt: float = 250.0          # total notional per leg, split equally across coins
    max_order_notional: float = 300.0   # hard cap for one open order (per leg, per coin)
    spot_taker_fee: float = 0.001
    drift_tol: float = 0.02             # allowed |spot - perp| / perp
    window: int = 9                     # trailing settlements used by the entry signal
    enter_apr: float = 0.06
    exit_apr: float = 0.02
    min_hold: int = 9                   # settlements before a fade-exit is allowed
    mgn_warn: float = 4.0               # mgnRatio tiers, in ratio form (1.0 = liquidation)
    mgn_close: float = 2.5
    mgn_emergency: float = 1.5
    mgn_unit: float = 1.0               # divide the raw API value by this to get ratio form
    adl_alert_rank: int = 4
    poll_sec: float = 2.0
    state_path: str = "live_state/carry_state.json"

    def __post_init__(self) -> None:
        if self.deploy_usdt <= 0:
            raise ValueError("deploy_usdt must be > 0")
        if not self.exit_apr < self.enter_apr:
            raise ValueError("exit_apr must be below enter_apr")
        if not self.mgn_emergency < self.mgn_close < self.mgn_warn:
            raise ValueError("need mgn_emergency < mgn_close < mgn_warn")
        if self.mgn_unit <= 0:
            raise ValueError("mgn_unit must be > 0")
        if self.per_coin_notional > self.max_order_notional:
            raise ValueError("per-coin notional exceeds max_order_notional")

    @property
    def per_coin_notional(self) -> float:
        return self.deploy_usdt / len(self.coins)
```

`src/carry/sizing.py`:

```python
"""Order sizing: lot rounding, spot/perp quantity matching, hedge drift."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from carry.types import Instrument


def _steps(x: float, step: float, rounding: str) -> float:
    d = Decimal(str(round(x, 12))) / Decimal(str(step))
    return float(d.to_integral_value(rounding=rounding) * Decimal(str(step)))


def floor_to(x: float, step: float) -> float:
    return _steps(x, step, ROUND_FLOOR)


def ceil_to(x: float, step: float) -> float:
    return _steps(x, step, ROUND_CEILING)


@dataclass(frozen=True)
class Sizing:
    contracts: float   # swap contracts to short
    coin_qty: float    # perp exposure in coins = contracts * ct_val
    spot_qty: float    # spot base quantity to buy (grossed up for the fee deducted in kind)
    notional: float    # coin_qty * price


def plan_position(notional_usdt: float, price: float, inst: Instrument,
                  spot_taker_fee: float) -> Sizing:
    coins = notional_usdt / price
    contracts = floor_to(coins / inst.ct_val, inst.swap_lot)
    if contracts < inst.swap_min:
        raise ValueError(f"{inst.coin}: notional {notional_usdt} too small for min {inst.swap_min} contracts")
    coin_qty = float(Decimal(str(contracts)) * Decimal(str(inst.ct_val)))
    spot_qty = ceil_to(coin_qty / (1.0 - spot_taker_fee), inst.spot_lot)
    if spot_qty < inst.spot_min:
        raise ValueError(f"{inst.coin}: spot quantity {spot_qty} below minimum {inst.spot_min}")
    return Sizing(contracts, coin_qty, spot_qty, coin_qty * price)


def drift(spot_qty: float, contracts: float, inst: Instrument) -> float:
    """(spot - perp) / perp in coin terms; inf if perp is zero but spot is not."""
    perp = contracts * inst.ct_val
    if perp == 0:
        return math.inf if spot_qty > 0 else 0.0
    return (spot_qty - perp) / perp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_carry_sizing.py tests/test_carry_config.py -q`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/types.py src/carry/config.py src/carry/sizing.py tests/test_carry_sizing.py tests/test_carry_config.py
git commit -m "feat(carry): types, validated config and order sizing

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Hedge check, entry decision, risk tiers

**Files:**
- Create: `src/carry/hedge.py`, `src/carry/decision.py`, `src/carry/risk.py`
- Test: `tests/test_carry_hedge.py`, `tests/test_carry_decision.py`, `tests/test_carry_risk.py`

**Interfaces:**
- Consumes: `Instrument`, `BTC` (types); `CarryConfig` (config); `drift` (sizing); `funding_apr` (`carry.funding_math`, already exists)
- Produces: `HedgeStatus` (str Enum: FLAT, OK, DRIFT, SPOT_ONLY, PERP_ONLY), `check_hedge(spot_qty, contracts, inst, tol) -> HedgeStatus`; `decide(rates: Sequence[float], in_position: bool, held: int, p: CarryConfig) -> str` returning `"open" | "close" | "hold"`; `Tier` (IntEnum NORMAL=0, WARN=1, CLOSE=2, EMERGENCY=3), `classify(mgn_ratio: float | None, has_position: bool, cfg: CarryConfig) -> Tier`

- [ ] **Step 1: Write the failing tests**

`tests/test_carry_hedge.py`:

```python
from carry.hedge import HedgeStatus, check_hedge
from carry.types import BTC


def test_statuses():
    assert check_hedge(0.015, 1.5, BTC, 0.02) == HedgeStatus.OK
    assert check_hedge(0.0152, 1.5, BTC, 0.02) == HedgeStatus.OK
    assert check_hedge(0.015, 1.0, BTC, 0.02) == HedgeStatus.DRIFT
    assert check_hedge(0.015, 0.0, BTC, 0.02) == HedgeStatus.SPOT_ONLY
    assert check_hedge(0.0, 1.5, BTC, 0.02) == HedgeStatus.PERP_ONLY
    assert check_hedge(0.0, 0.0, BTC, 0.02) == HedgeStatus.FLAT


def test_dust_below_minimums_counts_as_flat():
    assert check_hedge(1e-6, 0.005, BTC, 0.02) == HedgeStatus.FLAT
```

`tests/test_carry_decision.py`:

```python
from carry.config import CarryConfig
from carry.decision import decide

P = CarryConfig()  # window 9, enter 6%, exit 2%, min_hold 9


def test_opens_when_trailing_funding_high():
    assert decide([0.0001] * 9, False, 0, P) == "open"          # 10.95% APR


def test_holds_when_below_entry():
    assert decide([0.00005] * 9, False, 0, P) == "hold"         # 5.5% APR


def test_holds_with_short_history():
    assert decide([0.0001] * 5, False, 0, P) == "hold"


def test_closes_when_faded_after_min_hold():
    assert decide([0.0] * 9, True, 9, P) == "close"


def test_min_hold_delays_close():
    assert decide([0.0] * 9, True, 3, P) == "hold"


def test_hysteresis_band_holds_position():
    assert decide([0.00003] * 9, True, 20, P) == "hold"         # 3.3% APR, between exit and enter


def test_no_double_open():
    assert decide([0.0001] * 9, True, 20, P) == "hold"
```

`tests/test_carry_risk.py`:

```python
from carry.config import CarryConfig
from carry.risk import Tier, classify

C = CarryConfig()  # warn 4.0, close 2.5, emergency 1.5


def test_tiers():
    assert classify(5.0, True, C) == Tier.NORMAL
    assert classify(3.0, True, C) == Tier.WARN
    assert classify(2.5, True, C) == Tier.CLOSE
    assert classify(1.4, True, C) == Tier.EMERGENCY


def test_missing_ratio_with_position_is_warn():
    assert classify(None, True, C) == Tier.WARN


def test_no_position_is_normal_whatever_the_ratio():
    assert classify(None, False, C) == Tier.NORMAL
    assert classify(0.1, False, C) == Tier.NORMAL


def test_unit_divisor_for_percent_form():
    c = CarryConfig(mgn_unit=100.0)
    assert classify(250.0, True, c) == Tier.CLOSE
    assert classify(450.0, True, c) == Tier.NORMAL


def test_tier_ordering():
    assert Tier.EMERGENCY > Tier.CLOSE > Tier.WARN > Tier.NORMAL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_carry_hedge.py tests/test_carry_decision.py tests/test_carry_risk.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.hedge'`

- [ ] **Step 3: Write minimal implementation**

`src/carry/hedge.py`:

```python
"""Hedge-integrity classification of a spot/perp pair."""
from __future__ import annotations

from enum import Enum

from carry.sizing import drift
from carry.types import Instrument


class HedgeStatus(str, Enum):
    FLAT = "FLAT"
    OK = "OK"
    DRIFT = "DRIFT"
    SPOT_ONLY = "SPOT_ONLY"
    PERP_ONLY = "PERP_ONLY"


def check_hedge(spot_qty: float, contracts: float, inst: Instrument, tol: float) -> HedgeStatus:
    spot_open = spot_qty >= inst.spot_min
    perp_open = contracts >= inst.swap_min
    if not spot_open and not perp_open:
        return HedgeStatus.FLAT
    if spot_open and not perp_open:
        return HedgeStatus.SPOT_ONLY
    if perp_open and not spot_open:
        return HedgeStatus.PERP_ONLY
    return HedgeStatus.OK if abs(drift(spot_qty, contracts, inst)) <= tol else HedgeStatus.DRIFT
```

`src/carry/decision.py`:

```python
"""Entry/exit decision from trailing funding (mirrors the validated simulator rule)."""
from __future__ import annotations

from typing import Sequence

from carry.config import CarryConfig
from carry.funding_math import funding_apr


def decide(rates: Sequence[float], in_position: bool, held: int, p: CarryConfig) -> str:
    """rates: last settled funding rates, oldest first. held: settlements since entry."""
    if len(rates) < p.window:
        return "hold"
    avg_apr = funding_apr(sum(rates[-p.window:]) / p.window)
    if not in_position:
        return "open" if avg_apr >= p.enter_apr else "hold"
    if avg_apr < p.exit_apr and held >= p.min_hold:
        return "close"
    return "hold"
```

`src/carry/risk.py`:

```python
"""Account-level margin-ratio risk tiers."""
from __future__ import annotations

from enum import IntEnum

from carry.config import CarryConfig


class Tier(IntEnum):
    NORMAL = 0
    WARN = 1
    CLOSE = 2
    EMERGENCY = 3


def classify(mgn_ratio: float | None, has_position: bool, cfg: CarryConfig) -> Tier:
    if not has_position:
        return Tier.NORMAL
    if mgn_ratio is None:
        return Tier.WARN
    r = mgn_ratio / cfg.mgn_unit
    if r <= cfg.mgn_emergency:
        return Tier.EMERGENCY
    if r <= cfg.mgn_close:
        return Tier.CLOSE
    if r <= cfg.mgn_warn:
        return Tier.WARN
    return Tier.NORMAL
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_carry_hedge.py tests/test_carry_decision.py tests/test_carry_risk.py -q`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/hedge.py src/carry/decision.py src/carry/risk.py tests/test_carry_hedge.py tests/test_carry_decision.py tests/test_carry_risk.py
git commit -m "feat(carry): hedge check, entry decision and risk tiers

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: State persistence

**Files:**
- Create: `src/carry/state.py`
- Test: `tests/test_carry_state.py`

**Interfaces:**
- Produces: `CoinLeg(contracts: float, spot_qty: float, entry_ts: int)`, `CarryState(mode: str = "FLAT", legs: dict[str, CoinLeg], halted_reason: str = "")` with `to_dict()` / `from_dict(d)`, `save_state(path, state) -> None` (atomic), `load_state(path) -> CarryState` (default state if file missing). `mode` is one of `"FLAT" | "HEDGED" | "HALTED"`.

- [ ] **Step 1: Write the failing test**

```python
import json

from carry.state import CarryState, CoinLeg, load_state, save_state


def test_roundtrip(tmp_path):
    p = tmp_path / "s" / "state.json"
    st = CarryState(mode="HEDGED", legs={"BTC": CoinLeg(1.5, 0.015, 1000)})
    save_state(p, st)
    got = load_state(p)
    assert got.mode == "HEDGED"
    assert got.legs["BTC"] == CoinLeg(1.5, 0.015, 1000)


def test_missing_file_gives_default(tmp_path):
    st = load_state(tmp_path / "nope.json")
    assert st.mode == "FLAT" and st.legs == {}


def test_save_is_atomic_and_leaves_no_temp(tmp_path):
    p = tmp_path / "state.json"
    save_state(p, CarryState())
    save_state(p, CarryState(mode="HALTED", halted_reason="x"))
    assert json.loads(p.read_text())["halted_reason"] == "x"
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


def test_from_dict_tolerates_missing_keys():
    st = CarryState.from_dict({})
    assert st.mode == "FLAT" and st.legs == {} and st.halted_reason == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_state.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.state'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Persistent engine state (a cache: the exchange is the source of truth)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CoinLeg:
    contracts: float
    spot_qty: float
    entry_ts: int


@dataclass
class CarryState:
    mode: str = "FLAT"                       # FLAT | HEDGED | HALTED
    legs: dict[str, CoinLeg] = field(default_factory=dict)
    halted_reason: str = ""

    def to_dict(self) -> dict:
        return {"mode": self.mode, "halted_reason": self.halted_reason,
                "legs": {c: asdict(l) for c, l in self.legs.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "CarryState":
        return cls(mode=d.get("mode", "FLAT"), halted_reason=d.get("halted_reason", ""),
                   legs={c: CoinLeg(**l) for c, l in d.get("legs", {}).items()})


def save_state(path: str | Path, state: CarryState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    os.replace(tmp, p)


def load_state(path: str | Path) -> CarryState:
    p = Path(path)
    if not p.exists():
        return CarryState()
    return CarryState.from_dict(json.loads(p.read_text()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_state.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/state.py tests/test_carry_state.py
git commit -m "feat(carry): persistent engine state with atomic save

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Exchange protocol and FakeExchange

**Files:**
- Create: `src/carry/exchange.py`, `src/carry/fake_exchange.py`
- Test: `tests/test_carry_fake_exchange.py`

**Interfaces:**
- Consumes: `OrderResult`, `DEFAULT_INSTRUMENTS` (types)
- Produces: `Exchange` Protocol with methods `price(coin) -> float`, `funding_history(coin, n) -> list[float]` (last n settled rates, oldest first), `spot_balance(coin) -> float`, `swap_contracts(coin) -> float` (short size, positive), `swap_adl_rank(coin) -> int` (0-5), `mgn_ratio() -> float | None`, `sell_swap(coin, contracts) -> OrderResult` (open/increase short, market), `buy_swap(coin, contracts) -> OrderResult` (reduce-only close, market), `buy_spot(coin, qty) -> OrderResult` (market, `qty` in base coin), `sell_spot(coin, qty) -> OrderResult`. `FakeExchange(prices=None, funding=None, spot_fee=0.001, mgn=10.0)` implements it, with attributes `prices`, `funding`, `spot`, `swap`, `adl`, `mgn`, `calls` (list of `(op, coin, qty)`), and helpers `fail_next(op, n=1)`, `partial_next(op, fraction)`, `trigger_adl(coin, fraction)`. `op` is one of `"sell_swap" | "buy_swap" | "buy_spot" | "sell_spot"`. `OrderResult.filled` is gross base quantity for spot (the fee is deducted from the credited balance).

- [ ] **Step 1: Write the failing test**

```python
import pytest

from carry.fake_exchange import FakeExchange


def test_buy_spot_credits_net_of_fee():
    ex = FakeExchange()
    r = ex.buy_spot("BTC", 0.01)
    assert r.ok and r.filled == 0.01 and r.avg_price == 60000.0
    assert ex.spot_balance("BTC") == pytest.approx(0.01 * 0.999)


def test_swap_open_close_and_reduce_only():
    ex = FakeExchange()
    ex.sell_swap("BTC", 1.5)
    assert ex.swap_contracts("BTC") == 1.5
    r = ex.buy_swap("BTC", 2.0)          # cannot buy more than the short
    assert r.filled == 1.5 and ex.swap_contracts("BTC") == 0.0


def test_sell_spot_capped_at_balance():
    ex = FakeExchange()
    ex.buy_spot("ETH", 1.0)
    r = ex.sell_spot("ETH", 5.0)
    assert r.filled == pytest.approx(0.999) and ex.spot_balance("ETH") == pytest.approx(0.0)


def test_fail_next_counts_down():
    ex = FakeExchange()
    ex.fail_next("sell_swap", 2)
    assert not ex.sell_swap("BTC", 1.0).ok
    assert not ex.sell_swap("BTC", 1.0).ok
    assert ex.sell_swap("BTC", 1.0).ok
    assert ex.swap_contracts("BTC") == 1.0


def test_partial_next_applies_once():
    ex = FakeExchange()
    ex.partial_next("sell_swap", 0.5)
    assert ex.sell_swap("BTC", 2.0).filled == 1.0
    assert ex.sell_swap("BTC", 2.0).filled == 2.0


def test_trigger_adl_shrinks_perp():
    ex = FakeExchange()
    ex.sell_swap("BTC", 1.5)
    ex.trigger_adl("BTC", 0.3)
    assert ex.swap_contracts("BTC") == pytest.approx(1.05)


def test_funding_history_returns_last_n():
    ex = FakeExchange(funding={"BTC": [1, 2, 3, 4]})
    assert ex.funding_history("BTC", 2) == [3, 4]
    assert ex.funding_history("ETH", 2) == []


def test_calls_are_logged():
    ex = FakeExchange()
    ex.buy_spot("BTC", 0.01)
    ex.sell_swap("BTC", 1.0)
    assert [(c[0], c[1]) for c in ex.calls] == [("buy_spot", "BTC"), ("sell_swap", "BTC")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_fake_exchange.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.fake_exchange'`

- [ ] **Step 3: Write minimal implementation**

`src/carry/exchange.py`:

```python
"""The exchange surface the engine depends on (real OKX client comes in Phase 4)."""
from __future__ import annotations

from typing import Protocol

from carry.types import OrderResult


class Exchange(Protocol):
    def price(self, coin: str) -> float: ...
    def funding_history(self, coin: str, n: int) -> list[float]: ...
    def spot_balance(self, coin: str) -> float: ...
    def swap_contracts(self, coin: str) -> float: ...
    def swap_adl_rank(self, coin: str) -> int: ...
    def mgn_ratio(self) -> float | None: ...
    def sell_swap(self, coin: str, contracts: float) -> OrderResult: ...
    def buy_swap(self, coin: str, contracts: float) -> OrderResult: ...
    def buy_spot(self, coin: str, qty: float) -> OrderResult: ...
    def sell_spot(self, coin: str, qty: float) -> OrderResult: ...
```

`src/carry/fake_exchange.py`:

```python
"""In-memory exchange with fault injection, for tests and dry runs."""
from __future__ import annotations

from carry.types import OrderResult


class FakeExchange:
    def __init__(self, prices: dict[str, float] | None = None,
                 funding: dict[str, list[float]] | None = None,
                 spot_fee: float = 0.001, mgn: float | None = 10.0) -> None:
        self.prices = prices or {"BTC": 60000.0, "ETH": 3000.0}
        self.funding = funding or {}
        self.spot_fee = spot_fee
        self.mgn = mgn
        self.spot: dict[str, float] = {}
        self.swap: dict[str, float] = {}
        self.adl: dict[str, int] = {}
        self.calls: list[tuple[str, str, float]] = []
        self._fail: dict[str, int] = {}
        self._partial: dict[str, float] = {}

    # --- fault injection -------------------------------------------------
    def fail_next(self, op: str, n: int = 1) -> None:
        self._fail[op] = n

    def partial_next(self, op: str, fraction: float) -> None:
        self._partial[op] = fraction

    def trigger_adl(self, coin: str, fraction: float) -> None:
        self.swap[coin] = round(self.swap.get(coin, 0.0) * (1.0 - fraction), 10)

    def _gate(self, op: str, coin: str, qty: float) -> OrderResult | None:
        self.calls.append((op, coin, qty))
        if self._fail.get(op, 0) > 0:
            self._fail[op] -= 1
            return OrderResult(False, error=f"injected {op} failure")
        return None

    def _frac(self, op: str) -> float:
        return self._partial.pop(op, 1.0)

    # --- market data / account -------------------------------------------
    def price(self, coin: str) -> float:
        return self.prices[coin]

    def funding_history(self, coin: str, n: int) -> list[float]:
        return list(self.funding.get(coin, []))[-n:]

    def spot_balance(self, coin: str) -> float:
        return self.spot.get(coin, 0.0)

    def swap_contracts(self, coin: str) -> float:
        return self.swap.get(coin, 0.0)

    def swap_adl_rank(self, coin: str) -> int:
        return self.adl.get(coin, 0)

    def mgn_ratio(self) -> float | None:
        return self.mgn

    # --- orders (market, fill at current price) --------------------------
    def sell_swap(self, coin: str, contracts: float) -> OrderResult:
        bad = self._gate("sell_swap", coin, contracts)
        if bad:
            return bad
        q = contracts * self._frac("sell_swap")
        self.swap[coin] = self.swap.get(coin, 0.0) + q
        return OrderResult(True, q, self.prices[coin])

    def buy_swap(self, coin: str, contracts: float) -> OrderResult:
        bad = self._gate("buy_swap", coin, contracts)
        if bad:
            return bad
        q = min(contracts * self._frac("buy_swap"), self.swap.get(coin, 0.0))
        self.swap[coin] = self.swap.get(coin, 0.0) - q
        return OrderResult(True, q, self.prices[coin])

    def buy_spot(self, coin: str, qty: float) -> OrderResult:
        bad = self._gate("buy_spot", coin, qty)
        if bad:
            return bad
        q = qty * self._frac("buy_spot")
        self.spot[coin] = self.spot.get(coin, 0.0) + q * (1.0 - self.spot_fee)
        return OrderResult(True, q, self.prices[coin])

    def sell_spot(self, coin: str, qty: float) -> OrderResult:
        bad = self._gate("sell_spot", coin, qty)
        if bad:
            return bad
        q = min(qty * self._frac("sell_spot"), self.spot.get(coin, 0.0))
        self.spot[coin] = self.spot.get(coin, 0.0) - q
        return OrderResult(True, q, self.prices[coin])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_fake_exchange.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/exchange.py src/carry/fake_exchange.py tests/test_carry_fake_exchange.py
git commit -m "feat(carry): Exchange protocol and fault-injecting FakeExchange

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Executor (open with rollback, close)

**Files:**
- Create: `src/carry/executor.py`
- Test: `tests/test_carry_executor.py`

**Interfaces:**
- Consumes: `Exchange`, `FakeExchange` (tests), `Instrument`, `DEFAULT_INSTRUMENTS`, `CarryConfig`, `plan_position`, `floor_to` (sizing), `CoinLeg` (state)
- Produces: `HedgeError(RuntimeError)`; `Executor(ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig, alert: Callable[[str], None])` with `open(coin: str, notional: float, now_ms: int) -> CoinLeg | None` and `close(coin: str, leg: CoinLeg) -> bool`. `open`: raises `ValueError` if `notional > cfg.max_order_notional`; opens the perp short first, then buys spot; on any failure of the second leg rolls back the first (returns `None`); if the rollback itself fails it alerts `[CRITICAL] ...` and raises `HedgeError`. `close`: sells spot first (quantity = `min(balance, leg.spot_qty)` floored to the spot lot), then buys back `min(perp contracts, leg.contracts)`; each order is retried once; returns True only if both legs are closed (a leg below its minimum size counts as already closed).

- [ ] **Step 1: Write the failing test**

```python
import pytest

from carry.config import CarryConfig
from carry.executor import Executor, HedgeError
from carry.fake_exchange import FakeExchange
from carry.types import DEFAULT_INSTRUMENTS


def make():
    ex = FakeExchange()
    alerts: list[str] = []
    return ex, Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(), alerts.append), alerts


def ops(ex, name):
    return [c for c in ex.calls if c[0] == name]


def test_open_hedged_both_legs():
    ex, exe, _ = make()
    leg = exe.open("BTC", 125.0, now_ms=1000)
    assert leg is not None
    assert leg.contracts == pytest.approx(0.20)
    assert ex.swap["BTC"] == pytest.approx(0.20)
    assert ex.spot["BTC"] == pytest.approx(0.002, rel=1e-4)
    assert leg.spot_qty == pytest.approx(ex.spot["BTC"])
    assert leg.entry_ts == 1000


def test_open_rejects_notional_above_cap():
    _, exe, _ = make()
    with pytest.raises(ValueError):
        exe.open("BTC", 301.0, now_ms=0)


def test_spot_failure_rolls_back_the_perp():
    ex, exe, alerts = make()
    ex.fail_next("buy_spot")
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert ex.swap["BTC"] == pytest.approx(0.0)
    assert ex.spot.get("BTC", 0.0) == 0.0
    assert alerts


def test_partial_perp_fill_is_undone_before_buying_spot():
    ex, exe, _ = make()
    ex.partial_next("sell_swap", 0.5)
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert ex.swap["BTC"] == pytest.approx(0.0)
    assert ops(ex, "buy_spot") == []


def test_failed_rollback_raises_hedge_error_and_alerts_critical():
    ex, exe, alerts = make()
    ex.fail_next("buy_spot")
    ex.fail_next("buy_swap", 2)
    with pytest.raises(HedgeError):
        exe.open("BTC", 125.0, now_ms=0)
    assert any("CRITICAL" in a for a in alerts)
    assert ex.swap["BTC"] > 0            # naked short is still there; caller must halt


def test_close_flattens_both_legs():
    ex, exe, _ = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    assert exe.close("BTC", leg) is True
    assert ex.swap["BTC"] == pytest.approx(0.0)
    assert ex.spot["BTC"] < 1e-5


def test_close_after_adl_only_sells_spot():
    ex, exe, _ = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    ex.trigger_adl("BTC", 1.0)
    assert exe.close("BTC", leg) is True
    assert ex.spot["BTC"] < 1e-5
    assert ops(ex, "buy_swap") == []


def test_close_failure_returns_false_and_keeps_spot():
    ex, exe, _ = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    ex.fail_next("sell_spot", 2)
    assert exe.close("BTC", leg) is False
    assert ex.spot["BTC"] > 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_executor.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.executor'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Open/close a hedged position with rollback on partial failure."""
from __future__ import annotations

from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.sizing import floor_to, plan_position
from carry.state import CoinLeg
from carry.types import Instrument, OrderResult


class HedgeError(RuntimeError):
    """Raised when a rollback fails and an unhedged leg may remain open."""


def _retry(fn: Callable[[], OrderResult], tries: int = 2) -> OrderResult:
    r = fn()
    for _ in range(tries - 1):
        if r.ok:
            break
        r = fn()
    return r


class Executor:
    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 alert: Callable[[str], None]) -> None:
        self.ex, self.insts, self.cfg, self.alert = ex, insts, cfg, alert

    def open(self, coin: str, notional: float, now_ms: int) -> CoinLeg | None:
        if notional > self.cfg.max_order_notional:
            raise ValueError(f"notional {notional} exceeds max_order_notional {self.cfg.max_order_notional}")
        inst = self.insts[coin]
        sz = plan_position(notional, self.ex.price(coin), inst, self.cfg.spot_taker_fee)

        short = self.ex.sell_swap(coin, sz.contracts)          # perp first: worst case is an extra short
        if not short.ok or short.filled <= 0:
            self.alert(f"open {coin}: perp order failed ({short.error})")
            return None
        if short.filled < sz.contracts - 1e-9:
            self.alert(f"open {coin}: partial perp fill {short.filled}/{sz.contracts}, undoing")
            self._undo_short(coin, short.filled)
            return None

        buy = self.ex.buy_spot(coin, sz.spot_qty)
        if not buy.ok or buy.filled < sz.spot_qty - 1e-12:
            self.alert(f"open {coin}: spot leg failed ({buy.error or 'partial'}), rolling back")
            self._sell_all_spot(coin)
            self._undo_short(coin, short.filled)
            return None
        return CoinLeg(contracts=short.filled, spot_qty=self.ex.spot_balance(coin), entry_ts=now_ms)

    def close(self, coin: str, leg: CoinLeg) -> bool:
        inst = self.insts[coin]
        ok = True
        spot_qty = floor_to(min(self.ex.spot_balance(coin), leg.spot_qty), inst.spot_lot)
        if spot_qty >= inst.spot_min:
            r = _retry(lambda: self.ex.sell_spot(coin, spot_qty))
            ok = ok and r.ok and r.filled > 0
        contracts = floor_to(min(self.ex.swap_contracts(coin), leg.contracts), inst.swap_lot)
        if contracts >= inst.swap_min:
            r = _retry(lambda: self.ex.buy_swap(coin, contracts))
            ok = ok and r.ok and r.filled > 0
        return ok

    # --- helpers ---------------------------------------------------------
    def _sell_all_spot(self, coin: str) -> None:
        inst = self.insts[coin]
        qty = floor_to(self.ex.spot_balance(coin), inst.spot_lot)
        if qty >= inst.spot_min:
            _retry(lambda: self.ex.sell_spot(coin, qty))

    def _undo_short(self, coin: str, contracts: float) -> None:
        r = _retry(lambda: self.ex.buy_swap(coin, contracts))
        if not r.ok:
            msg = f"[CRITICAL] {coin}: could not undo {contracts} short contracts ({r.error}); naked short open"
            self.alert(msg)
            raise HedgeError(msg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_executor.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/executor.py tests/test_carry_executor.py
git commit -m "feat(carry): executor with two-leg open, rollback and close

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Watchdog

**Files:**
- Create: `src/carry/watchdog.py`
- Test: `tests/test_carry_watchdog.py`

**Interfaces:**
- Consumes: `Exchange`, `Instrument`, `CarryConfig`, `Executor` (+ `open`, `close`), `CarryState`, `check_hedge`, `HedgeStatus`, `FakeExchange`/`DEFAULT_INSTRUMENTS` (tests)
- Produces: `Watchdog(ex, insts, cfg, executor, alert)` with `check(state: CarryState) -> list[str]`. For every coin in `state.legs`: if `swap_adl_rank >= cfg.adl_alert_rank` it alerts once per rank change (no close); if the hedge status is not `OK` OR the perp is smaller than the recorded `leg.contracts` (any shrink, i.e. ADL/liquidation) it alerts `[CRITICAL]`, calls `executor.close` (spot sold first), removes the leg on success, and sets `state.mode = "HALTED"` with a reason (also on failure, keeping the leg). Returns human-readable action strings.

- [ ] **Step 1: Write the failing test**

```python
from carry.config import CarryConfig
from carry.executor import Executor
from carry.fake_exchange import FakeExchange
from carry.state import CarryState
from carry.types import DEFAULT_INSTRUMENTS
from carry.watchdog import Watchdog


def setup():
    ex = FakeExchange()
    alerts: list[str] = []
    cfg = CarryConfig()
    exe = Executor(ex, DEFAULT_INSTRUMENTS, cfg, alerts.append)
    wd = Watchdog(ex, DEFAULT_INSTRUMENTS, cfg, exe, alerts.append)
    st = CarryState(mode="HEDGED")
    for coin in ("BTC", "ETH"):
        st.legs[coin] = exe.open(coin, 125.0, 0)
    return ex, wd, st, alerts


def test_healthy_hedge_does_nothing():
    ex, wd, st, alerts = setup()
    assert wd.check(st) == []
    assert st.mode == "HEDGED" and set(st.legs) == {"BTC", "ETH"} and alerts == []


def test_full_adl_sells_spot_and_halts():
    ex, wd, st, alerts = setup()
    ex.trigger_adl("BTC", 1.0)
    actions = wd.check(st)
    assert any("BTC" in a for a in actions)
    assert "BTC" not in st.legs and "ETH" in st.legs
    assert ex.spot["BTC"] < 1e-5
    assert st.mode == "HALTED" and "BTC" in st.halted_reason
    assert any("CRITICAL" in a for a in alerts)


def test_partial_adl_closes_everything_for_that_coin():
    ex, wd, st, _ = setup()
    ex.trigger_adl("ETH", 0.3)
    wd.check(st)
    assert "ETH" not in st.legs
    assert ex.spot["ETH"] < 1e-4 and ex.swap["ETH"] < 0.01
    assert st.mode == "HALTED"


def test_single_lot_shrink_counts_as_adl():
    ex, wd, st, _ = setup()
    ex.swap["BTC"] = st.legs["BTC"].contracts - 0.01
    wd.check(st)
    assert "BTC" not in st.legs and st.mode == "HALTED"


def test_lost_spot_closes_the_naked_short():
    ex, wd, st, _ = setup()
    ex.spot["BTC"] = 0.0
    wd.check(st)
    assert "BTC" not in st.legs and ex.swap["BTC"] < 0.01


def test_adl_rank_alerts_once_and_does_not_close():
    ex, wd, st, alerts = setup()
    ex.adl["BTC"] = 5
    wd.check(st)
    wd.check(st)
    assert sum("ADL rank 5" in a for a in alerts) == 1
    assert set(st.legs) == {"BTC", "ETH"} and st.mode == "HEDGED"


def test_failed_close_keeps_leg_and_halts_with_critical_alert():
    ex, wd, st, alerts = setup()
    ex.trigger_adl("BTC", 1.0)
    ex.fail_next("sell_spot", 2)
    actions = wd.check(st)
    assert any("FAILED" in a for a in actions)
    assert "BTC" in st.legs and st.mode == "HALTED"
    assert sum("CRITICAL" in a for a in alerts) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_watchdog.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.watchdog'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Hedge-integrity watchdog: an ADL/liquidation of the perp leg flattens the spot leg at once."""
from __future__ import annotations

from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.executor import Executor
from carry.hedge import HedgeStatus, check_hedge
from carry.state import CarryState
from carry.types import Instrument


class Watchdog:
    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 executor: Executor, alert: Callable[[str], None]) -> None:
        self.ex, self.insts, self.cfg = ex, insts, cfg
        self.executor, self.alert = executor, alert
        self._last_rank: dict[str, int] = {}

    def check(self, state: CarryState) -> list[str]:
        actions: list[str] = []
        for coin in list(state.legs):
            leg = state.legs[coin]
            inst = self.insts[coin]

            rank = self.ex.swap_adl_rank(coin)
            if rank >= self.cfg.adl_alert_rank and self._last_rank.get(coin) != rank:
                self.alert(f"ADL rank {rank} on {coin} (alert only, hedge intact)")
            self._last_rank[coin] = rank

            contracts = self.ex.swap_contracts(coin)
            spot = self.ex.spot_balance(coin)
            status = check_hedge(spot, contracts, inst, self.cfg.drift_tol)
            shrunk = contracts < leg.contracts - 1e-9
            if status == HedgeStatus.OK and not shrunk:
                continue

            self.alert(f"[CRITICAL] hedge broken on {coin}: status={status.value}, "
                       f"perp {contracts} vs recorded {leg.contracts}, spot {spot}; flattening")
            if self.executor.close(coin, leg):
                del state.legs[coin]
                actions.append(f"closed {coin} (hedge broken)")
            else:
                self.alert(f"[CRITICAL] FAILED to flatten {coin}; manual action needed")
                actions.append(f"FAILED to close {coin}")
            state.mode = "HALTED"
            state.halted_reason = f"hedge broken on {coin}"
        return actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_watchdog.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/watchdog.py tests/test_carry_watchdog.py
git commit -m "feat(carry): ADL/hedge-integrity watchdog

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Engine (one loop iteration, force/rearm, end-to-end scenarios)

**Files:**
- Create: `src/carry/engine.py`
- Test: `tests/test_carry_engine.py`

**Interfaces:**
- Consumes: everything above. `decide(rates, in_position, held, cfg)`, `classify`, `Tier`, `check_hedge`, `HedgeStatus`, `Executor`, `HedgeError`, `Watchdog`, `CarryState`, `CoinLeg`.
- Produces: `Engine(ex, insts, cfg, alert, state: CarryState | None = None, save: Callable[[CarryState], None] | None = None)` with `step(now_ms: int) -> list[str]`, `force_open(coin, notional, now_ms) -> bool`, `force_close(coin) -> bool`, `rearm() -> None`, attribute `state`. `step` order: (1) if not halted, any coin with no recorded leg whose exchange holdings are not FLAT → halt (alert, NO trading); (2) watchdog on recorded legs; (3) if halted return; (4) risk tier (alert on tier change; `>= CLOSE` closes all legs and halts); (5) per coin: `decide` → open only when tier is `NORMAL`, close when signalled; `held = int((now_ms - leg.entry_ts) / 28_800_000)`. A `HedgeError` anywhere halts the engine. `save(state)` is called at the end of every `step`. Mode: `FLAT` when no legs and not halted, `HEDGED` when legs exist, `HALTED` after any watchdog/risk/unexpected-holdings halt. A halted engine keeps running the watchdog on residual legs but never opens.

- [ ] **Step 1: Write the failing test**

```python
import copy

import pytest

from carry.config import CarryConfig
from carry.engine import Engine
from carry.fake_exchange import FakeExchange
from carry.types import DEFAULT_INSTRUMENTS

DAY = 86_400_000
HIGH = {"BTC": [0.0001] * 9, "ETH": [0.0001] * 9}     # 10.95% APR
ZERO = {"BTC": [0.0] * 9, "ETH": [0.0] * 9}


def make(funding=HIGH, save=None):
    ex = FakeExchange(funding=dict(funding))
    alerts: list[str] = []
    eng = Engine(ex, DEFAULT_INSTRUMENTS, CarryConfig(), alerts.append, save=save)
    return ex, eng, alerts


def sent_orders(ex):
    return [c for c in ex.calls if c[0] in ("sell_swap", "buy_swap", "buy_spot", "sell_spot")]


def test_opens_both_coins_when_funding_is_high():
    ex, eng, _ = make()
    eng.step(0)
    assert set(eng.state.legs) == {"BTC", "ETH"} and eng.state.mode == "HEDGED"
    assert ex.swap["BTC"] == pytest.approx(0.20) and ex.swap["ETH"] == pytest.approx(0.41)


def test_does_not_open_when_funding_is_low():
    ex, eng, _ = make(funding={"BTC": [0.00005] * 9, "ETH": [0.00005] * 9})
    eng.step(0)
    assert eng.state.legs == {} and eng.state.mode == "FLAT" and sent_orders(ex) == []


def test_steady_state_makes_no_further_orders():
    ex, eng, _ = make()
    eng.step(0)
    n = len(sent_orders(ex))
    eng.step(1000)
    assert len(sent_orders(ex)) == n


def test_exit_respects_min_hold_then_closes():
    ex, eng, _ = make()
    eng.step(0)
    ex.funding = dict(ZERO)
    eng.step(1 * DAY)                       # held = 3 < 9
    assert set(eng.state.legs) == {"BTC", "ETH"}
    eng.step(3 * DAY)                       # held = 9
    assert eng.state.legs == {} and eng.state.mode == "FLAT"
    assert ex.swap["BTC"] == pytest.approx(0.0)


def test_adl_flattens_that_coin_halts_and_never_reopens():
    ex, eng, alerts = make()
    eng.step(0)
    ex.trigger_adl("BTC", 1.0)
    eng.step(1000)
    assert "BTC" not in eng.state.legs and "ETH" in eng.state.legs
    assert eng.state.mode == "HALTED"
    assert ex.spot["BTC"] < 1e-5
    n = len(sent_orders(ex))
    eng.step(2000)
    assert len(sent_orders(ex)) == n        # no re-entry while halted
    assert any("CRITICAL" in a for a in alerts)


def test_risk_close_tier_flattens_everything_and_halts():
    ex, eng, _ = make()
    eng.step(0)
    ex.mgn = 2.0                            # <= mgn_close (2.5)
    eng.step(1000)
    assert eng.state.legs == {} and eng.state.mode == "HALTED"
    assert ex.swap["BTC"] == pytest.approx(0.0) and ex.swap["ETH"] == pytest.approx(0.0)


def test_warn_tier_alerts_but_keeps_positions():
    ex, eng, alerts = make()
    eng.step(0)
    ex.mgn = 3.0
    eng.step(1000)
    assert set(eng.state.legs) == {"BTC", "ETH"} and eng.state.mode == "HEDGED"
    assert any("WARN" in a for a in alerts)


def test_one_leg_failure_rolls_back_without_halting():
    ex, eng, _ = make()
    ex.fail_next("buy_spot")                # BTC spot leg fails once, ETH succeeds
    eng.step(0)
    assert "BTC" not in eng.state.legs and "ETH" in eng.state.legs
    assert ex.swap["BTC"] == pytest.approx(0.0)
    assert eng.state.mode == "HEDGED"


def test_failed_rollback_halts_with_critical_alert():
    ex, eng, alerts = make()
    ex.fail_next("buy_spot")
    ex.fail_next("buy_swap", 2)
    eng.step(0)
    assert eng.state.mode == "HALTED" and eng.state.legs == {}
    assert any("CRITICAL" in a for a in alerts)


def test_unexpected_holdings_halt_without_trading():
    ex, eng, alerts = make()
    ex.spot["BTC"] = 0.5
    eng.step(0)
    assert eng.state.mode == "HALTED" and sent_orders(ex) == []
    assert any("unexpected" in a for a in alerts)


def test_force_open_and_force_close_bypass_the_signal():
    ex, eng, _ = make(funding={"BTC": [0.0] * 9, "ETH": [0.0] * 9})
    assert eng.force_open("BTC", 100.0, now_ms=0) is True
    assert "BTC" in eng.state.legs and eng.state.mode == "HEDGED"
    eng.step(1000)                          # signal says hold (held < min_hold): still open
    assert "BTC" in eng.state.legs
    assert eng.force_close("BTC") is True
    assert eng.state.legs == {} and eng.state.mode == "FLAT"


def test_force_open_refused_while_halted_and_rearm_rules():
    ex, eng, _ = make()
    eng.step(0)
    ex.trigger_adl("BTC", 1.0)
    eng.step(1000)                          # BTC flattened, ETH leg remains, HALTED
    with pytest.raises(RuntimeError):
        eng.force_open("BTC", 100.0, now_ms=2000)
    with pytest.raises(RuntimeError):
        eng.rearm()                         # ETH leg still open
    assert eng.force_close("ETH") is True
    eng.rearm()
    assert eng.state.mode == "FLAT" and eng.state.halted_reason == ""


def test_state_is_saved_after_every_step():
    saved = []
    ex, eng, _ = make(save=lambda s: saved.append(copy.deepcopy(s)))
    eng.step(0)
    eng.step(1000)
    assert len(saved) == 2 and saved[-1].mode == "HEDGED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_carry_engine.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
"""One loop iteration of the carry strategy: guard -> watchdog -> risk -> signal -> execute."""
from __future__ import annotations

from typing import Callable

from carry.config import CarryConfig
from carry.decision import decide
from carry.exchange import Exchange
from carry.executor import Executor, HedgeError
from carry.hedge import HedgeStatus, check_hedge
from carry.risk import Tier, classify
from carry.state import CarryState, CoinLeg
from carry.types import Instrument
from carry.watchdog import Watchdog

SETTLEMENT_MS = 8 * 3600 * 1000


class Engine:
    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 alert: Callable[[str], None], state: CarryState | None = None,
                 save: Callable[[CarryState], None] | None = None) -> None:
        self.ex, self.insts, self.cfg, self.alert = ex, insts, cfg, alert
        self.state = state or CarryState()
        self.save = save
        self.executor = Executor(ex, insts, cfg, alert)
        self.watchdog = Watchdog(ex, insts, cfg, self.executor, alert)
        self._last_tier = Tier.NORMAL

    # --- public ----------------------------------------------------------
    def step(self, now_ms: int) -> list[str]:
        try:
            out = self._step(now_ms)
        except HedgeError as e:
            self._halt(str(e))
            out = [f"HALT {e}"]
        if self.save:
            self.save(self.state)
        return out

    def force_open(self, coin: str, notional: float, now_ms: int) -> bool:
        if self.state.mode == "HALTED":
            raise RuntimeError("engine is halted; rearm() first")
        if coin in self.state.legs:
            raise RuntimeError(f"{coin} already open")
        try:
            return self._open(coin, notional, now_ms) is not None
        except HedgeError as e:
            self._halt(str(e))
            return False

    def force_close(self, coin: str) -> bool:
        leg = self.state.legs.get(coin)
        if leg is None:
            return True
        if not self.executor.close(coin, leg):
            self._halt(f"close {coin} failed")
            return False
        del self.state.legs[coin]
        if not self.state.legs and self.state.mode == "HEDGED":
            self.state.mode = "FLAT"
        return True

    def rearm(self) -> None:
        if self.state.legs:
            raise RuntimeError("legs still open; close them before rearm()")
        for coin in self.cfg.coins:
            inst = self.insts[coin]
            st = check_hedge(self.ex.spot_balance(coin), self.ex.swap_contracts(coin), inst, self.cfg.drift_tol)
            if st != HedgeStatus.FLAT:
                raise RuntimeError(f"{coin} holdings on the exchange ({st.value}); resolve manually")
        self.state.mode = "FLAT"
        self.state.halted_reason = ""

    # --- internals -------------------------------------------------------
    def _halt(self, reason: str) -> None:
        self.state.mode = "HALTED"
        self.state.halted_reason = reason
        self.alert(f"[CRITICAL] HALTED: {reason}")

    def _open(self, coin: str, notional: float, now_ms: int) -> CoinLeg | None:
        leg = self.executor.open(coin, notional, now_ms)
        if leg is not None:
            self.state.legs[coin] = leg
            self.state.mode = "HEDGED"
        return leg

    def _step(self, now_ms: int) -> list[str]:
        s = self.state
        out: list[str] = []

        if s.mode != "HALTED":
            for coin in self.cfg.coins:
                if coin in s.legs:
                    continue
                st = check_hedge(self.ex.spot_balance(coin), self.ex.swap_contracts(coin),
                                 self.insts[coin], self.cfg.drift_tol)
                if st != HedgeStatus.FLAT:
                    self._halt(f"unexpected {coin} holdings ({st.value}) with no recorded position; not trading")
                    return ["HALT unexpected holdings"]

        out += self.watchdog.check(s)
        if s.mode == "HALTED":
            return out

        tier = classify(self.ex.mgn_ratio(), bool(s.legs), self.cfg)
        if tier != self._last_tier:
            self.alert(f"risk tier {self._last_tier.name} -> {tier.name}")
            self._last_tier = tier
        if tier >= Tier.CLOSE:
            for coin in list(s.legs):
                if self.executor.close(coin, s.legs[coin]):
                    del s.legs[coin]
                    out.append(f"risk-closed {coin}")
            self._halt(f"risk tier {tier.name}")
            return out

        for coin in self.cfg.coins:
            leg = s.legs.get(coin)
            held = int((now_ms - leg.entry_ts) / SETTLEMENT_MS) if leg else 0
            action = decide(self.ex.funding_history(coin, self.cfg.window), leg is not None, held, self.cfg)
            if action == "open" and tier == Tier.NORMAL:
                out.append(f"opened {coin}" if self._open(coin, self.cfg.per_coin_notional, now_ms)
                           else f"open {coin} failed")
            elif action == "close" and leg is not None:
                if self.executor.close(coin, leg):
                    del s.legs[coin]
                    out.append(f"closed {coin}")
                else:
                    self._halt(f"close {coin} failed")
                    return out
        if not s.legs and s.mode == "HEDGED":
            s.mode = "FLAT"
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_carry_engine.py -q`
Expected: 13 passed. Then run the whole carry suite: `$PY -m pytest tests/test_carry_*.py -q` — expected all pass (the pre-existing 26 tests plus this phase's new ones).

- [ ] **Step 5: Commit**

```bash
git add src/carry/engine.py tests/test_carry_engine.py
git commit -m "feat(carry): engine step loop with risk, watchdog, force/rearm

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Phase 4-5 (not in this plan)

Written after this phase completes and once an OKX demo key exists: real `OkxExchange` implementing the `Exchange` protocol (python-okx; spot `tdMode` `cross` vs `cash` and market-buy `tgtCcy='base_ccy'` verified on demo; swap `tdMode='cross'`, net mode; `mgnRatio` unit and thresholds measured; funding via `funding_rate_history`), CLI (`run`, `force-open`, `force-close`, `rearm`, `status`), systemd unit and `~/.secrets/carry.env` loading, alert channel implementation, WebSocket `adl-warning` (optional), preflight (account flat, `acctLv=3`, `autoLoan=false`, sufficient USDT), demo integration run, then small live (200-300 USDT) with a forced smoke-test open/close.
