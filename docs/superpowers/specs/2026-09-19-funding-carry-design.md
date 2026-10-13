# Funding-Rate Carry (Spot Long + Perp Short) — Design

Date: 2026-09-19 · Branch: `feat/funding-carry` · Status: draft, pending user review

## 1. Goal

Delta-neutral carry on OKX: hold spot long (BTC, ETH) and an equal-notional USDT-margined
perpetual short. Earn the funding rate. Optionally lever the long leg (~1.5x) by borrowing
USDT; the extra leverage is only worth it while `funding_apr - borrow_apr > 0` after fees.

Capital: 1000 USDT initially. At 1.5x: ~1500 USDT notional per leg, ~500 USDT borrowed.
Scope: BTC and ETH only. Account: OKX multi-currency margin (acctLv=3, to be verified via
`GET /api/v5/account/config`). Portfolio margin (acctLv=4) is a later optimisation.

## 2. Non-goals

Multi-exchange, altcoins, option hedges, cross-exchange arbitrage, GUI, portfolio-margin
specific logic (first version), reuse of `live_engine` / `LiveExchange`.

## 3. Phases and go/no-go gates

1. **Research** — verify acctLv, borrow rates/limits, `mgnRatio` unit and liquidation
   line, fees, lot sizes, contract values, ADL fields/channel. Output: `docs/superpowers/specs/2026-09-19-funding-carry-research.md`.
2. **Viability** — historical funding vs borrow rate, net APR after fees, drawdown.
   Gate: net APR must be clearly positive under conservative borrow-rate scenarios,
   otherwise stop before writing execution code.
3. **Core logic + tests** — signal, sizing, risk tiers, watchdog state machine, FakeOKX.
4. **OKX demo integration** — full open/hold/ADL-trigger/close cycle.
5. **Small live** — 200–300 USDT, then scale to 1000.

Known unknowns to resolve in phase 1: OKX `funding-rate-history` depth (believed short;
Binance history may be used as proxy), no borrow-rate history (use scenarios),
whether `mgnRatio` under acctLv=3 stays far above 2.5 for a hedged book (threshold
calibration by shock simulation + demo measurement, not adopting 2.5 blindly).

## 4. Architecture (`src/carry/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `signal.py` | open/close/leverage-tier decision from funding, borrow rate, fees | pure |
| `sizing.py` | notional → spot qty and swap contracts (ctVal, lot rounding), drift calc | pure |
| `risk.py` | `mgnRatio` → tier (NORMAL / WARN / CLOSE / EMERGENCY) | pure |
| `state.py` | position/state persistence (`state.json`), exchange-first reconcile | stdlib |
| `okx_client.py` | thin python-okx wrapper: spot, swap, borrow, funding, account, adl | python-okx |
| `watchdog.py` | ADL detection (WS `adl-warning` + REST `adl` poll + hedge-drift check) | client, state |
| `runner.py` | main loop: reconcile → risk → signal → execute | all |
| `backtest_carry.py` (scripts/) | viability backtest | pure modules |

Pure modules have no network access and are fully unit-testable. `watchdog` and `runner`
run as independent processes/threads so one hanging cannot block the other's ability to flatten.

## 5. Strategy rules

- **Entry**: trailing funding APR > borrow APR + amortised round-trip fees + buffer.
  Round trip is four legs (spot buy/sell, perp open/close); taker ≈ 0.3% total, so
  positions must be sticky: hysteresis on entry/exit thresholds, minimum hold time.
- **Leverage tiers**: 1.5x while `funding - borrow` > threshold; drop to 1.0x when it
  weakens; full exit when funding turns negative or falls below amortised fees.
- **Hedge alignment**: spot in coin qty, swap in contracts (BTC 0.01, ETH 0.1 per
  contract — verify). Rebalance when notional drift exceeds a threshold.
- **Open order**: perp short first, then spot buy. Failure of the second leg → rollback
  the first after timeout. Worst case is an extra short, never an unhedged long.

## 6. Risk tiers (`mgnRatio`; thresholds calibrated in phase 1–4)

| Tier | Condition | Action |
|---|---|---|
| NORMAL | above T1 | follow signal |
| WARN | T2..T1 | no adds; reduce leverage 1.5x → 1.0x (repay borrow) |
| CLOSE | below T2 (user's "2.5" maps here) | close spot, then perp |
| EMERGENCY | below T3, or API down beyond timeout | market close all, alert |

## 7. Watchdog

```
NORMAL --(ADL event | adl rank >=4 | perp size shrinks unexpectedly)--> HEDGE_BROKEN
HEDGE_BROKEN --market sell spot--> FLAT_SPOT --repay borrow--> IDLE (alert; manual re-arm)
```

Three redundant triggers: WS `adl-warning`, REST `adl` field poll, and a per-loop
invariant `perp_contracts * ctVal ≈ spot_qty`. ADL response uses market orders only.
No automatic re-entry after a trigger. Runs independently of `runner`.

## 8. Error handling

- Exponential backoff on rate-limit / network errors; sustained failure → EMERGENCY.
- Exchange is source of truth; `state.json` is a cache, reconciled on every loop and restart.
- Per-order size cap, structured logs, alerts on ADL / close / emergency (channel TBD in plan).

## 9. Testing

- Table-driven unit tests for signal, sizing, risk, state machine (threshold boundaries, hysteresis).
- `FakeOKX` with fault injection: ADL, one-leg failure, rate limit, `mgnRatio` collapse.
- Viability backtest script with fee/borrow-rate sensitivity.
- OKX demo end-to-end run. No automated tests with real funds.

## 10. Open items (resolved in phase 1)

acctLv actual value; borrow rate level/limits for USDT; `mgnRatio` scale; fee tier;
funding-history depth; alert channel; credential handling (env vars, read-only key for probes).
