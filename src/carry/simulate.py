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
    lev_min_spread: float = 0.03  # go to max_leverage only if funding_apr - borrow_apr >= this
    lev_exit_spread: float = 0.015  # once levered, drop to 1.0x only below this (<= lev_min_spread)
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
    lev_changes: int = 0          # 1.0x <-> max_leverage transitions (not entries/exits)


def _pick_leverage(avg_apr: float, p: CarryParams, current_lev: float) -> float:
    """Leverage decision with hysteresis: enter levered at lev_min_spread, leave at lev_exit_spread."""
    if p.max_leverage <= 1.0:
        return 1.0
    spread = avg_apr - p.borrow_apr
    if current_lev == p.max_leverage:
        return p.max_leverage if spread >= p.lev_exit_spread else 1.0
    return p.max_leverage if spread >= p.lev_min_spread else 1.0


def simulate_carry(rates: Sequence[float], p: CarryParams) -> CarryResult:
    n = len(rates)
    if n <= p.window:
        return CarryResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    borrow_pp = borrow_cost_per_period(p.borrow_apr)
    cost_per_unit = p.fee_side + p.basis_slip
    equity = peak = max_dd = 0.0
    lev = 0.0
    held = in_mkt = trips = lev_changes = 0
    lev_sum = 0.0

    for i in range(p.window, n):
        avg_apr = funding_apr(sum(rates[i - p.window:i]) / p.window)
        target = lev
        if lev == 0.0:
            if avg_apr >= p.enter_apr:
                target = _pick_leverage(avg_apr, p, 0.0)
        elif avg_apr >= p.exit_apr:
            target = _pick_leverage(avg_apr, p, lev)
        elif held >= p.min_hold:
            target = 0.0

        equity -= abs(target - lev) * cost_per_unit
        if lev == 0.0 and target > 0.0:
            trips += 1
            held = 0
        elif lev > 0.0 and target > 0.0 and target != lev:
            lev_changes += 1
        lev = target

        if lev > 0.0:
            equity += lev * rates[i] - (lev - 1.0) * borrow_pp
            held += 1
            in_mkt += 1
            lev_sum += lev
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    if lev > 0.0:                      # residual position is closed at series end
        equity -= lev * cost_per_unit
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
        lev_changes=lev_changes,
    )
