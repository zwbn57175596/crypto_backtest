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
