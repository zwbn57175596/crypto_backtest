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
