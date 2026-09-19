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
