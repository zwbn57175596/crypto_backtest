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
