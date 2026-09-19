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
    ex.fail_next("buy_swap", 3)          # executor tries 3 times
    eng.step(0)
    assert eng.state.mode == "HALTED" and eng.state.legs == {}
    assert any("CRITICAL" in a for a in alerts)


def test_unexpected_exchange_error_halts():
    ex, eng, alerts = make()

    def boom():
        raise ConnectionError("boom")

    ex.mgn_ratio = boom
    eng.step(0)
    assert eng.state.mode == "HALTED"
    assert any("boom" in a for a in alerts)


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
