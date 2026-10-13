import pytest

from carry.config import CarryConfig
from carry.executor import Executor
from carry.fake_exchange import FakeExchange
from carry.state import CarryState
from carry.types import DEFAULT_INSTRUMENTS
from carry.watchdog import Watchdog, WatchdogError


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
    assert st.mode == "HALTED"


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
    ex.fail_next("sell_spot", 3)          # executor tries 3 times
    actions = wd.check(st)
    assert any("FAILED" in a for a in actions)
    assert "BTC" in st.legs and st.mode == "HALTED"
    assert sum("CRITICAL" in a for a in alerts) >= 2
    assert any("FAILED to flatten" in a for a in alerts)


def test_two_coins_broken_in_one_poll():
    ex, wd, st, _ = setup()
    ex.trigger_adl("BTC", 1.0)
    ex.trigger_adl("ETH", 0.3)
    actions = wd.check(st)
    assert len(actions) == 2
    assert st.legs == {} and st.mode == "HALTED"
    assert "BTC" in st.halted_reason and "ETH" in st.halted_reason


def test_one_close_fails_other_succeeds_loop_continues():
    ex, wd, st, alerts = setup()
    ex.trigger_adl("BTC", 1.0)
    ex.trigger_adl("ETH", 0.3)
    ex.fail_next("sell_spot", 3)          # consumed by the first coin the loop reaches (BTC)
    actions = wd.check(st)
    assert len(actions) == 2
    assert set(st.legs) == {"BTC"} and st.mode == "HALTED"
    assert ex.spot["ETH"] < 1e-4          # ETH was still flattened
    assert any("FAILED to flatten BTC" in a for a in alerts)


def test_retry_after_failed_close_succeeds():
    ex, wd, st, _ = setup()
    ex.trigger_adl("BTC", 1.0)
    ex.fail_next("sell_spot", 3)
    wd.check(st)
    assert "BTC" in st.legs
    reason = st.halted_reason
    actions = wd.check(st)
    assert "BTC" not in st.legs and any("closed BTC" in a for a in actions)
    assert st.mode == "HALTED" and st.halted_reason == reason   # reason did not grow


def test_raising_alerter_does_not_skip_flatten():
    ex = FakeExchange()
    cfg = CarryConfig()
    exe_alerts: list[str] = []
    exe = Executor(ex, DEFAULT_INSTRUMENTS, cfg, exe_alerts.append)

    def boom(_msg: str) -> None:
        raise RuntimeError("telegram down")

    wd = Watchdog(ex, DEFAULT_INSTRUMENTS, cfg, exe, boom)
    st = CarryState(mode="HEDGED")
    st.legs["BTC"] = exe.open("BTC", 125.0, 0)
    ex.trigger_adl("BTC", 1.0)
    wd.check(st)
    assert "BTC" not in st.legs and ex.spot["BTC"] < 1e-5
    assert st.mode == "HALTED"


def test_adl_rank_read_failure_does_not_block_integrity_check():
    ex, wd, st, alerts = setup()

    def bad_rank(coin: str) -> int:
        raise ConnectionError("rank endpoint down")

    ex.swap_adl_rank = bad_rank
    ex.trigger_adl("BTC", 1.0)
    wd.check(st)                          # must not raise
    assert "BTC" not in st.legs and "ETH" in st.legs and st.mode == "HALTED"
    assert any("ADL rank unavailable for BTC" in a for a in alerts)


def test_position_read_failure_isolated_per_coin_then_raises():
    ex, wd, st, alerts = setup()
    orig = ex.swap_contracts

    def flaky(coin: str) -> float:
        if coin == "ETH":
            raise ConnectionError("swap read down")
        return orig(coin)

    ex.swap_contracts = flaky
    ex.trigger_adl("BTC", 1.0)
    with pytest.raises(WatchdogError) as ei:
        wd.check(st)
    assert "ETH" in str(ei.value) and "ConnectionError" in str(ei.value)
    assert "BTC" not in st.legs and "ETH" in st.legs and st.mode == "HALTED"
    assert any("watchdog could not check ETH" in a for a in alerts)


def test_float_noise_is_not_a_shrink():
    ex, wd, st, alerts = setup()
    assert st.legs["BTC"].contracts == pytest.approx(0.2)
    n = len(ex.calls)
    ex.swap["BTC"] = 0.19999999999999998
    assert wd.check(st) == []
    assert ex.calls[n:] == [] and st.mode == "HEDGED" and alerts == []


def test_spot_is_sold_before_perp_is_bought_back():
    ex, wd, st, _ = setup()
    n = len(ex.calls)
    ex.trigger_adl("ETH", 0.3)
    wd.check(st)
    ops = [op for op, coin, _q in ex.calls[n:] if coin == "ETH"]
    assert ops[0] == "sell_spot"
    assert "buy_swap" in ops and ops.index("sell_spot") < ops.index("buy_swap")
