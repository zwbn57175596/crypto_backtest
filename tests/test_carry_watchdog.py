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
    ex.fail_next("sell_spot", 3)          # executor tries 3 times
    actions = wd.check(st)
    assert any("FAILED" in a for a in actions)
    assert "BTC" in st.legs and st.mode == "HALTED"
    assert sum("CRITICAL" in a for a in alerts) >= 2
