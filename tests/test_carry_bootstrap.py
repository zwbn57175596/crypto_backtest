import json

import pytest

from carry.bootstrap import exposure_report, load_state_or_halt, reconcile
from carry.config import CarryConfig
from carry.fake_exchange import FakeExchange
from carry.state import CarryState, CoinLeg, save_state
from carry.types import DEFAULT_INSTRUMENTS

CFG = CarryConfig()


def hedged_exchange(contracts=0.2, spot_qty=0.00200201):
    ex = FakeExchange()
    ex.sell_swap("BTC", contracts)
    ex.buy_spot("BTC", spot_qty)
    return ex


def test_missing_state_file_gives_default(tmp_path):
    st = load_state_or_halt(tmp_path / "none.json", [].append)
    assert st.mode == "FLAT" and st.legs == {}


def test_valid_state_file_is_loaded(tmp_path):
    p = tmp_path / "s.json"
    save_state(p, CarryState(mode="HEDGED", legs={"BTC": CoinLeg(0.2, 0.002, 5)}))
    assert load_state_or_halt(p, [].append).legs["BTC"].contracts == 0.2


def test_corrupt_state_starts_halted_and_keeps_evidence(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    alerts = []
    st = load_state_or_halt(p, alerts.append, clock=lambda: 1234.0)
    assert st.mode == "HALTED" and "unreadable" in st.halted_reason
    assert not p.exists() and (tmp_path / "s.json.corrupt-1234").exists()
    assert any("CRITICAL" in a for a in alerts)


def test_unknown_leg_keys_also_start_halted(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"mode": "HEDGED", "legs": {"BTC": {"contracts": 1, "bogus": 2}}}))
    assert load_state_or_halt(p, [].append).mode == "HALTED"


def test_reconcile_flat_exchange_is_a_noop():
    st = CarryState()
    assert reconcile(FakeExchange(), DEFAULT_INSTRUMENTS, CFG, st, [].append, 1000) == []
    assert st.mode == "FLAT" and st.legs == {}


def test_reconcile_adopts_a_healthy_hedge():
    ex, st = hedged_exchange(), CarryState()
    notes = reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, [].append, 1000)
    assert st.mode == "HEDGED" and st.legs["BTC"].contracts == pytest.approx(0.2)
    assert st.legs["BTC"].spot_qty == pytest.approx(0.002, rel=1e-3) and st.legs["BTC"].entry_ts == 1000
    assert any("adopted" in n and "BTC" in n for n in notes)


def test_reconcile_does_not_adopt_when_halted():
    st = CarryState(mode="HALTED", halted_reason="x")
    alerts = []
    reconcile(hedged_exchange(), DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and st.mode == "HALTED" and any("CRITICAL" in a for a in alerts)


def test_reconcile_never_adopts_a_naked_short():
    ex = FakeExchange()
    ex.sell_swap("BTC", 0.2)
    st, alerts = CarryState(), []
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and any("PERP_ONLY" in a and "CRITICAL" in a for a in alerts)
    assert ex.swap["BTC"] == 0.2                        # and it did not trade


def test_reconcile_does_not_adopt_an_oversized_hedge():
    ex = hedged_exchange(contracts=20.0, spot_qty=0.2002002)      # ~12000 USDT notional
    st, alerts = CarryState(), []
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and any("CRITICAL" in a for a in alerts)


def test_exposure_report_contents():
    ex = hedged_exchange()
    ex.adl["BTC"] = 3
    text = exposure_report(ex, DEFAULT_INSTRUMENTS, CFG, CarryState(mode="HEDGED", legs={"BTC": CoinLeg(0.2, 0.002, 1)}))
    for needle in ("mode=HEDGED", "BTC:", "perp=0.2", "hedge=OK", "adl=3", "ETH:", "mgnRatio=10.0"):
        assert needle in text


def test_exposure_report_never_raises():
    ex = FakeExchange()

    def boom():
        raise ConnectionError("down")

    ex.mgn_ratio = boom
    assert exposure_report(ex, DEFAULT_INSTRUMENTS, CFG, CarryState()).startswith("exposure unavailable: ConnectionError")
