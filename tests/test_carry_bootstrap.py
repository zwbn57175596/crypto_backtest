import json
import pathlib

import pytest

from carry.bootstrap import exposure_report, load_state_or_halt, reconcile
from carry.config import CarryConfig
from carry.fake_exchange import FakeExchange
from carry.state import CarryState, CoinLeg, load_state, save_state
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
    assert (tmp_path / "s.json.corrupt-1234").exists()
    assert load_state(p).mode == "HALTED"                # the halt is persisted at the original path
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
    assert st.legs["BTC"].spot_qty == pytest.approx(0.002, rel=1e-3) and st.legs["BTC"].entry_ts == 1000 - CFG.min_hold * 8 * 3600 * 1000
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
        raise ConnectionError("secret-detail")

    ex.mgn_ratio = boom
    text = exposure_report(ex, DEFAULT_INSTRUMENTS, CFG, CarryState())
    assert "mode=FLAT" in text and "mgnRatio=unavailable (ConnectionError)" in text
    assert "secret-detail" not in text


def test_exposure_report_survives_a_failing_coin():
    ex = hedged_exchange()

    def boom(coin):
        if coin == "ETH":
            raise TimeoutError("secret-detail")
        return 0.0

    orig = ex.swap_contracts
    ex.swap_contracts = lambda c: boom(c) if c == "ETH" else orig(c)
    text = exposure_report(ex, DEFAULT_INSTRUMENTS, CFG,
                           CarryState(mode="HALTED", halted_reason="because", legs={"BTC": CoinLeg(0.2, 0.002, 1)}))
    assert "mode=HALTED" in text and "reason=because" in text
    assert "ETH: unavailable (TimeoutError)" in text and "secret-detail" not in text
    assert "BTC: spot=0.002000008" in text and "perp=0.2" in text and "mgnRatio=10.0" in text


def test_exposure_report_survives_a_totally_broken_exchange():
    class Broken:
        def __getattr__(self, name):
            raise ConnectionError("down")

    text = exposure_report(Broken(), DEFAULT_INSTRUMENTS, CFG, CarryState())
    assert "mode=FLAT" in text and "BTC: unavailable (ConnectionError)" in text


# --- persistence of the HALTED fallback --------------------------------------
def test_corrupt_state_is_persisted_as_halted(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    load_state_or_halt(p, [].append, clock=lambda: 77.0)
    again = load_state(p)
    assert again.mode == "HALTED" and again.halted_reason and "unreadable" in again.halted_reason
    assert (tmp_path / "s.json.corrupt-77").read_text() == "{not json"
    # a second boot keeps the halt instead of defaulting to FLAT
    assert load_state_or_halt(p, [].append).mode == "HALTED"


def test_rename_failure_leaves_file_untouched_and_still_halts(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    p.write_text("{not json")

    def boom(self, target):
        raise OSError("read-only fs")

    monkeypatch.setattr(pathlib.Path, "rename", boom)
    alerts = []
    st = load_state_or_halt(p, alerts.append, clock=lambda: 5.0)
    assert st.mode == "HALTED" and "unreadable" in st.halted_reason
    assert p.read_text() == "{not json"
    assert not list(tmp_path.glob("*.corrupt-*"))
    assert any("CRITICAL" in a for a in alerts)


def test_raising_alert_does_not_break_load_state_or_halt(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")

    def bad_alert(msg):
        raise ConnectionError("telegram down")

    st = load_state_or_halt(p, bad_alert, clock=lambda: 9.0)
    assert st.mode == "HALTED"
    assert (tmp_path / "s.json.corrupt-9").exists()
    assert load_state(p).mode == "HALTED"          # saved despite the alert failure


@pytest.mark.parametrize("payload", [
    '{"mode": "BOGUS", "legs": {}}',
    '{"mode": "FLAT", "legs": {"BTC": {"contracts": 0.2, "spot_qty": 0.002, "entry_ts": 1}}}',
    '{"mode": "HEDGED", "legs": {"BTC": {"contracts": -0.2, "spot_qty": 0.002, "entry_ts": 1}}}',
    '{"mode": "HEDGED", "legs": {"BTC": {"contracts": 0.2, "spot_qty": NaN, "entry_ts": 1}}}',
    '{"mode": "HEDGED", "legs": {"BTC": {"contracts": 1e999, "spot_qty": 0.002, "entry_ts": 1}}}',
    '{"mode": "HEDGED", "legs": {"BTC": {"contracts": 0.2, "spot_qty": 0, "entry_ts": 1}}}',
    '{"mode": "HEDGED", "legs": {"BTC": {"contracts": 0.2, "spot_qty": 0.002, "entry_ts": -1}}}',
])
def test_semantically_invalid_state_starts_halted(tmp_path, payload):
    p = tmp_path / "s.json"
    p.write_text(payload)
    alerts = []
    st = load_state_or_halt(p, alerts.append, clock=lambda: 3.0)
    assert st.mode == "HALTED" and "unreadable" in st.halted_reason
    assert (tmp_path / "s.json.corrupt-3").exists() and any("CRITICAL" in a for a in alerts)


def test_halted_with_legs_and_hedged_without_legs_are_valid(tmp_path):
    p = tmp_path / "s.json"
    save_state(p, CarryState(mode="HALTED", halted_reason="r", legs={"BTC": CoinLeg(0.2, 0.002, 0)}))
    st = load_state_or_halt(p, [].append)
    assert st.mode == "HALTED" and st.halted_reason == "r" and "BTC" in st.legs
    save_state(p, CarryState(mode="HEDGED"))
    assert load_state_or_halt(p, [].append).mode == "HEDGED"
    assert not list(tmp_path.glob("*.corrupt-*"))


# --- reconcile hardening -----------------------------------------------------
def test_adoption_sends_no_orders_and_alerts_info():
    ex, st, alerts = hedged_exchange(), CarryState(), []
    before = len(ex.calls)
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert len(ex.calls) == before
    assert any(a.startswith("[INFO] adopted existing BTC hedge") for a in alerts)


def test_raising_alert_does_not_break_reconcile():
    def bad_alert(msg):
        raise ConnectionError("telegram down")

    ex = FakeExchange()
    ex.sell_swap("BTC", 0.2)
    st = CarryState()
    notes = reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, bad_alert, 1000)
    assert st.legs == {} and any("PERP_ONLY" in n for n in notes)
    ex2, st2 = hedged_exchange(), CarryState()
    reconcile(ex2, DEFAULT_INSTRUMENTS, CFG, st2, bad_alert, 1000)
    assert "BTC" in st2.legs


def _boom_price(coin):
    raise ConnectionError("price feed down")


def test_price_failure_on_naked_short_still_alerts():
    ex = FakeExchange()
    ex.sell_swap("BTC", 0.2)
    ex.price = _boom_price
    st, alerts = CarryState(), []
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and any("PERP_ONLY" in a and "CRITICAL" in a for a in alerts)


def test_price_failure_on_healthy_hedge_is_not_adopted():
    ex = hedged_exchange()
    ex.price = _boom_price
    st, alerts = CarryState(), []
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and st.mode == "FLAT"
    assert any("CRITICAL" in a and "BTC" in a and "price unavailable" in a for a in alerts)


def test_price_is_not_read_when_it_cannot_matter():
    ex = FakeExchange()
    ex.sell_swap("BTC", 0.2)             # PERP_ONLY
    ex.price = _boom_price
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, CarryState(), [].append, 1000)
    ex2 = hedged_exchange()
    ex2.price = _boom_price
    reconcile(ex2, DEFAULT_INSTRUMENTS, CFG, CarryState(mode="HALTED", halted_reason="x"), [].append, 1000)


def _boundary(price):
    ex = FakeExchange(prices={"BTC": price, "ETH": 3000.0})
    ex.sell_swap("BTC", 1.0)            # 1 contract = 0.01 BTC
    ex.buy_spot("BTC", 0.01 / 0.999)
    return ex


def test_adoption_boundary_is_1_2x_the_order_cap():
    cfg = CarryConfig(max_order_notional=300.0)                 # 1.2 x cap = 360
    st = CarryState()
    reconcile(_boundary(36000.0), DEFAULT_INSTRUMENTS, cfg, st, [].append, 1)       # notional 360.0
    assert "BTC" in st.legs
    st2, alerts = CarryState(), []
    reconcile(_boundary(36001.0), DEFAULT_INSTRUMENTS, cfg, st2, alerts.append, 1)  # slightly above
    assert st2.legs == {} and any("CRITICAL" in a for a in alerts)


def test_spot_only_and_drift_are_alerted_not_adopted():
    ex = FakeExchange()
    ex.buy_spot("BTC", 0.002)
    st, alerts = CarryState(), []
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1)
    assert st.legs == {} and any("SPOT_ONLY" in a and "CRITICAL" in a for a in alerts)
    ex2 = hedged_exchange(spot_qty=0.004)                       # spot twice the perp
    st2, alerts2 = CarryState(), []
    reconcile(ex2, DEFAULT_INSTRUMENTS, CFG, st2, alerts2.append, 1)
    assert st2.legs == {} and any("DRIFT" in a and "CRITICAL" in a for a in alerts2)


def test_coin_already_in_state_is_skipped():
    ex = hedged_exchange()
    st = CarryState(mode="HEDGED", legs={"BTC": CoinLeg(0.5, 0.005, 7)})
    alerts = []
    assert reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000) == []
    assert st.legs["BTC"].entry_ts == 7 and alerts == []
