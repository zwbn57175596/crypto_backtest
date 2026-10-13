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


# --- fix round 1 -----------------------------------------------------------
def n_calls(ex, op, coin):
    return sum(1 for c in ex.calls if c[0] == op and c[1] == coin)


def test_open_retry_storm_is_throttled_then_halts():
    cfg = CarryConfig()
    ex, eng, _ = make()
    ex.fail_next("buy_spot", 200)
    for i in range(10):
        eng.step(i * 2000)
    assert n_calls(ex, "sell_swap", "BTC") == 1 and n_calls(ex, "sell_swap", "ETH") == 1
    assert eng.state.mode != "HALTED"
    t = cfg.open_cooldown_ms
    eng.step(t)                                        # cooldown elapsed: second attempt
    assert n_calls(ex, "sell_swap", "BTC") == 2
    eng.step(2 * t + 1)                                # third consecutive failure
    assert eng.state.mode == "HALTED" and "BTC" in eng.state.halted_reason
    assert eng.state.legs == {}


def test_failed_close_in_signal_branch_halts_and_keeps_leg():
    ex, eng, _ = make()
    eng.step(0)
    ex.funding = dict(ZERO)
    ex.fail_next("sell_spot", 3)
    eng.step(3 * DAY)
    assert eng.state.mode == "HALTED"
    assert "BTC" in eng.state.legs                     # first coin's close failed; step returned


def test_failed_risk_close_is_retried_while_halted():
    ex, eng, _ = make()
    eng.step(0)
    ex.mgn = 2.0
    ex.fail_next("sell_spot", 3)
    eng.step(1000)
    assert eng.state.mode == "HALTED" and set(eng.state.legs) == {"BTC"}   # ETH closed, BTC failed
    eng.step(3000)                                     # still HALTED, tier still CLOSE: retries
    assert eng.state.legs == {} and eng.state.mode == "HALTED"
    assert ex.spot["BTC"] < 1e-5 and ex.swap["BTC"] == pytest.approx(0.0)


def test_raising_alert_callable_does_not_break_step_or_save():
    ex = FakeExchange(funding=dict(HIGH))
    saved = []

    def bad_alert(msg):
        raise RuntimeError("alerter down")

    eng = Engine(ex, DEFAULT_INSTRUMENTS, CarryConfig(), bad_alert,
                 save=lambda s: saved.append(copy.deepcopy(s)))
    eng.step(0)
    ex.trigger_adl("BTC", 1.0)
    eng.step(1000)                                     # must not raise
    assert eng.state.mode == "HALTED" and "BTC" not in eng.state.legs
    assert len(saved) == 2 and saved[-1].mode == "HALTED"


def test_raising_save_is_reported_not_raised():
    def bad_save(state):
        raise OSError("disk full")

    ex, eng, alerts = make(save=bad_save)
    eng.step(0)                                        # must not raise
    assert any("CRITICAL" in a and "state save failed" in a for a in alerts)


def test_watchdog_error_path_halts_and_saves():
    saved = []
    ex, eng, _ = make(save=lambda s: saved.append(copy.deepcopy(s)))
    eng.step(0)
    ex.trigger_adl("BTC", 1.0)
    orig = ex.swap_contracts

    def flaky(coin):
        if coin == "ETH":
            raise ConnectionError("eth down")
        return orig(coin)

    ex.swap_contracts = flaky
    eng.step(1000)
    assert eng.state.mode == "HALTED"
    assert "BTC" not in eng.state.legs and "ETH" in eng.state.legs
    assert set(saved[-1].legs) == {"ETH"} and saved[-1].mode == "HALTED"


def test_rearm_refuses_when_exchange_not_flat():
    ex, eng, _ = make()
    ex.spot["BTC"] = 0.5
    eng.step(0)
    assert eng.state.mode == "HALTED" and eng.state.legs == {}
    with pytest.raises(RuntimeError):
        eng.rearm()
    assert eng.state.mode == "HALTED"


def test_force_close_failure_keeps_leg_and_halts():
    ex, eng, _ = make(funding=dict(ZERO))
    assert eng.force_open("BTC", 100.0, now_ms=0) is True
    ex.fail_next("sell_spot", 3)
    assert eng.force_close("BTC") is False
    assert "BTC" in eng.state.legs and eng.state.mode == "HALTED"


def test_force_open_refused_for_unknown_coin_and_non_normal_tier():
    ex, eng, _ = make(funding=dict(ZERO))
    with pytest.raises(ValueError):
        eng.force_open("DOGE", 100.0, now_ms=0)
    assert eng.force_open("BTC", 100.0, now_ms=0) is True
    ex.mgn = 3.0                                       # WARN tier
    with pytest.raises(RuntimeError):
        eng.force_open("ETH", 100.0, now_ms=1000)
    assert "ETH" not in eng.state.legs


def test_backwards_clock_does_not_raise_or_close_early():
    ex, eng, _ = make()
    eng.step(5 * DAY)
    ex.funding = dict(ZERO)
    eng.step(1 * DAY)                                  # clock went backwards
    assert set(eng.state.legs) == {"BTC", "ETH"} and eng.state.mode == "HEDGED"


def test_manual_ops_are_persisted():
    saved = []
    ex, eng, _ = make(funding=dict(ZERO), save=lambda s: saved.append(copy.deepcopy(s)))
    eng.force_open("BTC", 100.0, now_ms=0)
    assert len(saved) == 1 and "BTC" in saved[-1].legs
    eng.force_close("BTC")
    assert len(saved) == 2 and saved[-1].legs == {}
    eng.rearm()
    assert len(saved) == 3


# --- final fix wave --------------------------------------------------------
from carry.risk import Tier  # noqa: E402
from carry.types import BTC  # noqa: E402


def test_watchdog_error_does_not_skip_risk_close():
    ex, eng, _ = make()
    eng.step(0)
    orig = ex.swap_contracts

    def flaky(coin):
        if coin == "ETH":
            raise ConnectionError("eth down")
        return orig(coin)

    ex.swap_contracts = flaky
    ex.mgn = 2.0                                       # CLOSE tier
    eng.step(1000)
    assert "BTC" not in eng.state.legs and ex.spot["BTC"] < 1e-5 and orig("BTC") == pytest.approx(0.0)
    assert "ETH" in eng.state.legs                     # its close cannot complete
    assert eng.state.mode == "HALTED"


def test_rearm_resets_open_throttle_and_tier():
    cfg = CarryConfig()
    ex, eng, _ = make()
    ex.fail_next("buy_spot", 200)
    t = cfg.open_cooldown_ms
    for now in (0, t, 2 * t + 1):
        eng.step(now)
    assert eng.state.mode == "HALTED" and eng.state.legs == {}
    ex.fail_next("buy_spot", 0)
    eng.rearm()
    eng.step(2 * t + 2)                                # no cooldown wait
    assert "BTC" in eng.state.legs and eng.state.mode == "HEDGED"


def test_rearm_resets_last_tier():
    ex, eng, _ = make(funding=dict(ZERO))
    eng.force_open("BTC", 100.0, now_ms=0)
    ex.mgn = 3.0
    eng.step(1000)
    assert eng._last_tier == Tier.WARN
    eng.force_close("BTC")
    eng.rearm()
    assert eng._last_tier == Tier.NORMAL


def test_engine_rejects_coin_missing_from_instruments():
    with pytest.raises(ValueError):
        Engine(FakeExchange(), {"BTC": BTC}, CarryConfig(), lambda m: None)


# --- final fix wave ------------------------------------------------------------------
from carry.executor import PreTradeError  # noqa: E402


def make_cfg(cfg, funding=HIGH):
    ex = FakeExchange(funding=dict(funding))
    alerts: list[str] = []
    return ex, Engine(ex, DEFAULT_INSTRUMENTS, cfg, alerts.append), alerts


def test_uncalibrated_engine_opens_nothing_but_alerts_once():
    ex, eng, alerts = make_cfg(CarryConfig(mgn_calibrated=False))
    for i in range(3):
        eng.step(i * 1000)
    assert eng.state.legs == {} and sent_orders(ex) == []
    warns = [a for a in alerts if "not calibrated" in a]
    assert len(warns) == 1 and warns[0].startswith("[WARNING]")


def test_uncalibrated_engine_still_runs_watchdog_risk_and_force_open():
    ex, eng, alerts = make_cfg(CarryConfig(mgn_calibrated=False))
    assert eng.force_open("BTC", 100.0, 0) is True and "BTC" in eng.state.legs      # force_open not blocked
    ex.trigger_adl("BTC", 1.0)
    eng.step(1000)                                                                   # watchdog still flattens
    assert "BTC" not in eng.state.legs and ex.spot["BTC"] == pytest.approx(0.0, abs=1e-7)
    ex2, eng2, _ = make_cfg(CarryConfig(mgn_calibrated=False))
    eng2.force_open("BTC", 100.0, 0)
    ex2.mgn = 2.0                                                                    # CLOSE tier still closes
    eng2.step(1000)
    assert eng2.state.legs == {} and eng2.state.mode == "HALTED"


def test_percentage_looking_margin_ratio_alerts_once_but_still_classifies():
    ex, eng, alerts = make_cfg(CarryConfig())
    ex.mgn = 350.0
    for i in range(3):
        eng.step(i * 1000)
    assert len([a for a in alerts if "looks like a percentage" in a]) == 1
    assert set(eng.state.legs) == {"BTC", "ETH"}                                     # NORMAL tier, opened as usual


class PriceDown(FakeExchange):
    down = True

    def price(self, coin):
        if self.down:
            raise ConnectionError("price feed down")
        return super().price(coin)


def test_pre_trade_read_failure_skips_with_short_cooldown_and_never_halts():
    ex = PriceDown(funding=dict(HIGH))
    alerts: list[str] = []
    cfg = CarryConfig(max_open_fails=1)
    eng = Engine(ex, DEFAULT_INSTRUMENTS, cfg, alerts.append)
    eng.step(0)
    assert eng.state.mode == "FLAT" and eng.state.legs == {} and sent_orders(ex) == []
    assert any(a.startswith("[WARNING]") and "pre-trade read failed (ConnectionError)" in a and "skipping" in a
               for a in alerts)
    assert eng._open_fails == {}                                                     # not counted
    n = len(alerts)
    eng.step(30_000)                                                                 # inside 60 s: no retry
    assert len(alerts) == n
    ex.down = False
    eng.step(61_000)                                                                 # after 60 s: retried and opens
    assert set(eng.state.legs) == {"BTC", "ETH"}


def test_force_open_pre_trade_failure_returns_false_without_halting():
    ex = PriceDown(funding=dict(HIGH))
    eng = Engine(ex, DEFAULT_INSTRUMENTS, CarryConfig(), [].append)
    assert eng.force_open("BTC", 100.0, 0) is False
    assert eng.state.mode == "FLAT" and sent_orders(ex) == []


def test_hedge_error_after_first_order_still_halts():
    ex, eng, alerts = make()
    ex.partial_next("sell_swap", 0.5)
    ex.fail_next("buy_swap", 3)                                                      # undo of the partial short fails
    eng.step(0)
    assert eng.state.mode == "HALTED"


def test_halted_engine_alerts_on_unrecorded_holdings_without_acting():
    ex, eng, alerts = make(funding=ZERO)
    eng.state.mode = "HALTED"
    ex.spot["BTC"] = 0.01
    eng.step(0)
    assert eng.state.mode == "HALTED" and sent_orders(ex) == []
    assert any(a.startswith("[CRITICAL] HALTED: unexpected BTC holdings (SPOT_ONLY)") for a in alerts)


def test_halted_engine_survives_failing_holdings_reads():
    ex, eng, alerts = make(funding=ZERO)
    eng.state.mode = "HALTED"

    def boom(coin):
        raise ConnectionError("down")
    ex.spot_balance = boom
    assert eng.step(0) is not None and eng.state.mode == "HALTED"
