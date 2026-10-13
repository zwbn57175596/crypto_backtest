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
    ex.fail_next("buy_swap", 3)
    with pytest.raises(HedgeError):
        exe.open("BTC", 125.0, now_ms=0)
    assert any("CRITICAL" in a for a in alerts)
    assert ex.swap["BTC"] > 0            # naked short is still there; caller must halt


def test_close_flattens_both_legs():
    ex, exe, _ = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    res = exe.close("BTC", leg)
    assert res and res.ok
    assert ex.swap["BTC"] == pytest.approx(0.0)
    assert ex.spot["BTC"] < 1e-5


def test_close_after_adl_only_sells_spot():
    ex, exe, _ = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    ex.trigger_adl("BTC", 1.0)
    assert exe.close("BTC", leg)
    assert ex.spot["BTC"] < 1e-5
    assert ops(ex, "buy_swap") == []


def test_close_failure_returns_false_keeps_spot_and_leaves_perp_untouched():
    ex, exe, alerts = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    ex.fail_next("sell_spot", 3)
    res = exe.close("BTC", leg)
    assert not res
    assert res.spot_closed is False and res.perp_closed is False
    assert ex.spot["BTC"] > 1e-5
    assert ex.swap["BTC"] == pytest.approx(0.20)     # hedge kept intact
    assert ops(ex, "buy_swap") == []
    assert any("CRITICAL" in a for a in alerts)


@pytest.mark.parametrize("bad", [0, -5, float("nan")])
def test_open_rejects_non_positive_or_nan_notional(bad):
    _, exe, _ = make()
    with pytest.raises(ValueError):
        exe.open("BTC", bad, now_ms=0)


def test_partial_spot_fill_is_rolled_back_flat():
    ex, exe, _ = make()
    ex.partial_next("buy_spot", 0.5)
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert ex.spot.get("BTC", 0.0) < 1e-5
    assert ex.swap["BTC"] == pytest.approx(0.0)


def test_partial_rollback_fill_is_finished_by_the_drain_loop():
    ex, exe, _ = make()
    ex.fail_next("buy_spot")
    ex.partial_next("buy_swap", 0.5)
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert ex.swap["BTC"] == pytest.approx(0.0)


def test_spot_liquidation_failure_keeps_perp_and_raises():
    ex, exe, alerts = make()
    ex.partial_next("buy_spot", 0.5)
    ex.fail_next("sell_spot", 3)
    with pytest.raises(HedgeError):
        exe.open("BTC", 125.0, now_ms=0)
    assert any("CRITICAL" in a for a in alerts)
    assert ex.swap["BTC"] == pytest.approx(0.20)     # hedge kept as intact as possible


def test_close_perp_failure_reports_naked_short():
    ex, exe, alerts = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    ex.fail_next("buy_swap", 3)
    res = exe.close("BTC", leg)
    assert not res
    assert res.spot_closed is True and res.perp_closed is False
    assert any("CRITICAL" in a for a in alerts)
    assert ex.swap["BTC"] > 0
    assert ex.spot["BTC"] < 1e-5


def test_close_partial_perp_buyback_is_finished_by_the_loop():
    ex, exe, _ = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    ex.partial_next("buy_swap", 0.5)
    res = exe.close("BTC", leg)
    assert res
    assert ex.swap["BTC"] == pytest.approx(0.0)


def test_exchange_exception_rolls_back_perp():
    ex, exe, alerts = make()

    def boom(coin, qty):
        raise ConnectionError("boom")

    ex.buy_spot = boom
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert ex.swap["BTC"] == pytest.approx(0.0)
    assert any("boom" in a for a in alerts)


def test_pre_existing_holdings_are_untouched():
    ex, exe, _ = make()
    ex.spot["BTC"] = 0.5
    leg = exe.open("BTC", 125.0, now_ms=0)
    assert leg is not None
    assert leg.spot_qty == pytest.approx(0.002, rel=1e-4)
    assert exe.close("BTC", leg)
    assert ex.spot["BTC"] == pytest.approx(0.5, abs=1e-4)
    assert ex.swap["BTC"] == pytest.approx(0.0)


def test_raising_alert_callable_does_not_skip_rollback():
    ex = FakeExchange()

    def bad_alert(msg: str) -> None:
        raise RuntimeError("telegram down")

    exe = Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(), bad_alert)
    ex.fail_next("buy_spot")
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert ex.swap["BTC"] == pytest.approx(0.0)
    assert ex.spot.get("BTC", 0.0) == 0.0


def test_position_read_exception_during_open_raises_hedge_error():
    ex, exe, alerts = make()
    real = ex.swap_contracts
    calls = {"n": 0}

    def flaky(coin):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("read failed")
        return real(coin)

    ex.swap_contracts = flaky
    with pytest.raises(HedgeError):
        exe.open("BTC", 125.0, now_ms=0)
    assert any("CRITICAL" in a for a in alerts)


def test_position_read_exception_during_close_returns_falsy_result():
    ex, exe, alerts = make()
    leg = exe.open("BTC", 125.0, now_ms=0)

    def boom(coin):
        raise ConnectionError("read failed")

    ex.spot_balance = boom
    res = exe.close("BTC", leg)
    assert not res
    assert any("CRITICAL" in a for a in alerts)


def test_settle_delay_after_each_order():
    ex = FakeExchange()
    slept: list[float] = []
    exe = Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(settle_sec=0.5), [].append, sleep=slept.append)
    leg = exe.open("BTC", 125.0, now_ms=0)
    assert leg is not None
    assert slept == [0.5, 0.5]            # after the perp order and after the spot order
    slept.clear()
    assert exe.close("BTC", leg)
    assert slept == [0.5, 0.5]            # one sell_spot attempt + one buy_swap attempt


def test_no_sleep_when_settle_is_zero():
    ex = FakeExchange()
    slept: list[float] = []
    exe = Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(), [].append, sleep=slept.append)
    leg = exe.open("BTC", 125.0, now_ms=0)
    exe.close("BTC", leg)
    assert slept == []


# --- final fix wave: order results, lagging reads, pre-trade errors ---------------------
from carry.executor import PreTradeError  # noqa: E402
from carry.types import OrderResult  # noqa: E402


class Lagging(FakeExchange):
    """Real fills, scripted results, and position reads that lag `lag` reads behind the order."""

    def __init__(self, perp_result=None, spot_result=None, perp_lag=0, spot_lag=0, swallow_perp=False,
                 swallow_spot=False):
        super().__init__()
        self.perp_result, self.spot_result = perp_result, spot_result
        self.perp_lag, self.spot_lag, self.swallow_perp, self.swallow_spot = perp_lag, spot_lag, swallow_perp, swallow_spot
        self._stale_perp = self._stale_spot = None
        self._perp_lag_left = self._spot_lag_left = 0

    def sell_swap(self, coin, contracts):
        self._stale_perp = self.swap.get(coin, 0.0)
        if self.swallow_perp:
            self.calls.append(("sell_swap", coin, contracts))
        else:
            super().sell_swap(coin, contracts)
        self._perp_lag_left = self.perp_lag
        return self.perp_result or OrderResult(True, contracts, 1.0)

    def buy_spot(self, coin, qty):
        self._stale_spot = self.spot.get(coin, 0.0)
        if self.swallow_spot:
            self.calls.append(("buy_spot", coin, qty))
        else:
            super().buy_spot(coin, qty)
        self._spot_lag_left = self.spot_lag
        return self.spot_result or OrderResult(True, qty, 1.0)

    def swap_contracts(self, coin):
        if self._perp_lag_left > 0:
            self._perp_lag_left -= 1
            return self._stale_perp
        return super().swap_contracts(coin)

    def spot_balance(self, coin):
        if self._spot_lag_left > 0:
            self._spot_lag_left -= 1
            return self._stale_spot
        return super().spot_balance(coin)


def lag_make(**kw):
    ex = Lagging(**kw)
    alerts: list[str] = []
    return ex, Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(), alerts.append), alerts


def test_perp_fill_visible_only_on_second_read_still_opens_hedged():
    ex, exe, alerts = lag_make(perp_lag=1)
    leg = exe.open("BTC", 125.0, now_ms=0)
    assert leg is not None and leg.contracts == pytest.approx(0.20)
    assert not any("CRITICAL" in a for a in alerts) and ops(ex, "buy_swap") == []


def test_perp_reported_filled_but_never_visible_alerts_critical_and_returns_none():
    ex, exe, alerts = lag_make(swallow_perp=True)
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert any(a.startswith("[CRITICAL] open BTC: perp order reported") and "no position is visible" in a
               for a in alerts)
    assert ops(ex, "buy_spot") == []


def test_unconfirmed_perp_result_triggers_the_re_read():
    ex, exe, alerts = lag_make(perp_result=OrderResult(False, error="UNCONFIRMED: no terminal state"), perp_lag=1)
    assert exe.open("BTC", 125.0, now_ms=0) is not None


def test_plain_perp_rejection_is_not_re_read_or_critical():
    ex, exe, alerts = lag_make(perp_result=OrderResult(False, error="rejected"), swallow_perp=True)
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert any("perp order not filled" in a for a in alerts) and not any("CRITICAL" in a for a in alerts)


def test_spot_fill_visible_only_on_second_read_still_opens_hedged():
    ex, exe, alerts = lag_make(spot_lag=1)
    assert exe.open("BTC", 125.0, now_ms=0) is not None
    assert not any("CRITICAL" in a for a in alerts)


def test_spot_reported_filled_but_never_visible_alerts_critical_then_rolls_back_the_perp():
    ex, exe, alerts = lag_make(swallow_spot=True)
    assert exe.open("BTC", 125.0, now_ms=0) is None
    assert any(a.startswith("[CRITICAL] open BTC: spot order reported") for a in alerts)
    assert ex.swap["BTC"] == pytest.approx(0.0)


def test_drain_alerts_when_an_order_is_rejected():
    ex, exe, alerts = make()
    leg = exe.open("BTC", 125.0, now_ms=0)
    ex.fail_next("sell_spot", 3)
    assert not exe.close("BTC", leg)
    assert any(a == "order rejected: injected sell_spot failure" for a in alerts)


class ReadsDown(FakeExchange):
    def __init__(self, what):
        super().__init__()
        self.what = what

    def price(self, coin):
        if self.what == "price":
            raise ConnectionError("down")
        return super().price(coin)

    def swap_contracts(self, coin):
        if self.what == "swap":
            raise ConnectionError("down")
        return super().swap_contracts(coin)

    def spot_balance(self, coin):
        if self.what == "spot":
            raise TimeoutError("down")
        return super().spot_balance(coin)


@pytest.mark.parametrize("what,exc", [("price", "ConnectionError"), ("swap", "ConnectionError"), ("spot", "TimeoutError")])
def test_pre_trade_read_failures_raise_pre_trade_error_before_any_order(what, exc):
    ex = ReadsDown(what)
    exe = Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(), [].append)
    with pytest.raises(PreTradeError, match=f"pre-trade read failed \\({exc}\\)"):
        exe.open("BTC", 125.0, now_ms=0)
    assert ex.calls == []


def test_plan_position_value_error_stays_raw():
    ex, exe, _ = make()
    with pytest.raises(ValueError):
        exe.open("BTC", 1.0, now_ms=0)                    # far below the minimum order size
