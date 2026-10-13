import inspect
import re

import pytest

from carry.okx_exchange import OkxExchange, _fmt

NOW_MS = 1_800_000_000_000


class Clock:
    def __init__(self):
        self.t, self.slept = NOW_MS / 1000, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def ok(data):
    return {"code": "0", "msg": "", "data": data}


FILLED = {"state": "filled", "accFillSz": "0.2", "avgPx": "60000"}


class FakeTrade:
    def __init__(self, states=(FILLED,)):
        self.states = list(states)
        self.placed, self.cancelled, self.get_calls = [], [], 0
        self.errors, self.reject, self.top_code = [], False, None
        self.responses = []                # scripted place_order responses, consumed after `errors`
        self.get_mode = "normal"           # "normal" | "empty" | "raise"
        self.visible_after = 1             # get_order finds the order once this many POSTs were made
        self.cancel_flip_at = None         # after this many cancels the order becomes `canceled`
        self.cancel_flip_row = {"state": "canceled", "accFillSz": "0", "avgPx": ""}
        self.hidden = set()                # clOrdIds whose get_order answers empty data
        self.notfound = set()              # clOrdIds whose get_order answers code 51603
        self.order_list = []               # rows returned by get_order_list (filtered by instId)

    def place_order(self, **kw):
        self.placed.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        if self.responses:
            return self.responses.pop(0)
        if self.top_code:
            return {"code": self.top_code, "msg": "bad", "data": []}
        if self.reject:                    # real OKX shape: envelope code "1" + row-level sCode
            return {"code": "1", "msg": "", "data": [{"sCode": "51008", "sMsg": "insufficient balance",
                                                     "clOrdId": kw["clOrdId"]}]}
        return ok([{"sCode": "0", "ordId": "1", "clOrdId": kw["clOrdId"]}])

    def get_order(self, instId, ordId="", clOrdId=""):
        self.get_calls += 1
        if self.get_mode == "raise":
            raise ConnectionError("get_order down")
        if self.get_mode == "empty" or clOrdId in self.hidden:
            return ok([])
        if clOrdId in self.notfound:
            return {"code": "51603", "msg": "order does not exist", "data": []}
        if len(self.placed) < self.visible_after:
            return {"code": "51603", "msg": "order does not exist", "data": []}
        row = self.states[0] if len(self.states) == 1 else self.states.pop(0)
        return ok([dict(row, instId=instId, clOrdId=clOrdId)])

    def get_order_list(self, instType="", uly="", instId="", ordType="", state="", after="", before="",
                       limit="", instFamily=""):
        return ok([dict(r) for r in self.order_list if not instId or r["instId"] == instId])

    def cancel_order(self, instId, ordId="", clOrdId=""):
        self.cancelled.append(clOrdId)
        if self.cancel_flip_at is not None and len(self.cancelled) >= self.cancel_flip_at:
            self.states = [self.cancel_flip_row]
        return ok([{"sCode": "0"}])


class FakeAccount:
    def __init__(self):
        self.bal_calls = 0

    def get_account_balance(self, ccy=""):
        self.bal_calls += 1
        return ok([{"details": [{"ccy": "BTC", "cashBal": "0.002"}], "mgnRatio": ""}])


def make(trade, **kw):
    clock = Clock()
    return OkxExchange(trade, FakeAccount(), None, None, clock=clock.now, sleep=clock.sleep, **kw), clock


def test_fmt():
    assert _fmt(0.2) == "0.2" and _fmt(1e-8) == "0.00000001" and _fmt(100.0) == "100" and _fmt(1.5) == "1.5"


def test_sell_swap_open_short_params_and_fill():
    tr = FakeTrade()
    ex, _ = make(tr)
    r = ex.sell_swap("BTC", 0.2)
    kw = tr.placed[0]
    assert (kw["instId"], kw["tdMode"], kw["side"], kw["ordType"], kw["sz"]) == \
        ("BTC-USDT-SWAP", "cross", "sell", "market", "0.2")
    assert kw["clOrdId"].startswith("carry")
    assert "reduceOnly" not in kw and "posSide" not in kw and "tgtCcy" not in kw
    assert r.ok and r.filled == pytest.approx(0.2) and r.avg_price == pytest.approx(60000.0)


def test_buy_swap_is_reduce_only():
    tr = FakeTrade()
    make(tr)[0].buy_swap("BTC", 0.2)
    kw = tr.placed[0]
    assert kw["side"] == "buy" and kw["reduceOnly"] == "true" and kw["tdMode"] == "cross"


def test_spot_td_mode_defaults_to_cross():
    # "cash" tdMode is rejected by OKX once the account is in multi-currency
    # margin mode (acctLv=3, required by preflight): sCode 51000 "Parameter
    # tdMode error", confirmed against the real API on 2026-10-01. "cross"
    # is the only tdMode that works for spot orders in that account mode;
    # it does not borrow since preflight hard-blocks autoLoan=true.
    tr = FakeTrade()
    ex, _ = make(tr)
    ex.buy_spot("BTC", 0.002)
    assert tr.placed[0]["tdMode"] == "cross"


def test_spot_orders_use_base_currency_target_and_configured_td_mode():
    tr = FakeTrade()
    ex, _ = make(tr, spot_td_mode="cash")
    ex.buy_spot("BTC", 0.00200201)
    ex.sell_spot("BTC", 0.002)
    buy, sell = tr.placed
    assert (buy["instId"], buy["side"], buy["tgtCcy"], buy["tdMode"], buy["sz"]) == \
        ("BTC-USDT", "buy", "base_ccy", "cash", "0.00200201")
    assert (sell["side"], sell["tgtCcy"], sell["sz"]) == ("sell", "base_ccy", "0.002")
    assert "reduceOnly" not in buy


def test_partial_then_canceled_reports_partial_fill():
    tr = FakeTrade([{"state": "partially_filled", "accFillSz": "0.1", "avgPx": "60000"},
                    {"state": "canceled", "accFillSz": "0.1", "avgPx": "60000"}])
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.filled == pytest.approx(0.1) and "canceled" in r.error


def test_live_order_at_timeout_is_cancelled():
    tr = FakeTrade([{"state": "live", "accFillSz": "0", "avgPx": ""}])
    tr.cancel_flip_at = 1
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert len(tr.cancelled) == 1 and not r.ok and "canceled" in r.error


def test_ambiguous_placement_is_resolved_by_client_order_id():
    tr = FakeTrade()
    tr.errors = [ConnectionError("t")] * 3               # all 3 placement attempts time out
    tr.visible_after = 3                                 # ...and only the 3rd POST really reached OKX
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert len({p["clOrdId"] for p in tr.placed}) == 1 and len(tr.placed) == 3
    assert r.ok and r.filled == pytest.approx(0.2)


def test_ambiguous_placement_with_unknown_order_fails_cleanly():
    tr = FakeTrade([{"state": "live", "accFillSz": "0"}])
    tr.errors = [ConnectionError("t")] * 3
    tr.get_order = lambda **kw: {"code": "51603", "msg": "order not found", "data": []}
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.filled == 0.0 and "ConnectionError" in r.error
    assert len({p["clOrdId"] for p in tr.placed}) == 1 and len(tr.placed) == 3


def test_row_rejection_does_not_poll():
    tr = FakeTrade()
    tr.reject = True
    r = make(tr)[0].buy_spot("BTC", 0.002)
    assert not r.ok and "51008" in r.error and tr.get_calls == 0


def test_top_level_code_without_row_is_ambiguous_and_probed_never_a_clean_reject():
    tr = FakeTrade()
    tr.top_code = "51000"
    tr.visible_after = 99                                # the order provably does not exist
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.filled == 0.0 and "51000" in r.error
    assert tr.get_calls >= 1 and len({p["clOrdId"] for p in tr.placed}) == 1   # probed, same id reused


def test_client_order_ids_are_unique_and_valid():
    tr = FakeTrade()
    ex, _ = make(tr)
    ex.sell_swap("BTC", 0.2)
    ex.sell_swap("BTC", 0.2)
    a, b = (p["clOrdId"] for p in tr.placed)
    assert a != b and all(re.fullmatch(r"[A-Za-z0-9]{1,32}", x) for x in (a, b))


def test_write_clears_the_read_cache():
    tr = FakeTrade()
    ex, _ = make(tr)
    ex.spot_balance("BTC")
    ex.spot_balance("BTC")
    assert ex.account.bal_calls == 1
    ex.sell_swap("BTC", 0.2)
    ex.spot_balance("BTC")
    assert ex.account.bal_calls == 2


def test_real_shaped_rejection_code1_with_row_scode_does_not_poll_or_repost():
    tr = FakeTrade()
    tr.responses = [{"code": "1", "msg": "", "data": [{"sCode": "51008", "sMsg": "insufficient balance"}]}]
    r = make(tr)[0].buy_spot("BTC", 0.002)
    assert not r.ok and "51008" in r.error and tr.get_calls == 0 and len(tr.placed) == 1


def test_lost_post_then_duplicate_clordid_resolves_to_the_live_order():
    tr = FakeTrade()
    tr.errors = [ConnectionError("t")]
    tr.visible_after = 2                                 # probe after POST 1 finds nothing
    tr.responses = [{"code": "1", "msg": "", "data": [{"sCode": "51016", "sMsg": "duplicate clOrdId"}]}]
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert len(tr.placed) == 2 and len({p["clOrdId"] for p in tr.placed}) == 1
    assert r.ok and r.filled == pytest.approx(0.2)


def test_probe_before_repost_finds_the_order_so_posts_exactly_once():
    tr = FakeTrade()
    tr.errors = [ConnectionError("t")]                   # POST lost after reaching OKX
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert len(tr.placed) == 1 and r.ok and r.filled == pytest.approx(0.2)


def test_unknown_probe_does_not_repost_and_ends_unconfirmed():
    tr = FakeTrade()
    tr.errors = [ConnectionError("t")]
    tr.get_mode = "raise"
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert len(tr.placed) == 1 and not r.ok and r.error.startswith("UNCONFIRMED")


def test_never_visible_order_is_cancelled_alerted_and_blocks_until_resolved():
    tr, alerts = FakeTrade(), []
    tr.get_mode = "empty"
    ex, _ = make(tr, order_timeout_sec=2.0, alert=alerts.append)
    r = ex.sell_swap("BTC", 0.2)
    cl = tr.placed[0]["clOrdId"]
    assert not r.ok and r.error.startswith("UNCONFIRMED") and r.filled == 0.0
    assert tr.cancelled and set(tr.cancelled) == {cl}      # up to 3 cancel rounds, all by clOrdId
    assert any("[CRITICAL]" in a and cl in a for a in alerts)
    r2 = ex.sell_swap("BTC", 0.2)                          # blocked: no POST at all
    assert not r2.ok and r2.error.startswith("BLOCKED") and len(tr.placed) == 1
    tr.get_mode = "normal"                                 # the old order turns out terminal
    r3 = ex.sell_swap("BTC", 0.2)
    assert r3.ok and len(tr.placed) == 2


def test_clear_unconfirmed_is_an_operator_override():
    tr = FakeTrade()
    tr.get_mode = "empty"
    ex, _ = make(tr, order_timeout_sec=2.0)
    ex.sell_swap("BTC", 0.2)
    assert ex.sell_swap("BTC", 0.2).error.startswith("BLOCKED") and len(tr.placed) == 1
    ex.clear_unconfirmed()
    ex.sell_swap("BTC", 0.2)
    assert len(tr.placed) == 2                             # it placed again (and, still blind, went unconfirmed)


def test_blocked_alert_is_sent_once_per_distinct_set():
    tr, alerts = FakeTrade(), []
    tr.get_mode = "empty"
    ex, _ = make(tr, order_timeout_sec=2.0, alert=alerts.append)
    ex.sell_swap("BTC", 0.2)
    n = len(alerts)
    ex.sell_swap("BTC", 0.2)
    ex.sell_swap("BTC", 0.2)
    assert len(alerts) == n + 1


def test_a_raising_alert_sink_never_breaks_the_order_path():
    tr = FakeTrade()
    tr.get_mode = "empty"

    def boom(_msg):
        raise RuntimeError("sink down")

    r = make(tr, order_timeout_sec=2.0, alert=boom)[0].sell_swap("BTC", 0.2)
    assert r.error.startswith("UNCONFIRMED")


def test_live_at_timeout_cancel_succeeds_on_second_round_is_a_normal_partial():
    tr, alerts = FakeTrade([{"state": "live", "accFillSz": "0.05", "avgPx": "60000"}]), []
    tr.cancel_flip_at = 2
    tr.cancel_flip_row = {"state": "canceled", "accFillSz": "0.05", "avgPx": "60000"}
    ex, _ = make(tr, order_timeout_sec=2.0, alert=alerts.append)
    r = ex.sell_swap("BTC", 0.2)
    assert len(tr.cancelled) == 2 and not r.ok and r.filled == pytest.approx(0.05)
    assert r.error == "state=canceled" and alerts == []
    assert ex.sell_swap("BTC", 0.2).error != "" and len(tr.placed) == 2      # not blocked


def test_call_budget_is_bounded_for_a_never_terminal_order():
    tr = FakeTrade([{"state": "live", "accFillSz": "0", "avgPx": ""}])
    r = make(tr)[0].sell_swap("BTC", 0.2)                  # timeout 10s, poll 0.5s
    assert r.error.startswith("UNCONFIRMED")
    assert tr.get_calls <= 10 / 0.5 + 2 + 3 * 2 and len(tr.cancelled) <= 3


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_sizes_make_no_api_call(bad):
    tr = FakeTrade()
    ex, _ = make(tr)
    for call in (ex.sell_swap, ex.buy_swap, ex.buy_spot, ex.sell_spot):
        r = call("BTC", bad)
        assert not r.ok and r.error == "invalid size"
    assert tr.placed == [] and tr.get_calls == 0 and tr.cancelled == []


def test_place_order_kwargs_conform_to_python_okx_signature():
    trade_mod = pytest.importorskip("okx.Trade")
    tr = FakeTrade()
    ex, _ = make(tr)
    ex.sell_swap("BTC", 0.2)
    ex.buy_swap("BTC", 0.2)
    ex.buy_spot("BTC", 0.002)
    ex.sell_spot("BTC", 0.002)
    assert len(tr.placed) == 4
    sig = inspect.signature(trade_mod.TradeAPI.place_order)
    for kw in tr.placed:
        sig.bind(None, **kw)


def _blocked_btc_swap(tr, ex, clock, cl="carryHIDDEN"):
    """Register an unconfirmed BTC swap order whose get_order stays empty (unresolvable)."""
    tr.hidden.add(cl)
    ex._unconfirmed[cl] = ("BTC-USDT-SWAP", clock.now())
    return cl


def test_never_existed_order_stops_blocking_after_the_notfound_grace():
    tr, alerts = FakeTrade(), []
    tr.get_mode = "empty"
    ex, clock = make(tr, order_timeout_sec=2.0, alert=alerts.append)
    ex.sell_swap("BTC", 0.2)                               # registers an unconfirmed entry
    cl = tr.placed[0]["clOrdId"]
    tr.get_mode = "normal"
    tr.visible_after = 2                                   # get_order: code 51603 until a 2nd POST exists
    r = ex.sell_swap("BTC", 0.2)
    assert r.error.startswith("BLOCKED") and len(tr.placed) == 1     # NOT_FOUND but younger than grace
    clock.t += 61
    r = ex.sell_swap("BTC", 0.2)
    assert len(tr.placed) == 2 and r.ok
    assert any(a.startswith("[INFO]") and cl in a and "resolved" in a for a in alerts)


def test_empty_data_never_counts_as_notfound_even_after_the_grace():
    tr = FakeTrade()
    tr.get_mode = "empty"
    ex, clock = make(tr, order_timeout_sec=2.0)
    ex.sell_swap("BTC", 0.2)
    clock.t += 3600
    assert ex.sell_swap("BTC", 0.2).error.startswith("BLOCKED") and len(tr.placed) == 1


def test_blocking_is_per_instrument_and_reduce_only_is_never_blocked():
    tr = FakeTrade()
    ex, clock = make(tr)
    _blocked_btc_swap(tr, ex, clock)
    r = ex.sell_swap("BTC", 0.2)
    assert r.error.startswith("BLOCKED") and "BTC-USDT-SWAP" in r.error and tr.placed == []
    assert ex.sell_swap("ETH", 0.2).ok
    assert ex.buy_spot("BTC", 0.002).ok
    assert ex.buy_swap("BTC", 0.2).ok                      # reduce-only: allowed while BTC swap is blocked
    assert [p["instId"] for p in tr.placed] == ["ETH-USDT-SWAP", "BTC-USDT", "BTC-USDT-SWAP"]
    assert ex.sell_swap("BTC", 0.2).error.startswith("BLOCKED")          # still blocked afterwards


def test_reads_are_never_blocked():
    tr = FakeTrade()
    ex, clock = make(tr)
    _blocked_btc_swap(tr, ex, clock)
    assert ex.sell_swap("BTC", 0.2).error.startswith("BLOCKED")
    assert ex.spot_balance("BTC") == pytest.approx(0.002) and ex.mgn_ratio() is None


def test_adopt_pending_orders_registers_only_live_carry_orders_and_blocks():
    tr, alerts = FakeTrade([{"state": "live", "accFillSz": "0", "avgPx": ""}]), []
    tr.visible_after = 0
    tr.order_list = [{"instId": "BTC-USDT-SWAP", "clOrdId": "carryABC", "state": "live"},
                     {"instId": "BTC-USDT-SWAP", "clOrdId": "manual1", "state": "live"},
                     {"instId": "BTC-USDT-SWAP", "clOrdId": "carryDEF", "state": "filled"}]
    ex, _ = make(tr, alert=alerts.append)
    assert ex.adopt_pending_orders() == ["carryABC"]
    assert any("[CRITICAL]" in a and "carryABC" in a for a in alerts)
    assert ex.sell_swap("BTC", 0.2).error.startswith("BLOCKED") and tr.placed == []
    tr.states = [FILLED]                                   # the adopted order is now terminal
    assert ex.sell_swap("BTC", 0.2).ok and len(tr.placed) == 1


def test_pending_carry_orders_does_not_register():
    tr = FakeTrade()
    tr.order_list = [{"instId": "ETH-USDT", "clOrdId": "carryXYZ", "state": "partially_filled"}]
    ex, _ = make(tr)
    assert ex.pending_carry_orders() == ["carryXYZ"]
    assert ex._unconfirmed == {}
    assert ex.sell_swap("ETH", 0.2).ok                     # not blocked, nothing adopted


def test_unconfirmed_result_reports_last_known_fill_and_price():
    tr = FakeTrade([{"state": "partially_filled", "accFillSz": "0.05", "avgPx": "60000"}])
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert r.error.startswith("UNCONFIRMED")
    assert r.filled == pytest.approx(0.05) and r.avg_price == pytest.approx(60000.0)


def test_ambiguous_placement_error_is_concatenated_with_the_terminal_state():
    tr = FakeTrade([{"state": "canceled", "accFillSz": "0", "avgPx": ""}])
    tr.errors = [ConnectionError("t")]
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.error.startswith("place_order: ConnectionError") and r.error.endswith("; state=canceled")


def test_single_attempt_with_absent_order_returns_immediately():
    tr = FakeTrade()
    tr.errors = [ConnectionError("t")]
    tr.visible_after = 99                                  # probe proves absence
    ex, _ = make(tr, retries=1)
    r = ex.sell_swap("BTC", 0.2)
    assert not r.ok and "ConnectionError" in r.error
    assert len(tr.placed) == 1 and tr.get_calls == 1 and tr.cancelled == [] and ex._unconfirmed == {}


def test_malformed_terminal_row_does_not_raise():
    tr = FakeTrade([{"state": "canceled", "accFillSz": "abc", "avgPx": "nan"}])
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.filled == 0.0 and r.error == "state=canceled"
    tr2 = FakeTrade([{"state": "filled", "accFillSz": "x", "avgPx": None}])
    r2 = make(tr2)[0].sell_swap("BTC", 0.2)
    assert r2.filled == 0.0 and r2.avg_price == 0.0


def test_cancel_and_confirm_does_not_sleep_after_the_last_round():
    tr = FakeTrade([{"state": "live", "accFillSz": "0", "avgPx": ""}])
    ex, clock = make(tr, order_timeout_sec=2.0)
    ex.sell_swap("BTC", 0.2)
    polls = int(2.0 / 0.5)                                 # 4 poll sleeps in the wait loop
    assert clock.slept.count(0.5) == polls + 2             # + 2 (not 3) between the 3 cancel rounds

