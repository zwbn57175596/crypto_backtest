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

    def place_order(self, **kw):
        self.placed.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        if self.top_code:
            return {"code": self.top_code, "msg": "bad", "data": []}
        if self.reject:
            return ok([{"sCode": "51008", "sMsg": "insufficient balance", "clOrdId": kw["clOrdId"]}])
        return ok([{"sCode": "0", "ordId": "1", "clOrdId": kw["clOrdId"]}])

    def get_order(self, instId, ordId="", clOrdId=""):
        self.get_calls += 1
        row = self.states[0] if len(self.states) == 1 else self.states.pop(0)
        return ok([dict(row, instId=instId, clOrdId=clOrdId)])

    def cancel_order(self, instId, ordId="", clOrdId=""):
        self.cancelled.append(clOrdId)
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
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert len(tr.cancelled) == 1 and not r.ok and "live" in r.error


def test_ambiguous_placement_is_resolved_by_client_order_id():
    tr = FakeTrade()
    tr.errors = [ConnectionError("t")] * 3               # all 3 placement attempts time out
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert len({p["clOrdId"] for p in tr.placed}) == 1 and len(tr.placed) == 3
    assert r.ok and r.filled == pytest.approx(0.2)


def test_ambiguous_placement_with_unknown_order_fails_cleanly():
    tr = FakeTrade([{"state": "live", "accFillSz": "0"}])
    tr.errors = [ConnectionError("t")] * 3
    tr.get_order = lambda **kw: {"code": "51603", "msg": "order not found", "data": []}
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.filled == 0.0 and "ConnectionError" in r.error


def test_row_rejection_does_not_poll():
    tr = FakeTrade()
    tr.reject = True
    r = make(tr)[0].buy_spot("BTC", 0.002)
    assert not r.ok and "51008" in r.error and tr.get_calls == 0


def test_non_transient_top_level_code_does_not_poll_or_retry():
    tr = FakeTrade()
    tr.top_code = "51000"
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert not r.ok and tr.get_calls == 0 and len(tr.placed) == 1


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
