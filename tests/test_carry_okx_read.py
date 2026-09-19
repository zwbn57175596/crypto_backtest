import pytest

from carry.okx_exchange import ExchangeError, OkxExchange

H8 = 8 * 3600 * 1000
NOW_MS = 1_800_000_000_000


class Clock:
    def __init__(self, t=NOW_MS / 1000):
        self.t, self.slept = t, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def ok(data):
    return {"code": "0", "msg": "", "data": data}


class FakeMarket:
    def __init__(self, last=None, ts=NOW_MS):
        self.last = last or {"BTC-USDT": "60000", "BTC-USDT-SWAP": "60050", "ETH-USDT": "3000", "ETH-USDT-SWAP": "3001"}
        self.ts, self.calls, self.errors = ts, [], []

    def get_ticker(self, instId):
        self.calls.append(instId)
        if self.errors:
            raise self.errors.pop(0)
        return ok([{"instId": instId, "last": self.last[instId], "ts": str(self.ts)}])


class FakePublic:
    def __init__(self, latest=NOW_MS - 3_600_000, n=12):
        self.latest, self.n, self.calls = latest, n, []

    def funding_rate_history(self, instId, limit="", after="", before=""):
        self.calls.append((instId, limit))
        rows = [{"fundingTime": str(self.latest - i * H8), "fundingRate": str((self.n - i) * 1e-5)} for i in range(self.n)]
        return ok(rows)                                            # newest first, like OKX


class FakeAccount:
    def __init__(self):
        self.details = [{"ccy": "BTC", "cashBal": "0.002"}, {"ccy": "USDT", "cashBal": "1000"}]
        self.positions = []
        self.mgn = ""
        self.bal_calls, self.pos_calls = 0, 0

    def get_account_balance(self, ccy=""):
        self.bal_calls += 1
        det = [d for d in self.details if not ccy or d["ccy"] == ccy]
        return ok([{"details": det, "mgnRatio": self.mgn}])

    def get_positions(self, instType="", instId="", posId=""):
        self.pos_calls += 1
        return ok([p for p in self.positions if not instId or p["instId"] == instId])


def make(clock=None, **kw):
    clock = clock or Clock()
    parts = dict(trade=None, account=FakeAccount(), public=FakePublic(), market=FakeMarket())
    parts.update({k: kw.pop(k) for k in list(kw) if k in parts})
    ex = OkxExchange(parts["trade"], parts["account"], parts["public"], parts["market"],
                     clock=clock.now, sleep=clock.sleep, **kw)
    return ex, parts, clock


def test_price_returns_spot_last():
    ex, _, _ = make()
    assert ex.price("BTC") == 60000.0


def test_price_rejects_stale_tick():
    ex, _, _ = make(market=FakeMarket(ts=NOW_MS - 120_000))
    with pytest.raises(ExchangeError):
        ex.price("BTC")


def test_price_rejects_spot_swap_divergence():
    ex, _, _ = make(market=FakeMarket({"BTC-USDT": "60000", "BTC-USDT-SWAP": "63000", "ETH-USDT": "1", "ETH-USDT-SWAP": "1"}))
    with pytest.raises(ExchangeError):
        ex.price("BTC")


def test_price_rejects_non_positive_or_nan():
    for bad in ("0", "-5", "nan"):
        ex, _, _ = make(market=FakeMarket({"BTC-USDT": bad, "BTC-USDT-SWAP": "60000", "ETH-USDT": "1", "ETH-USDT-SWAP": "1"}))
        with pytest.raises(ExchangeError):
            ex.price("BTC")


def test_funding_history_oldest_first_and_sliced():
    ex, parts, _ = make()
    rates = ex.funding_history("BTC", 3)
    assert rates == pytest.approx([10e-5, 11e-5, 12e-5])
    assert parts["public"].calls == [("BTC-USDT-SWAP", "3")]


def test_funding_history_is_cached_until_next_settlement():
    ex, parts, clock = make()
    ex.funding_history("BTC", 9)
    clock.t += 600
    ex.funding_history("BTC", 9)
    assert len(parts["public"].calls) == 1
    clock.t += 8 * 3600                                            # past latest + 8h + 60s
    ex.funding_history("BTC", 9)
    assert len(parts["public"].calls) == 2


def test_funding_history_refetches_for_larger_n_and_validates():
    ex, parts, _ = make()
    ex.funding_history("BTC", 3)
    ex.funding_history("BTC", 9)
    assert len(parts["public"].calls) == 2
    assert ex.funding_history("BTC", 0) == []
    with pytest.raises(ExchangeError):
        ex.funding_history("BTC", 101)


def test_funding_history_empty_raises():
    ex, _, _ = make(public=FakePublic(n=0))
    with pytest.raises(ExchangeError):
        ex.funding_history("BTC", 3)


def test_spot_balance():
    ex, parts, _ = make()
    assert ex.spot_balance("BTC") == pytest.approx(0.002)
    assert ex.spot_balance("ETH") == 0.0                           # not in details
    parts["account"].details = [{"ccy": "BTC", "cashBal": ""}]
    ex2, _, _ = make(account=parts["account"])
    assert ex2.spot_balance("BTC") == 0.0


def pos(inst, p, side="net", adl="3"):
    return {"instId": inst, "pos": p, "posSide": side, "adl": adl}


def test_swap_contracts_net_mode_short_is_unsigned():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2")]
    ex, _, _ = make(account=acc)
    assert ex.swap_contracts("BTC") == pytest.approx(0.2)
    assert ex.swap_contracts("ETH") == 0.0


def test_swap_contracts_long_position_raises():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "0.3")]
    ex, _, _ = make(account=acc)
    with pytest.raises(ExchangeError):
        ex.swap_contracts("BTC")


def test_swap_contracts_long_short_mode():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "0.2", side="short"), pos("BTC-USDT-SWAP", "0", side="long")]
    ex, _, _ = make(account=acc)
    assert ex.swap_contracts("BTC") == pytest.approx(0.2)
    acc.positions = [pos("BTC-USDT-SWAP", "0.5", side="long")]
    ex2, _, _ = make(account=acc)
    with pytest.raises(ExchangeError):
        ex2.swap_contracts("BTC")


def test_adl_rank():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2", adl="4")]
    ex, _, _ = make(account=acc)
    assert ex.swap_adl_rank("BTC") == 4
    assert ex.swap_adl_rank("ETH") == 0
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2", adl="")]
    ex2, _, _ = make(account=acc)
    assert ex2.swap_adl_rank("BTC") == 0


def test_mgn_ratio_parsing():
    for raw, want in (("", None), ("3.25", 3.25), ("nan", None), ("inf", None), ("abc", None)):
        acc = FakeAccount()
        acc.mgn = raw
        ex, _, _ = make(account=acc)
        assert ex.mgn_ratio() == want


def test_reads_are_cached_briefly_and_position_call_is_shared():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2")]
    ex, _, clock = make(account=acc)
    ex.spot_balance("BTC")
    ex.spot_balance("BTC")
    assert acc.bal_calls == 1
    ex.swap_contracts("BTC")
    ex.swap_adl_rank("BTC")
    assert acc.pos_calls == 1
    clock.t += 2.0
    ex.spot_balance("BTC")
    assert acc.bal_calls == 2


def test_transient_errors_are_retried_with_backoff():
    market = FakeMarket()
    market.errors = [ConnectionError("t"), ConnectionError("t")]
    ex, _, clock = make(market=market)
    assert ex.price("BTC") == 60000.0
    assert clock.slept[:2] == [0.5, 1.0]


def test_transient_error_code_exhausts_retries_and_others_fail_fast():
    class Bad:
        def __init__(self, code):
            self.code, self.n = code, 0

        def get_ticker(self, instId):
            self.n += 1
            return {"code": self.code, "msg": "x", "data": []}

    for code, calls in (("50011", 3), ("51000", 1)):
        bad = Bad(code)
        ex, _, _ = make(market=bad)
        with pytest.raises(ExchangeError) as e:
            ex.price("BTC")
        assert e.value.code == code and bad.n == calls
