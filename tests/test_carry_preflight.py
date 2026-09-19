import pytest

from carry.config import CarryConfig
from carry.preflight import preflight
from carry.types import DEFAULT_INSTRUMENTS

CFG = CarryConfig()


def ok(data):
    return {"code": "0", "msg": "", "data": data}


class FakeAccount:
    def __init__(self, acct_lv="3", auto_loan=False, pos_mode="net_mode", details=None, positions=None):
        self.acct_lv, self.auto_loan, self.pos_mode = acct_lv, auto_loan, pos_mode
        self.details = details if details is not None else [{"ccy": "USDT", "cashBal": "1000", "availBal": "1000"}]
        self.positions, self.fail = positions or [], False
        self.leverage, self.leverage_fail = "5", False

    def get_account_config(self):
        if self.fail:
            raise ConnectionError("down")
        return ok([{"acctLv": self.acct_lv, "autoLoan": self.auto_loan, "posMode": self.pos_mode}])

    def get_account_balance(self, ccy=""):
        return ok([{"details": self.details, "mgnRatio": ""}])

    def get_leverage(self, mgnMode, ccy="", instId=""):
        if self.leverage_fail:
            raise ConnectionError("down")
        return ok([{"instId": instId, "mgnMode": mgnMode, "lever": self.leverage}])

    def get_positions(self, instType="", instId="", posId=""):
        return ok(self.positions)


class FakePublic:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def get_instruments(self, instType, instId=""):
        coin = instId.split("-")[0]
        i = DEFAULT_INSTRUMENTS[coin]
        if instType == "SWAP":
            row = {"ctVal": str(i.ct_val), "lotSz": str(i.swap_lot), "minSz": str(i.swap_min)}
        else:
            row = {"lotSz": str(i.spot_lot), "minSz": str(i.spot_min)}
        row.update(self.overrides.get(instId, {}))
        return ok([row])


def run(**kw):
    public = kw.pop("public", FakePublic())
    return preflight(FakeAccount(**kw), public, CFG, DEFAULT_INSTRUMENTS)


def test_healthy_account_passes():
    r = run()
    assert r.ok and r.problems == []


def test_account_level_autoloan_and_position_mode_are_checked():
    assert any("acctLv" in p for p in run(acct_lv="2").problems)
    assert any("autoLoan" in p for p in run(auto_loan=True).problems)
    assert any("posMode" in p for p in run(pos_mode="long_short_mode").problems)


def test_stray_coin_balance_is_a_problem_but_dust_below_min_is_not():
    usdt = {"ccy": "USDT", "cashBal": "1000", "availBal": "1000"}
    assert any("BTC" in p for p in run(details=[usdt, {"ccy": "BTC", "cashBal": "0.5"}]).problems)
    assert run(details=[usdt, {"ccy": "BTC", "cashBal": "0.000001"}]).ok


def test_other_coins_are_only_warnings():
    r = run(details=[{"ccy": "USDT", "cashBal": "1000", "availBal": "1000"}, {"ccy": "OKB", "cashBal": "5"}])
    assert r.ok and any("OKB" in w for w in r.warnings)


def test_not_enough_usdt_is_a_problem():
    assert any("USDT" in p for p in run(details=[{"ccy": "USDT", "cashBal": "100", "availBal": "100"}]).problems)


def test_open_swap_position_is_a_problem():
    assert any("position" in p for p in run(positions=[{"instId": "BTC-USDT-SWAP", "pos": "-0.2"}]).problems)


def test_changed_contract_spec_is_a_problem():
    assert any("ETH-USDT-SWAP" in p and "ctVal" in p
               for p in run(public=FakePublic({"ETH-USDT-SWAP": {"ctVal": "0.05"}})).problems)


def test_read_failure_is_reported_not_raised():
    acc = FakeAccount()
    acc.fail = True
    r = preflight(acc, FakePublic(), CFG, DEFAULT_INSTRUMENTS)
    assert any("could not read" in p and "ConnectionError" in p for p in r.problems)


# --- final fix wave: leverage, buffer, calibration ----------------------------------
def test_swap_leverage_is_reported_as_a_warning_per_swap():
    r = run()
    assert r.ok
    assert "swap leverage BTC-USDT-SWAP = 5x" in r.warnings and "swap leverage ETH-USDT-SWAP = 5x" in r.warnings


def test_low_swap_leverage_is_a_problem():
    acc = FakeAccount()
    acc.leverage = "1"
    r = preflight(acc, FakePublic(), CFG, DEFAULT_INSTRUMENTS)
    assert any("swap leverage BTC-USDT-SWAP is 1x, below 2x" in p for p in r.problems)


def test_unreadable_swap_leverage_is_a_problem():
    acc = FakeAccount()
    acc.leverage_fail = True
    r = preflight(acc, FakePublic(), CFG, DEFAULT_INSTRUMENTS)
    assert any("could not read swap leverage BTC-USDT-SWAP" in p and "ConnectionError" in p for p in r.problems)


def test_thin_margin_buffer_is_a_warning_and_1_2x_stays_a_problem():
    thin = run(details=[{"ccy": "USDT", "cashBal": "350", "availBal": "350"}])     # 1.4 x 250
    assert thin.ok and any("thin margin buffer" in w for w in thin.warnings)
    assert not any("thin margin buffer" in w for w in run().warnings)
    assert not run(details=[{"ccy": "USDT", "cashBal": "200", "availBal": "200"}]).ok


def test_uncalibrated_config_is_a_warning_not_a_problem():
    r = preflight(FakeAccount(), FakePublic(), CarryConfig(mgn_calibrated=False), DEFAULT_INSTRUMENTS)
    assert r.ok and any("not calibrated" in w for w in r.warnings)
    assert not any("not calibrated" in w for w in run().warnings)


@pytest.mark.parametrize("raw", ["", None])
def test_missing_or_empty_leverage_value_is_a_problem_not_zero_x(raw):
    acc = FakeAccount()
    acc.leverage = raw
    r = preflight(acc, FakePublic(), CFG, DEFAULT_INSTRUMENTS)
    assert any("could not read swap leverage BTC-USDT-SWAP: " in p and "no leverage value" in p for p in r.problems)
    assert not any("below 2x" in p for p in r.problems)
