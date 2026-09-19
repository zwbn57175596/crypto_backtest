from pathlib import Path

from carry.okx_probe import collect_probe, render_markdown


def ok(data):
    return {"code": "0", "msg": "", "data": data}


class FakePublic:
    def get_instruments(self, instType, instId):
        if instType == "SWAP":
            return ok([{"ctVal": "0.01", "ctMult": "1", "lotSz": "1", "minSz": "1", "tickSz": "0.1"}])
        return ok([{"lotSz": "0.00000001", "minSz": "0.00001", "tickSz": "0.1"}])

    def get_funding_rate(self, instId):
        return ok([{"fundingRate": "0.0001", "fundingTime": "1700000000000",
                    "nextFundingTime": "1700028800000"}])

    def get_interest_rate_loan_quota(self):
        return ok([{"basic": [{"ccy": "USDT", "rate": "0.0000015", "quota": "1000000"}]}])


class FakeAccount:
    def get_account_config(self):
        return ok([{"acctLv": "3", "autoLoan": True, "posMode": "net_mode"}])

    def get_interest_rate(self, ccy):
        return ok([{"ccy": ccy, "interestRate": "0.00000833"}])

    def get_fee_rates(self, instType):
        return ok([{"maker": "-0.0008", "taker": "-0.001"}])

    def get_account_balance(self):
        return ok([{"totalEq": "1000", "adjEq": "990", "imr": "0", "mmr": "0",
                    "mgnRatio": "", "notionalUsd": "0"}])


def test_collect_public_only():
    out = collect_probe(FakePublic(), None)
    assert out["swaps"]["BTC-USDT-SWAP"]["ctVal"] == "0.01"
    assert out["funding"]["ETH-USDT-SWAP"]["interval_h"] == 8.0
    assert out["spots"]["BTC-USDT"]["minSz"] == "0.00001"
    assert "account" not in out


def test_collect_with_account():
    out = collect_probe(FakePublic(), FakeAccount())
    assert out["account"]["acctLv"] == "3"
    assert out["borrow"]["apr_if_hourly"] == 0.00000833 * 24 * 365
    assert out["fees"]["SPOT"]["taker"] == "-0.001"
    assert out["balance"]["mgnRatio"] == ""


def test_render_markdown_mentions_key_facts():
    md = render_markdown(collect_probe(FakePublic(), FakeAccount()))
    for needle in ("acctLv", "BTC-USDT-SWAP", "apr_if_hourly", "mgnRatio"):
        assert needle in md


def test_probe_code_is_read_only():
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.glob("src/carry/*.py")) + sorted(root.glob("scripts/carry_*.py"))
    assert files, "no carry sources found to scan"
    banned_names = ("place_order", "place_multiple_orders", "amend_order", "cancel_order",
                    "close_positions", "set_leverage", "set_account_level", "borrow_repay",
                    "funds_transfer", "withdrawal", "purchase_redempt",
                    "TradeAPI", "FundingAPI", "SubAccountAPI")
    banned_imports = ("import okx.Trade", "import okx.Funding", "import okx.SubAccount",
                      "from okx import Trade", "from okx import Funding",
                      "from okx import SubAccount", "okx.Trade", "okx.Funding", "okx.SubAccount")
    for path in files:
        text = path.read_text()
        for banned in banned_names + banned_imports:
            assert banned not in text, f"{banned} found in {path.relative_to(root)}"
