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


TRADE_VERBS = ("place_order", "cancel_order", "TradeAPI", "import okx.Trade",
               "from okx import Trade", "okx.Trade")
TRADE_EXEMPT = ("src/carry/okx_exchange.py", "src/carry/cli.py")
ALWAYS_BANNED = ("place_multiple_orders", "amend_order", "close_positions", "set_leverage",
                 "set_account_level", "borrow_repay", "funds_transfer", "withdrawal",
                 "purchase_redempt", "SubAccountAPI", "FundingAPI", "okx.Funding",
                 "okx.SubAccount", "import okx.Funding", "import okx.SubAccount",
                 "from okx import Funding", "from okx import SubAccount")


def find_violations(rel_path: str, text: str) -> list[str]:
    """Banned tokens in `text`: ALWAYS_BANNED everywhere, TRADE_VERBS outside TRADE_EXEMPT."""
    found = [b for b in ALWAYS_BANNED if b in text]
    if rel_path not in TRADE_EXEMPT:
        found += [b for b in TRADE_VERBS if b in text]
    return found


def _scanned_files() -> tuple[Path, list[Path]]:
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.glob("src/carry/*.py")) + sorted(root.glob("scripts/carry_*.py"))
    assert files, "no carry sources found to scan"
    return root, files


def test_probe_code_is_read_only():
    root, files = _scanned_files()
    for path in files:
        rel = path.relative_to(root).as_posix()
        assert find_violations(rel, path.read_text()) == [], f"banned token(s) in {rel}"


def test_adapter_is_the_sanctioned_trade_caller():
    text = (Path(__file__).resolve().parents[1] / "src/carry/okx_exchange.py").read_text()
    assert "place_order" in text and "cancel_order" in text


def test_guard_flags_trade_verb_in_non_exempt_module():
    assert find_violations("src/carry/other.py", "self.trade.place_order(**kw)") == ["place_order"]
    assert find_violations("src/carry/okx_exchange.py", "self.trade.place_order(**kw)") == []
    assert find_violations("src/carry/okx_exchange.py", "x.set_leverage()") == ["set_leverage"]
