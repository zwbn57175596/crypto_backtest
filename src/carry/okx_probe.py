"""Read-only OKX facts needed to design the carry strategy (no trading calls)."""
from __future__ import annotations


def _data(resp: dict, what: str) -> list[dict]:
    if resp.get("code") != "0":
        raise RuntimeError(f"OKX {what}: code={resp.get('code')} msg={resp.get('msg')}")
    return resp["data"]


def collect_probe(public, account,
                  swap_ids: tuple[str, ...] = ("BTC-USDT-SWAP", "ETH-USDT-SWAP"),
                  spot_ids: tuple[str, ...] = ("BTC-USDT", "ETH-USDT")) -> dict:
    out: dict = {"swaps": {}, "spots": {}, "funding": {}, "loan_quota": None}
    for i in swap_ids:
        d = _data(public.get_instruments(instType="SWAP", instId=i), f"instruments {i}")[0]
        out["swaps"][i] = {k: d[k] for k in ("ctVal", "ctMult", "lotSz", "minSz", "tickSz")}
        f = _data(public.get_funding_rate(instId=i), f"funding-rate {i}")[0]
        out["funding"][i] = {
            "rate": float(f["fundingRate"]),
            "interval_h": (int(f["nextFundingTime"]) - int(f["fundingTime"])) / 3_600_000,
        }
    for i in spot_ids:
        d = _data(public.get_instruments(instType="SPOT", instId=i), f"instruments {i}")[0]
        out["spots"][i] = {k: d[k] for k in ("lotSz", "minSz", "tickSz")}
    try:
        out["loan_quota"] = _data(public.get_interest_rate_loan_quota(), "loan-quota")
    except AttributeError:
        out["loan_quota"] = "unavailable: method missing in installed python-okx"

    if account is not None:
        cfg = _data(account.get_account_config(), "account-config")[0]
        out["account"] = {k: cfg.get(k) for k in ("acctLv", "autoLoan", "posMode")}
        ir = _data(account.get_interest_rate(ccy="USDT"), "interest-rate")[0]
        out["borrow"] = {"raw_interestRate": ir["interestRate"],
                         # ASSUMPTION: OKX quotes this per hour — confirm against the web UI
                         "apr_if_hourly": float(ir["interestRate"]) * 24 * 365}
        out["fees"] = {}
        for t in ("SPOT", "SWAP"):
            d = _data(account.get_fee_rates(instType=t), f"fee-rates {t}")[0]
            out["fees"][t] = {"maker": d["maker"], "taker": d["taker"]}  # OKX sign: negative = fee paid
        bal = _data(account.get_account_balance(), "balance")[0]
        out["balance"] = {k: bal.get(k) for k in
                          ("totalEq", "adjEq", "imr", "mmr", "mgnRatio", "notionalUsd")}
    return out


def render_markdown(findings: dict) -> str:
    lines = ["# OKX probe (read-only)", ""]
    for section in ("account", "borrow", "fees", "balance", "loan_quota"):
        if section in findings:
            lines += [f"## {section}", "```", str(findings[section]), "```", ""]
    lines += ["## swaps", "```", *(f"{k}: {v}" for k, v in findings["swaps"].items()), "```", ""]
    lines += ["## spots", "```", *(f"{k}: {v}" for k, v in findings["spots"].items()), "```", ""]
    lines += ["## funding", "```", *(f"{k}: {v}" for k, v in findings["funding"].items()), "```", ""]
    return "\n".join(lines)
