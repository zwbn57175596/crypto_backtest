"""Pre-trade account checks. Read-only."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from carry.config import CarryConfig
from carry.types import Instrument


@dataclass
class PreflightResult:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _num(x: Any) -> float:
    return 0.0 if x in (None, "") else float(x)


def _rows(resp: dict, what: str) -> list[dict]:
    if str(resp.get("code")) != "0":
        raise RuntimeError(f"{what}: code={resp.get('code')} msg={resp.get('msg')}")
    return resp["data"]


def preflight(account: Any, public: Any, cfg: CarryConfig, insts: dict[str, Instrument]) -> PreflightResult:
    r = PreflightResult()

    def guarded(label: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            r.problems.append(f"could not read {label}: {type(e).__name__}: {e}")

    def check_config() -> None:
        c = _rows(account.get_account_config(), "account config")[0]
        if str(c.get("acctLv")) != "3":
            r.problems.append(f"acctLv is {c.get('acctLv')}, need 3 (multi-currency margin)")
        if c.get("autoLoan") not in (False, "false"):
            r.problems.append("autoLoan must be off (no borrowing in v1)")
        if c.get("posMode") != "net_mode":
            r.problems.append(f"posMode is {c.get('posMode')}, need net_mode")

    def check_balances() -> None:
        details = _rows(account.get_account_balance(), "balance")[0].get("details", [])
        usdt = 0.0
        for d in details:
            ccy, cash = d.get("ccy"), _num(d.get("cashBal"))
            if ccy in cfg.coins:
                if abs(cash) >= insts[ccy].spot_min:
                    r.problems.append(f"account holds {cash} {ccy}; the account must hold no {ccy} before the bot starts")
            elif ccy == "USDT":
                usdt = _num(d.get("availBal") if d.get("availBal") not in (None, "") else cash)
            elif abs(cash) > 0:
                r.warnings.append(f"other balance: {cash} {ccy}")
        if usdt < 1.2 * cfg.deploy_usdt:
            r.problems.append(f"USDT available {usdt} < 1.2 x deploy_usdt ({cfg.deploy_usdt})")
        elif usdt < 1.6 * cfg.deploy_usdt:
            r.warnings.append(f"thin margin buffer: USDT available {usdt} < 1.6 x deploy_usdt ({cfg.deploy_usdt})")

    def check_positions() -> None:
        for p in _rows(account.get_positions(instType="SWAP"), "positions"):
            if _num(p.get("pos")) != 0:
                r.problems.append(f"open swap position {p.get('instId')}: {p.get('pos')}")

    def check_specs() -> None:
        for coin in cfg.coins:
            i = insts[coin]
            swap = _rows(public.get_instruments(instType="SWAP", instId=i.swap_id), "swap spec")[0]
            spot = _rows(public.get_instruments(instType="SPOT", instId=i.spot_id), "spot spec")[0]
            for inst_id, row, want in (
                (i.swap_id, swap, {"ctVal": i.ct_val, "lotSz": i.swap_lot, "minSz": i.swap_min}),
                (i.spot_id, spot, {"lotSz": i.spot_lot, "minSz": i.spot_min}),
            ):
                for key, expected in want.items():
                    if not math.isclose(_num(row.get(key)), expected, rel_tol=1e-9):
                        r.problems.append(f"{inst_id} {key} is {row.get(key)}, expected {expected}")

    def check_leverage(inst: Instrument) -> None:
        rows = _rows(account.get_leverage(mgnMode="cross", instId=inst.swap_id), "leverage")
        lever = _num(rows[0].get("lever"))
        if lever < 2:
            r.problems.append(f"swap leverage {inst.swap_id} is {lever:g}x, below 2x")
        else:
            r.warnings.append(f"swap leverage {inst.swap_id} = {lever:g}x")

    if not cfg.mgn_calibrated:
        r.warnings.append("mgn_unit/thresholds not calibrated (mgn_calibrated=false): signal-driven opens are disabled")
    guarded("account config", check_config)
    guarded("balances", check_balances)
    guarded("positions", check_positions)
    guarded("instrument specs", check_specs)
    for coin in cfg.coins:
        guarded(f"swap leverage {insts[coin].swap_id}", lambda i=insts[coin]: check_leverage(i))
    return r
