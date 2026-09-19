"""Startup safety: unreadable state starts HALTED, reconcile adopts only healthy hedges, exposure report."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.hedge import HedgeStatus, check_hedge
from carry.state import CarryState, CoinLeg, load_state
from carry.types import Instrument


def load_state_or_halt(path: str | Path, alert: Callable[[str], None],
                       clock: Callable[[], float] = time.time) -> CarryState:
    try:
        return load_state(path)
    except Exception as e:  # noqa: BLE001 - corrupt JSON, unknown keys, bad types
        p = Path(path)
        bad = p.with_name(f"{p.name}.corrupt-{int(clock())}")
        try:
            p.rename(bad)
        except OSError:
            pass
        alert(f"[CRITICAL] state file unreadable ({type(e).__name__}); moved to {bad.name}; starting HALTED")
        return CarryState(mode="HALTED", halted_reason=f"state file unreadable: {type(e).__name__}")


def reconcile(ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig, state: CarryState,
              alert: Callable[[str], None], now_ms: int) -> list[str]:
    """Compare the exchange with the state at startup. Adopts healthy hedges only; never trades."""
    notes: list[str] = []
    for coin in cfg.coins:
        if coin in state.legs:
            continue
        inst = insts[coin]
        spot, contracts = ex.spot_balance(coin), ex.swap_contracts(coin)
        status = check_hedge(spot, contracts, inst, cfg.drift_tol)
        if status == HedgeStatus.FLAT:
            continue
        notional = contracts * inst.ct_val * ex.price(coin)
        if (status == HedgeStatus.OK and state.mode != "HALTED"
                and notional <= 1.2 * cfg.max_order_notional):
            state.legs[coin] = CoinLeg(contracts=contracts, spot_qty=spot, entry_ts=now_ms)
            state.mode = "HEDGED"
            notes.append(f"adopted existing {coin} hedge: {contracts} contracts, spot {spot}")
            continue
        msg = (f"[CRITICAL] startup: {coin} exchange holdings not adopted "
               f"(status={status.value}, spot={spot}, perp={contracts} contracts, mode={state.mode})")
        alert(msg)
        notes.append(msg)
    return notes


def exposure_report(ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig, state: CarryState) -> str:
    try:
        lines = [f"mode={state.mode} reason={state.halted_reason or '-'} legs={sorted(state.legs)}"]
        for coin in cfg.coins:
            inst = insts[coin]
            spot, contracts = ex.spot_balance(coin), ex.swap_contracts(coin)
            status = check_hedge(spot, contracts, inst, cfg.drift_tol)
            lines.append(f"{coin}: spot={spot:.8g} perp={contracts:.8g} contracts "
                         f"hedge={status.value} adl={ex.swap_adl_rank(coin)}")
        lines.append(f"mgnRatio={ex.mgn_ratio()}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"exposure unavailable: {type(e).__name__}: {e}"
