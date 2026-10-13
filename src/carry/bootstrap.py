"""Startup safety: unreadable state starts HALTED, reconcile adopts only healthy hedges, exposure report."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.hedge import HedgeStatus, check_hedge
from carry.state import CarryState, CoinLeg, load_state, save_state
from carry.types import Instrument


MODES = frozenset({"FLAT", "HEDGED", "HALTED"})


def _alert(alert: Callable[[str], None], msg: str) -> None:
    """Send an alert without ever raising: startup safety must not die on a broken alerter."""
    try:
        alert(msg)
    except Exception:  # noqa: BLE001
        pass


def _validate(state: CarryState) -> None:
    """Semantic checks on a parsed state; raises ValueError so the caller treats it as unreadable."""
    if state.mode not in MODES:
        raise ValueError(f"unknown mode {state.mode!r}")
    if state.mode == "FLAT" and state.legs:
        raise ValueError("mode FLAT but legs recorded")
    for coin, leg in state.legs.items():
        for name in ("contracts", "spot_qty"):
            v = getattr(leg, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{coin} leg {name} not finite and positive")
        if leg.entry_ts < 0:
            raise ValueError(f"{coin} leg entry_ts negative")


def load_state_or_halt(path: str | Path, alert: Callable[[str], None],
                       clock: Callable[[], float] = time.time) -> CarryState:
    """Load the state; anything unreadable/invalid is quarantined and replaced by a persisted HALTED state."""
    try:
        state = load_state(path)
        _validate(state)
        return state
    except Exception as e:  # noqa: BLE001 - corrupt JSON, unknown keys, bad types, invalid values
        p = Path(path)
        halted = CarryState(mode="HALTED", halted_reason=f"state file unreadable: {type(e).__name__}")
        bad = p.with_name(f"{p.name}.corrupt-{int(clock())}")
        try:
            p.rename(bad)
            renamed = True
        except OSError:
            renamed = False
        if renamed:
            try:
                # Persist the halt now: a crash before the engine's first step must not let
                # the next boot see "no file" and default to FLAT.
                save_state(p, halted)
            except OSError:
                pass
            _alert(alert, f"[CRITICAL] state file unreadable ({type(e).__name__}); "
                          f"moved to {bad.name}; starting HALTED")
        else:
            _alert(alert, f"[CRITICAL] state file unreadable ({type(e).__name__}) and could not be moved aside; "
                          f"starting HALTED; file left in place")
        return halted


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
        why = ""
        if status == HedgeStatus.OK and state.mode != "HALTED":
            try:
                notional = contracts * inst.ct_val * ex.price(coin)
            except Exception as e:  # noqa: BLE001
                why = f"; price unavailable ({type(e).__name__})"
            else:
                if notional <= 1.2 * cfg.max_order_notional:
                    # back-dated by min_hold so a restart does not restart the fade-exit clock
                    entry_ts = now_ms - cfg.min_hold * 8 * 3600 * 1000
                    state.legs[coin] = CoinLeg(contracts=contracts, spot_qty=spot, entry_ts=entry_ts)
                    state.mode = "HEDGED"
                    note = f"adopted existing {coin} hedge: {contracts} contracts, spot {spot}"
                    notes.append(note)
                    _alert(alert, f"[INFO] {note}")
                    continue
                why = f"; notional {notional:.2f} exceeds 1.2x max_order_notional"
        msg = (f"[CRITICAL] startup: {coin} exchange holdings not adopted "
               f"(status={status.value}, spot={spot}, perp={contracts} contracts, mode={state.mode}){why}")
        _alert(alert, msg)
        notes.append(msg)
    return notes


def exposure_report(ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig, state: CarryState) -> str:
    """Multi-line exposure summary. Never raises and never leaks exception text (class names only)."""
    try:
        lines = [f"mode={state.mode} reason={state.halted_reason or '-'} legs={sorted(state.legs)}"]
        for coin in cfg.coins:
            try:
                inst = insts[coin]
                spot, contracts = ex.spot_balance(coin), ex.swap_contracts(coin)
                status = check_hedge(spot, contracts, inst, cfg.drift_tol)
                lines.append(f"{coin}: spot={spot:.8g} perp={contracts:.8g} contracts "
                             f"hedge={status.value} adl={ex.swap_adl_rank(coin)}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"{coin}: unavailable ({type(e).__name__})")
        try:
            lines.append(f"mgnRatio={ex.mgn_ratio()}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"mgnRatio=unavailable ({type(e).__name__})")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"exposure unavailable: {type(e).__name__}"
