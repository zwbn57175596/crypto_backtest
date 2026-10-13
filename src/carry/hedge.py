"""Hedge-integrity classification of a spot/perp pair."""
from __future__ import annotations

from enum import Enum

from carry.sizing import drift
from carry.types import Instrument


class HedgeStatus(str, Enum):
    FLAT = "FLAT"
    OK = "OK"
    DRIFT = "DRIFT"
    SPOT_ONLY = "SPOT_ONLY"
    PERP_ONLY = "PERP_ONLY"


def check_hedge(spot_qty: float, contracts: float, inst: Instrument, tol: float) -> HedgeStatus:
    spot_open = spot_qty >= inst.spot_min
    perp_open = contracts >= inst.swap_min
    if not spot_open and not perp_open:
        return HedgeStatus.FLAT
    if spot_open and not perp_open:
        return HedgeStatus.SPOT_ONLY
    if perp_open and not spot_open:
        return HedgeStatus.PERP_ONLY
    return HedgeStatus.OK if abs(drift(spot_qty, contracts, inst)) <= tol else HedgeStatus.DRIFT
