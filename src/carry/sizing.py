"""Order sizing: lot rounding, spot/perp quantity matching, hedge drift."""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from carry.types import Instrument


def _steps(x: float, step: float, rounding: str) -> float:
    d = Decimal(str(round(x, 12))) / Decimal(str(step))
    return float(d.to_integral_value(rounding=rounding) * Decimal(str(step)))


def floor_to(x: float, step: float) -> float:
    return _steps(x, step, ROUND_FLOOR)


def ceil_to(x: float, step: float) -> float:
    return _steps(x, step, ROUND_CEILING)


@dataclass(frozen=True)
class Sizing:
    contracts: float   # swap contracts to short
    coin_qty: float    # perp exposure in coins = contracts * ct_val
    spot_qty: float    # spot base quantity to buy (grossed up for the fee deducted in kind)
    notional: float    # coin_qty * price


def plan_position(notional_usdt: float, price: float, inst: Instrument,
                  spot_taker_fee: float) -> Sizing:
    if not (math.isfinite(price) and price > 0):
        raise ValueError(f"price must be finite and > 0, got {price}")
    if not (math.isfinite(notional_usdt) and notional_usdt > 0):
        raise ValueError(f"notional must be finite and > 0, got {notional_usdt}")
    coins = notional_usdt / price
    contracts = floor_to(coins / inst.ct_val, inst.swap_lot)
    if contracts < inst.swap_min:
        raise ValueError(f"{inst.coin}: notional {notional_usdt} too small for min {inst.swap_min} contracts")
    coin_qty = float(Decimal(str(contracts)) * Decimal(str(inst.ct_val)))
    spot_qty = ceil_to(coin_qty / (1.0 - spot_taker_fee), inst.spot_lot)
    if spot_qty < inst.spot_min:
        raise ValueError(f"{inst.coin}: spot quantity {spot_qty} below minimum {inst.spot_min}")
    return Sizing(contracts, coin_qty, spot_qty, coin_qty * price)


def drift(spot_qty: float, contracts: float, inst: Instrument) -> float:
    """(spot - perp) / perp in coin terms; inf if perp is zero but spot is not."""
    perp = contracts * inst.ct_val
    if perp == 0:
        return math.inf if spot_qty > 0 else 0.0
    return (spot_qty - perp) / perp
