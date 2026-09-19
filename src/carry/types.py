"""Shared value types for the carry engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    coin: str
    spot_id: str
    swap_id: str
    ct_val: float      # coins per swap contract
    swap_lot: float    # contract lot size
    swap_min: float    # minimum contracts
    spot_lot: float    # base-coin lot size
    spot_min: float    # minimum base-coin quantity


BTC = Instrument("BTC", "BTC-USDT", "BTC-USDT-SWAP", 0.01, 0.01, 0.01, 1e-8, 1e-5)
ETH = Instrument("ETH", "ETH-USDT", "ETH-USDT-SWAP", 0.1, 0.01, 0.01, 1e-6, 1e-4)
DEFAULT_INSTRUMENTS: dict[str, Instrument] = {"BTC": BTC, "ETH": ETH}


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    filled: float = 0.0       # base coins (spot) or contracts (swap), gross
    avg_price: float = 0.0
    error: str = ""
