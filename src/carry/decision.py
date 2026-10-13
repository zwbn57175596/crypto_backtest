"""Entry/exit decision from trailing funding (mirrors the validated simulator rule)."""
from __future__ import annotations

from typing import Sequence

from carry.config import CarryConfig
from carry.funding_math import funding_apr


def decide(rates: Sequence[float], in_position: bool, held: int, p: CarryConfig) -> str:
    """rates: last settled funding rates, oldest first. held: settlements since entry."""
    if len(rates) < p.window:
        return "hold"
    avg_apr = funding_apr(sum(rates[-p.window:]) / p.window)
    if not in_position:
        return "open" if avg_apr >= p.enter_apr else "hold"
    if avg_apr < p.exit_apr and held >= p.min_hold:
        return "close"
    return "hold"
