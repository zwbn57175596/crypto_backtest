"""Account-level margin-ratio risk tiers."""
from __future__ import annotations

from enum import IntEnum

from carry.config import CarryConfig


class Tier(IntEnum):
    NORMAL = 0
    WARN = 1
    CLOSE = 2
    EMERGENCY = 3


def classify(mgn_ratio: float | None, has_position: bool, cfg: CarryConfig) -> Tier:
    if not has_position:
        return Tier.NORMAL
    if mgn_ratio is None:
        return Tier.WARN
    r = mgn_ratio / cfg.mgn_unit
    if r <= cfg.mgn_emergency:
        return Tier.EMERGENCY
    if r <= cfg.mgn_close:
        return Tier.CLOSE
    if r <= cfg.mgn_warn:
        return Tier.WARN
    return Tier.NORMAL
