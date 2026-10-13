"""Persistent engine state (a cache: the exchange is the source of truth)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CoinLeg:
    contracts: float
    spot_qty: float
    entry_ts: int


@dataclass
class CarryState:
    mode: str = "FLAT"                       # FLAT | HEDGED | HALTED
    legs: dict[str, CoinLeg] = field(default_factory=dict)
    halted_reason: str = ""

    def to_dict(self) -> dict:
        return {"mode": self.mode, "halted_reason": self.halted_reason,
                "legs": {c: asdict(l) for c, l in self.legs.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "CarryState":
        return cls(mode=d.get("mode", "FLAT"), halted_reason=d.get("halted_reason", ""),
                   legs={c: CoinLeg(**l) for c, l in d.get("legs", {}).items()})


def save_state(path: str | Path, state: CarryState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    os.replace(tmp, p)


def load_state(path: str | Path) -> CarryState:
    p = Path(path)
    if not p.exists():
        return CarryState()
    return CarryState.from_dict(json.loads(p.read_text()))
