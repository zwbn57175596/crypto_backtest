"""Validated configuration for the carry engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarryConfig:
    coins: tuple[str, ...] = ("BTC", "ETH")
    deploy_usdt: float = 250.0          # total notional per leg, split equally across coins
    max_order_notional: float = 300.0   # hard cap for one open order (per leg, per coin)
    spot_taker_fee: float = 0.001
    drift_tol: float = 0.02             # allowed |spot - perp| / perp
    window: int = 9                     # trailing settlements used by the entry signal
    enter_apr: float = 0.06
    exit_apr: float = 0.02
    min_hold: int = 9                   # settlements before a fade-exit is allowed
    mgn_warn: float = 4.0               # mgnRatio tiers, in ratio form (1.0 = liquidation)
    mgn_close: float = 2.5
    mgn_emergency: float = 1.5
    mgn_unit: float = 1.0               # divide the raw API value by this to get ratio form
    adl_alert_rank: int = 4
    poll_sec: float = 2.0
    open_cooldown_ms: int = 1_800_000   # wait after a failed open before retrying that coin
    max_open_fails: int = 3             # consecutive failed opens on one coin before halting
    settle_sec: float = 0.0             # pause after each order before re-reading positions
    state_path: str = "live_state/carry_state.json"

    def __post_init__(self) -> None:
        if not self.coins:
            raise ValueError("coins must not be empty")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.min_hold < 0:
            raise ValueError("min_hold must be >= 0")
        if self.deploy_usdt <= 0:
            raise ValueError("deploy_usdt must be > 0")
        if not self.exit_apr < self.enter_apr:
            raise ValueError("exit_apr must be below enter_apr")
        if not self.mgn_emergency < self.mgn_close < self.mgn_warn:
            raise ValueError("need mgn_emergency < mgn_close < mgn_warn")
        if self.mgn_unit <= 0:
            raise ValueError("mgn_unit must be > 0")
        if self.open_cooldown_ms < 0:
            raise ValueError("open_cooldown_ms must be >= 0")
        if self.max_open_fails < 1:
            raise ValueError("max_open_fails must be >= 1")
        if self.settle_sec < 0:
            raise ValueError("settle_sec must be >= 0")
        if self.per_coin_notional > self.max_order_notional:
            raise ValueError("per-coin notional exceeds max_order_notional")

    @property
    def per_coin_notional(self) -> float:
        return self.deploy_usdt / len(self.coins)
