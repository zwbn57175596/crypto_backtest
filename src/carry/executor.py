"""Open/close a hedged position with rollback on partial failure.

The exchange is the source of truth: every order's outcome is judged by
re-reading balances/positions, never by the order result alone, and every
sale / buy-back is capped by a snapshot delta so pre-existing holdings are
never touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.sizing import drift, floor_to, plan_position
from carry.state import CoinLeg
from carry.types import Instrument


class HedgeError(RuntimeError):
    """Raised when a rollback fails and an unhedged leg may remain open."""


@dataclass(frozen=True)
class CloseResult:
    spot_closed: bool
    perp_closed: bool

    @property
    def ok(self) -> bool:
        return self.spot_closed and self.perp_closed

    def __bool__(self) -> bool:
        return self.ok


class Executor:
    MAX_TRIES = 3

    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 alert: Callable[[str], None]) -> None:
        self.ex, self.insts, self.cfg, self.alert = ex, insts, cfg, alert

    def open(self, coin: str, notional: float, now_ms: int) -> CoinLeg | None:
        if not (0 < notional <= self.cfg.max_order_notional):
            raise ValueError(f"notional {notional} must be in (0, {self.cfg.max_order_notional}]")
        inst = self.insts[coin]
        sz = plan_position(notional, self.ex.price(coin), inst, self.cfg.spot_taker_fee)
        before_perp = self.ex.swap_contracts(coin)
        before_spot = self.ex.spot_balance(coin)

        try:                                                   # perp first: worst case is an extra short
            self.ex.sell_swap(coin, sz.contracts)
        except Exception as e:  # noqa: BLE001
            self.alert(f"open {coin}: perp order error: {e!r}")
        held = self.ex.swap_contracts(coin) - before_perp
        if held < inst.swap_min:
            self.alert(f"open {coin}: perp order not filled")
            return None
        if held < sz.contracts - 1e-9:
            self.alert(f"open {coin}: partial perp fill {held}/{sz.contracts}, undoing")
            self._undo_short(coin, held)
            return None

        try:
            self.ex.buy_spot(coin, sz.spot_qty)
        except Exception as e:  # noqa: BLE001
            self.alert(f"open {coin}: spot order error: {e!r}")
        bought = self.ex.spot_balance(coin) - before_spot
        if bought >= inst.spot_min and abs(drift(bought, held, inst)) <= self.cfg.drift_tol:
            return CoinLeg(contracts=held, spot_qty=bought, entry_ts=now_ms)

        self.alert(f"open {coin}: hedge not established (spot {bought} vs perp {held} contracts), rolling back")
        if not self._drain_spot(coin, bought):
            msg = f"[CRITICAL] {coin}: could not liquidate {bought} spot during rollback; perp short kept to stay hedged"
            self.alert(msg)
            raise HedgeError(msg)
        self._undo_short(coin, held)
        return None

    def close(self, coin: str, leg: CoinLeg) -> CloseResult:
        if not self._drain_spot(coin, leg.spot_qty):
            self.alert(f"[CRITICAL] {coin}: spot leg not closed; perp left untouched to keep the hedge")
            return CloseResult(False, False)
        if not self._drain_perp(coin, leg.contracts):
            self.alert(f"[CRITICAL] {coin}: perp buy-back failed; naked short remains")
            return CloseResult(True, False)
        return CloseResult(True, True)

    # --- helpers ---------------------------------------------------------
    def _drain(self, read: Callable[[], float], order: Callable[[float], object],
               cap: float, lot: float, minimum: float) -> bool:
        """Reduce a position by up to `cap` using `order(qty)`; success judged by re-reading `read()`."""
        start = read()
        target = max(start - cap, 0.0)
        for _ in range(self.MAX_TRIES):
            excess = floor_to(read() - target, lot)
            if excess < minimum:
                return True
            try:
                order(excess)                 # result ignored; the next read() decides
            except Exception as e:            # noqa: BLE001 - exchange errors count as a failed attempt
                self.alert(f"order error: {e!r}")
        return floor_to(read() - target, lot) < minimum

    def _drain_spot(self, coin: str, cap: float) -> bool:
        inst = self.insts[coin]
        return self._drain(lambda: self.ex.spot_balance(coin), lambda q: self.ex.sell_spot(coin, q),
                           cap, inst.spot_lot, inst.spot_min)

    def _drain_perp(self, coin: str, cap: float) -> bool:
        inst = self.insts[coin]
        return self._drain(lambda: self.ex.swap_contracts(coin), lambda q: self.ex.buy_swap(coin, q),
                           cap, inst.swap_lot, inst.swap_min)

    def _undo_short(self, coin: str, contracts: float) -> None:
        if not self._drain_perp(coin, contracts):
            msg = f"[CRITICAL] {coin}: could not undo {contracts} short contracts; naked short open"
            self.alert(msg)
            raise HedgeError(msg)
