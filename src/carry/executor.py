"""Open/close a hedged position with rollback on partial failure.

The exchange is the source of truth: every order's outcome is judged by
re-reading balances/positions, never by the order result alone, and every
sale / buy-back is capped by a snapshot delta so pre-existing holdings are
never touched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.sizing import Sizing, drift, floor_to, plan_position
from carry.state import CoinLeg
from carry.types import Instrument, OrderResult


class HedgeError(RuntimeError):
    """Raised when a rollback fails and an unhedged leg may remain open."""


class PreTradeError(RuntimeError):
    """A read needed before the first order failed; no order has been placed, so this is safe to retry."""


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
                 alert: Callable[[str], None], sleep: Callable[[float], None] = time.sleep) -> None:
        self.ex, self.insts, self.cfg, self.alert = ex, insts, cfg, alert
        self._sleep = sleep

    def open(self, coin: str, notional: float, now_ms: int) -> CoinLeg | None:
        if not (0 < notional <= self.cfg.max_order_notional):
            raise ValueError(f"notional {notional} must be in (0, {self.cfg.max_order_notional}]")
        inst = self.insts[coin]
        try:
            price = self.ex.price(coin)
        except Exception as e:  # noqa: BLE001 - no order placed yet
            raise PreTradeError(f"{coin}: pre-trade read failed ({type(e).__name__})") from e
        sz = plan_position(notional, price, inst, self.cfg.spot_taker_fee)
        try:
            before_perp = self.ex.swap_contracts(coin)  # baselines: no exposure exists yet
            before_spot = self.ex.spot_balance(coin)
        except Exception as e:  # noqa: BLE001 - no order placed yet
            raise PreTradeError(f"{coin}: pre-trade read failed ({type(e).__name__})") from e

        try:
            return self._open_legs(coin, inst, sz, before_perp, before_spot, now_ms)
        except HedgeError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = f"[CRITICAL] open {coin}: unexpected error {e!r}; exposure may be open, verify manually"
            self._alert(msg)
            raise HedgeError(msg) from e

    def _open_legs(self, coin: str, inst: Instrument, sz: Sizing, before_perp: float,
                   before_spot: float, now_ms: int) -> CoinLeg | None:
        res: OrderResult | None = None
        try:                                                   # perp first: worst case is an extra short
            res = self.ex.sell_swap(coin, sz.contracts)
        except Exception as e:  # noqa: BLE001
            self._alert(f"open {coin}: perp order error: {e!r}")
        self._settle()
        held = self.ex.swap_contracts(coin) - before_perp
        if held < inst.swap_min and self._reported_fill(res):
            self._settle()                                     # the position read may lag the order result
            held = self.ex.swap_contracts(coin) - before_perp
            if held < inst.swap_min:
                self._alert(f"[CRITICAL] open {coin}: perp order reported {self._describe(res)} "
                            "but no position is visible \u2014 verify manually")
                return None
        if held < inst.swap_min:
            self._alert(f"open {coin}: perp order not filled")
            return None
        if held < sz.contracts - 1e-9:
            self._alert(f"open {coin}: partial perp fill {held}/{sz.contracts}, undoing")
            self._undo_short(coin, held)
            return None

        sres: OrderResult | None = None
        try:
            sres = self.ex.buy_spot(coin, sz.spot_qty)
        except Exception as e:  # noqa: BLE001
            self._alert(f"open {coin}: spot order error: {e!r}")
        self._settle()
        bought = self.ex.spot_balance(coin) - before_spot
        if bought < inst.spot_min and self._reported_fill(sres):
            self._settle()
            bought = self.ex.spot_balance(coin) - before_spot
            if bought < inst.spot_min:
                self._alert(f"[CRITICAL] open {coin}: spot order reported {self._describe(sres)} "
                            "but no balance increase is visible \u2014 verify manually")
        if bought >= inst.spot_min and abs(drift(bought, held, inst)) <= self.cfg.drift_tol:
            return CoinLeg(contracts=held, spot_qty=bought, entry_ts=now_ms)

        self._alert(f"open {coin}: hedge not established (spot {bought} vs perp {held} contracts), rolling back")
        if not self._drain_spot(coin, bought):
            msg = f"[CRITICAL] {coin}: could not liquidate {bought} spot during rollback; perp short kept to stay hedged"
            self._alert(msg)
            raise HedgeError(msg)
        self._undo_short(coin, held)
        return None

    def close(self, coin: str, leg: CoinLeg) -> CloseResult:
        try:
            if not self._drain_spot(coin, leg.spot_qty):
                self._alert(f"[CRITICAL] {coin}: spot leg not closed; perp left untouched to keep the hedge")
                return CloseResult(False, False)
            if not self._drain_perp(coin, leg.contracts):
                self._alert(f"[CRITICAL] {coin}: perp buy-back failed; naked short remains")
                return CloseResult(True, False)
            return CloseResult(True, True)
        except Exception as e:  # noqa: BLE001
            self._alert(f"[CRITICAL] close {coin}: unexpected error {e!r}; position state unknown, verify manually")
            return CloseResult(False, False)

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _reported_fill(res: OrderResult | None) -> bool:
        """The order result claims the order went through (or may have): a missing position is suspicious."""
        return res is not None and (res.ok or res.filled > 0 or res.error.startswith("UNCONFIRMED"))

    @staticmethod
    def _describe(res: OrderResult | None) -> str:
        if res is None:
            return "nothing"
        if res.ok:
            return f"ok (filled {res.filled})"
        return f"filled {res.filled}, error {res.error!r}" if res.filled > 0 else f"error {res.error!r}"

    def _alert(self, msg: str) -> None:
        """Alert without ever raising: a broken alerter must not skip a rollback step."""
        try:
            self.alert(msg)
        except Exception:  # noqa: BLE001
            pass

    def _settle(self) -> None:
        if self.cfg.settle_sec > 0:
            self._sleep(self.cfg.settle_sec)

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
                res = order(excess)           # success is judged by the next read(); the result is for visibility
            except Exception as e:            # noqa: BLE001 - exchange errors count as a failed attempt
                self._alert(f"order error: {e!r}")
            else:
                if isinstance(res, OrderResult) and not res.ok and res.filled <= 0:
                    self._alert(f"order rejected: {res.error}")
            self._settle()
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
            self._alert(msg)
            raise HedgeError(msg)
