"""Open/close a hedged position with rollback on partial failure."""
from __future__ import annotations

from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.sizing import floor_to, plan_position
from carry.state import CoinLeg
from carry.types import Instrument, OrderResult


class HedgeError(RuntimeError):
    """Raised when a rollback fails and an unhedged leg may remain open."""


def _retry(fn: Callable[[], OrderResult], tries: int = 2) -> OrderResult:
    r = fn()
    for _ in range(tries - 1):
        if r.ok:
            break
        r = fn()
    return r


class Executor:
    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 alert: Callable[[str], None]) -> None:
        self.ex, self.insts, self.cfg, self.alert = ex, insts, cfg, alert

    def open(self, coin: str, notional: float, now_ms: int) -> CoinLeg | None:
        if notional > self.cfg.max_order_notional:
            raise ValueError(f"notional {notional} exceeds max_order_notional {self.cfg.max_order_notional}")
        inst = self.insts[coin]
        sz = plan_position(notional, self.ex.price(coin), inst, self.cfg.spot_taker_fee)

        short = self.ex.sell_swap(coin, sz.contracts)          # perp first: worst case is an extra short
        if not short.ok or short.filled <= 0:
            self.alert(f"open {coin}: perp order failed ({short.error})")
            return None
        if short.filled < sz.contracts - 1e-9:
            self.alert(f"open {coin}: partial perp fill {short.filled}/{sz.contracts}, undoing")
            self._undo_short(coin, short.filled)
            return None

        buy = self.ex.buy_spot(coin, sz.spot_qty)
        if not buy.ok or buy.filled < sz.spot_qty - 1e-12:
            self.alert(f"open {coin}: spot leg failed ({buy.error or 'partial'}), rolling back")
            self._sell_all_spot(coin)
            self._undo_short(coin, short.filled)
            return None
        return CoinLeg(contracts=short.filled, spot_qty=self.ex.spot_balance(coin), entry_ts=now_ms)

    def close(self, coin: str, leg: CoinLeg) -> bool:
        inst = self.insts[coin]
        ok = True
        spot_qty = floor_to(min(self.ex.spot_balance(coin), leg.spot_qty), inst.spot_lot)
        if spot_qty >= inst.spot_min:
            r = _retry(lambda: self.ex.sell_spot(coin, spot_qty))
            ok = ok and r.ok and r.filled > 0
        contracts = floor_to(min(self.ex.swap_contracts(coin), leg.contracts), inst.swap_lot)
        if contracts >= inst.swap_min:
            r = _retry(lambda: self.ex.buy_swap(coin, contracts))
            ok = ok and r.ok and r.filled > 0
        return ok

    # --- helpers ---------------------------------------------------------
    def _sell_all_spot(self, coin: str) -> None:
        inst = self.insts[coin]
        qty = floor_to(self.ex.spot_balance(coin), inst.spot_lot)
        if qty >= inst.spot_min:
            _retry(lambda: self.ex.sell_spot(coin, qty))

    def _undo_short(self, coin: str, contracts: float) -> None:
        r = _retry(lambda: self.ex.buy_swap(coin, contracts))
        if not r.ok:
            msg = f"[CRITICAL] {coin}: could not undo {contracts} short contracts ({r.error}); naked short open"
            self.alert(msg)
            raise HedgeError(msg)
