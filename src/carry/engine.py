"""One loop iteration of the carry strategy: guard -> watchdog -> risk -> signal -> execute."""
from __future__ import annotations

from typing import Callable

from carry.config import CarryConfig
from carry.decision import decide
from carry.exchange import Exchange
from carry.executor import Executor, HedgeError
from carry.hedge import HedgeStatus, check_hedge
from carry.risk import Tier, classify
from carry.state import CarryState, CoinLeg
from carry.types import Instrument
from carry.watchdog import Watchdog

SETTLEMENT_MS = 8 * 3600 * 1000


class Engine:
    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 alert: Callable[[str], None], state: CarryState | None = None,
                 save: Callable[[CarryState], None] | None = None) -> None:
        self.ex, self.insts, self.cfg, self.alert = ex, insts, cfg, alert
        self.state = state if state is not None else CarryState()
        self.save = save
        self.executor = Executor(ex, insts, cfg, self._alert)
        self.watchdog = Watchdog(ex, insts, cfg, self.executor, self._alert)
        self._last_tier = Tier.NORMAL
        self._open_block_until: dict[str, int] = {}
        self._open_fails: dict[str, int] = {}

    # --- public ----------------------------------------------------------
    def step(self, now_ms: int) -> list[str]:
        try:
            out = self._step(now_ms)
        except HedgeError as e:
            self._halt(str(e))
            out = [f"HALT {e}"]
        except Exception as e:  # noqa: BLE001 - fail safe: any unexpected error stops trading
            self._halt(f"unexpected error: {e!r}")
            out = ["HALT unexpected error"]
        finally:
            self._save()
        return out

    def force_open(self, coin: str, notional: float, now_ms: int) -> bool:
        try:
            if self.state.mode == "HALTED":
                raise RuntimeError("engine is halted; rearm() first")
            if coin not in self.cfg.coins:
                raise ValueError(f"unknown coin {coin!r}")
            if coin in self.state.legs:
                raise RuntimeError(f"{coin} already open")
            tier = self._tier()
            if tier != Tier.NORMAL:
                raise RuntimeError(f"risk tier is {tier.name}; refusing to open")
            try:
                return self._open(coin, notional, now_ms) is not None
            except HedgeError as e:
                self._halt(str(e))
                return False
        finally:
            self._save()

    def force_close(self, coin: str) -> bool:
        try:
            leg = self.state.legs.get(coin)
            if leg is None:
                return True
            if not self.executor.close(coin, leg):
                self._halt(f"close {coin} failed")
                return False
            del self.state.legs[coin]
            if not self.state.legs and self.state.mode == "HEDGED":
                self.state.mode = "FLAT"
            return True
        finally:
            self._save()

    def rearm(self) -> None:
        try:
            if self.state.legs:
                raise RuntimeError("legs still open; close them before rearm()")
            for coin in self.cfg.coins:
                inst = self.insts[coin]
                st = check_hedge(self.ex.spot_balance(coin), self.ex.swap_contracts(coin), inst, self.cfg.drift_tol)
                if st != HedgeStatus.FLAT:
                    raise RuntimeError(f"{coin} holdings on the exchange ({st.value}); resolve manually")
            self.state.mode = "FLAT"
            self.state.halted_reason = ""
        finally:
            self._save()

    # --- internals -------------------------------------------------------
    def _alert(self, msg: str) -> None:
        """Alert without ever raising: a broken alerter must not abort a safety action."""
        try:
            self.alert(msg)
        except Exception:  # noqa: BLE001
            pass

    def _save(self) -> None:
        if self.save is None:
            return
        try:
            self.save(self.state)
        except Exception as e:  # noqa: BLE001
            self._alert(f"[CRITICAL] state save failed: {e!r}")

    def _halt(self, reason: str) -> None:
        s = self.state
        s.mode = "HALTED"
        if reason not in s.halted_reason:
            s.halted_reason = f"{s.halted_reason}; {reason}" if s.halted_reason else reason
        self._alert(f"[CRITICAL] HALTED: {reason}")

    def _tier(self) -> Tier:
        return classify(self.ex.mgn_ratio(), bool(self.state.legs), self.cfg)

    def _open(self, coin: str, notional: float, now_ms: int) -> CoinLeg | None:
        leg = self.executor.open(coin, notional, now_ms)
        if leg is not None:
            self.state.legs[coin] = leg
            self.state.mode = "HEDGED"
            self._open_fails.pop(coin, None)
            self._open_block_until.pop(coin, None)
        return leg

    def _step(self, now_ms: int) -> list[str]:
        s = self.state
        out: list[str] = []

        if s.mode != "HALTED":
            for coin in self.cfg.coins:
                if coin in s.legs:
                    continue
                st = check_hedge(self.ex.spot_balance(coin), self.ex.swap_contracts(coin),
                                 self.insts[coin], self.cfg.drift_tol)
                if st != HedgeStatus.FLAT:
                    self._halt(f"unexpected {coin} holdings ({st.value}) with no recorded position; not trading")
                    return ["HALT unexpected holdings"]

        out += self.watchdog.check(s)

        # Risk is managed even when halted: a residual leg must keep being closed at CLOSE+.
        if s.legs or s.mode != "HALTED":
            tier = self._tier()
        else:
            tier = Tier.NORMAL
        if tier != self._last_tier:
            self._alert(f"risk tier {self._last_tier.name} -> {tier.name}")
            self._last_tier = tier
        if tier >= Tier.CLOSE and s.legs:
            for coin in list(s.legs):
                if self.executor.close(coin, s.legs[coin]):
                    del s.legs[coin]
                    out.append(f"risk-closed {coin}")
            if s.mode != "HALTED":
                self._halt(f"risk tier {tier.name}")
            return out
        if s.mode == "HALTED":
            return out

        for coin in self.cfg.coins:
            leg = s.legs.get(coin)
            held = max(0, int((now_ms - leg.entry_ts) / SETTLEMENT_MS)) if leg else 0
            action = decide(self.ex.funding_history(coin, self.cfg.window), leg is not None, held, self.cfg)
            if action == "open" and tier == Tier.NORMAL:
                if now_ms < self._open_block_until.get(coin, 0):
                    continue
                if self._open(coin, self.cfg.per_coin_notional, now_ms) is not None:
                    out.append(f"opened {coin}")
                    continue
                n = self._open_fails.get(coin, 0) + 1
                self._open_fails[coin] = n
                self._open_block_until[coin] = now_ms + self.cfg.open_cooldown_ms
                self._alert(f"open {coin} failed ({n}/{self.cfg.max_open_fails}); "
                            f"next attempt in {self.cfg.open_cooldown_ms // 1000}s")
                out.append(f"open {coin} failed")
                if n >= self.cfg.max_open_fails:
                    self._halt(f"open {coin} failed {n} times in a row")
                    return out
            elif action == "close" and leg is not None:
                if self.executor.close(coin, leg):
                    del s.legs[coin]
                    out.append(f"closed {coin}")
                else:
                    self._halt(f"close {coin} failed")
                    return out
        if not s.legs and s.mode == "HEDGED":
            s.mode = "FLAT"
        return out
