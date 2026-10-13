"""Hedge-integrity watchdog: an ADL/liquidation of the perp leg flattens the spot leg at once."""
from __future__ import annotations

from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.executor import Executor
from carry.hedge import HedgeStatus, check_hedge
from carry.state import CarryState
from carry.types import Instrument


class WatchdogError(RuntimeError):
    """Raised after a full pass when at least one coin could not be checked."""


class Watchdog:
    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 executor: Executor, alert: Callable[[str], None]) -> None:
        self.ex, self.insts, self.cfg = ex, insts, cfg
        self.executor, self.alert = executor, alert
        self._last_rank: dict[str, int] = {}

    def _alert(self, msg: str) -> None:
        """Alert without ever raising: a broken alerter must not skip a flatten step."""
        try:
            self.alert(msg)
        except Exception:  # noqa: BLE001
            pass

    def check(self, state: CarryState) -> list[str]:
        actions: list[str] = []
        errors: dict[str, Exception] = {}
        for coin in list(state.legs):
            try:
                self._check_coin(state, coin, actions)
            except Exception as e:  # noqa: BLE001 - one coin must not stop the others
                self._alert(f"[CRITICAL] watchdog could not check {coin}: {e!r}")
                errors[coin] = e
        if errors:
            first = next(iter(errors.values()))
            raise WatchdogError(f"watchdog could not check {', '.join(errors)}; first error: {first!r}")
        return actions

    def _check_coin(self, state: CarryState, coin: str, actions: list[str]) -> None:
        leg = state.legs[coin]
        inst = self.insts[coin]

        try:
            rank = self.ex.swap_adl_rank(coin)
            if rank >= self.cfg.adl_alert_rank and self._last_rank.get(coin) != rank:
                self._alert(f"ADL rank {rank} on {coin} (alert only, hedge intact)")
            self._last_rank[coin] = rank
        except Exception as e:  # noqa: BLE001 - advisory only; do the authoritative check regardless
            self._alert(f"ADL rank unavailable for {coin}: {e!r}")

        contracts = self.ex.swap_contracts(coin)
        spot = self.ex.spot_balance(coin)
        status = check_hedge(spot, contracts, inst, self.cfg.drift_tol)
        shrunk = contracts < leg.contracts - 1e-9
        if status == HedgeStatus.OK and not shrunk:
            return

        # Halt first so the engine stops trading even if anything below fails.
        reason = f"hedge broken on {coin}"
        state.mode = "HALTED"
        if reason not in state.halted_reason:
            state.halted_reason = f"{state.halted_reason}; {reason}" if state.halted_reason else reason

        self._alert(f"[CRITICAL] hedge broken on {coin}: status={status.value}, "
                    f"perp {contracts} vs recorded {leg.contracts}, spot {spot}; flattening")
        if self.executor.close(coin, leg):
            del state.legs[coin]
            self._last_rank.pop(coin, None)
            actions.append(f"closed {coin} (hedge broken)")
        else:
            self._alert(f"[CRITICAL] FAILED to flatten {coin}; manual action needed")
            actions.append(f"FAILED to close {coin}")
