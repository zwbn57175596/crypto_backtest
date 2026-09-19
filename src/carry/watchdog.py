"""Hedge-integrity watchdog: an ADL/liquidation of the perp leg flattens the spot leg at once."""
from __future__ import annotations

from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.executor import Executor
from carry.hedge import HedgeStatus, check_hedge
from carry.state import CarryState
from carry.types import Instrument


class Watchdog:
    def __init__(self, ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig,
                 executor: Executor, alert: Callable[[str], None]) -> None:
        self.ex, self.insts, self.cfg = ex, insts, cfg
        self.executor, self.alert = executor, alert
        self._last_rank: dict[str, int] = {}

    def check(self, state: CarryState) -> list[str]:
        actions: list[str] = []
        for coin in list(state.legs):
            leg = state.legs[coin]
            inst = self.insts[coin]

            rank = self.ex.swap_adl_rank(coin)
            if rank >= self.cfg.adl_alert_rank and self._last_rank.get(coin) != rank:
                self.alert(f"ADL rank {rank} on {coin} (alert only, hedge intact)")
            self._last_rank[coin] = rank

            contracts = self.ex.swap_contracts(coin)
            spot = self.ex.spot_balance(coin)
            status = check_hedge(spot, contracts, inst, self.cfg.drift_tol)
            shrunk = contracts < leg.contracts - 1e-9
            if status == HedgeStatus.OK and not shrunk:
                continue

            self.alert(f"[CRITICAL] hedge broken on {coin}: status={status.value}, "
                       f"perp {contracts} vs recorded {leg.contracts}, spot {spot}; flattening")
            if self.executor.close(coin, leg):
                del state.legs[coin]
                actions.append(f"closed {coin} (hedge broken)")
            else:
                self.alert(f"[CRITICAL] FAILED to flatten {coin}; manual action needed")
                actions.append(f"FAILED to close {coin}")
            state.mode = "HALTED"
            state.halted_reason = f"hedge broken on {coin}"
        return actions
